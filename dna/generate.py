# dna/generate.py
from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, List, Dict, Any


# ============================================================================
# Autoregressive generation (full-context per step; causal masking)
# ============================================================================


def sample(
    model: eqx.Module,
    prompt_ids: Int[Array, "T0"],
    max_new_tokens: int,
    *,
    key,
    temperature: float = 0.8,
    greedy: bool = False,
    pad_id: int = 0,
    eos_id: Optional[int] = None,
    router_temp: float = 1.0,
    select_temp: Optional[float] = None,
    gumbel_tau: float = 1.0,
) -> Int[Array, "T"]:
    """Autoregressive sampler."""
    if eos_id is None:
        eos_id = int(pad_id)

    prompt_len = int(prompt_ids.shape[0])
    total_len = int(prompt_len + max_new_tokens)

    # ---- Allocate output token buffer and write prompt ----
    tokens: Int[Array, "T"] = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)

    # ---- Greedy override if temperature <= 0 ----
    force_greedy = greedy or (temperature <= 0.0)

    def step(carry, _):
        toks, cur, done, k = carry
        k, subkey = jax.random.split(k)

        # Valid positions are strictly < cur
        attn_mask = jnp.arange(total_len, dtype=jnp.int32) < cur  # (T,)

        logits, _stats = model(
            toks,
            key=subkey,
            inference=True,
            mask=attn_mask,
            gumbel_tau=gumbel_tau,  # ignored at inference=True
            router_temp=router_temp,
            select_temp=select_temp,
        )

        vocab = logits.shape[-1]
        cur_idx = cur - jnp.asarray(1, jnp.int32)
        last_logits = jax.lax.dynamic_slice(logits, (cur_idx, 0), (1, vocab))[0]

        def pick_greedy(lg):
            return jnp.argmax(lg, axis=-1).astype(jnp.int32)

        def pick_sample(rng, lg):
            scaled = lg / jnp.clip(temperature, 1e-6, None)
            return jax.random.categorical(rng, scaled).astype(jnp.int32)

        next_tok = jax.lax.cond(
            force_greedy,
            lambda _: pick_greedy(last_logits),
            lambda rng: pick_sample(rng, last_logits),
            operand=subkey,
        )

        # After EOS, keep writing PAD
        next_tok = jax.lax.select(done, jnp.asarray(pad_id, jnp.int32), next_tok)

        toks = toks.at[cur].set(next_tok)
        new_done = done | (next_tok == eos_id)
        return (toks, cur + 1, new_done, k), None

    (tokens, _, _, _), _ = jax.lax.scan(
        step,
        (tokens, jnp.asarray(prompt_len, jnp.int32), jnp.asarray(False), key),
        None,
        length=int(max_new_tokens),
    )

    return tokens


def generate(
    model,
    tok,
    *,
    key: jax.Array,
    gen_len: int = 100,
    per_prompt: int = 1,
    router_temp,
    select_temp,
    gumbel_tau,
    prompts: Optional[List[str]] = None,
    n_examples: int = 5,
    temperature: float = 0.8,
    greedy: bool = False,
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []

    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    for p in prompts[:n_examples]:
        prompt_ids = jnp.array(tok.encode(p), dtype=jnp.int32)
        key, *subs = jax.random.split(key, per_prompt + 1)
        subs = jnp.stack(subs)

        @jax.vmap
        def _sample(k):
            return sample(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=gen_len,
                temperature=temperature,
                key=k,
                router_temp=router_temp,
                select_temp=select_temp,
                gumbel_tau=gumbel_tau,
                greedy=greedy,
                pad_id=pad_id,
                eos_id=eos_id,
            )

        toks = _sample(subs)  
        prompt_result = {"prompt": p, "completions": []}

        for seq in jax.device_get(toks):
            seq = list(map(int, list(seq)))
            if eos_id in seq:
                idx = seq.index(eos_id)
                seq = seq[: idx + 1]
                stopped_eos = True
            else:
                stopped_eos = False

            text = tok.decode(seq, skip_special_tokens=True)
            text_with_special = tok.decode(seq, skip_special_tokens=False)

            prompt_result["completions"].append(
                {
                    "tokens": seq,
                    "text": text,
                    "text_with_special": text_with_special,
                    "length": len(seq),
                    "stopped_eos": bool(stopped_eos),
                }
            )

        results.append(prompt_result)

    return results
