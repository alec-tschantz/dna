# dna/generate.py
from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Int, Array


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
    router_temperature: float = 1.0,
    select_temperature: Optional[float] = None,
    gumbel_tau: float = 1.0,
) -> Int[Array, "T"]:
    """Autoregressive sampler.

    Notes
    -----
    - Full forward pass at each step with a causal attention mask.
    - `inference=True` disables dropout and Gumbel in routers.
    - By default, selection temp follows router temp unless overridden.
    """
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
            inference=True,  # deterministic generation
            attention_mask=attn_mask,
            gumbel_tau=gumbel_tau,  # ignored at inference=True
            router_temperature=router_temperature,
            select_temperature=select_temperature,
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
