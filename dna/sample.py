from __future__ import annotations
from typing import Any, List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int
from jax import lax, random


@eqx.filter_jit
def sample_tokens(
    model: Any,
    prompt_ids: Int[Array, "B T0"],
    prompt_lens: Int[Array, "B"],
    *,
    key: Array,
    max_new: int,
    temperature: float = 0.8,
    pad_id: int = 50256,
    eos_id: int = 50256,
) -> Int[Array, "B T_total"]:
    B, T0 = prompt_ids.shape
    total_len = T0 + max_new
    tokens = jnp.concatenate(
        [prompt_ids, jnp.full((B, max_new), pad_id, dtype=jnp.int32)], axis=1
    )
    cur_pos = prompt_lens.astype(jnp.int32)
    done = jnp.zeros((B,), dtype=bool)
    positions: Int[Array, "1 T_total"] = jnp.arange(total_len, dtype=jnp.int32)[None, :]

    def step_fn(carry, _):
        tokens_cur, pos_cur, done_cur, key_cur = carry
        key_cur, subkey = random.split(key_cur)
        attn_mask = positions < pos_cur[:, None]
        logits = model(tokens_cur, attn_mask, key=subkey, inference=True)
        last_idx = jnp.maximum(pos_cur, 1) - 1
        logits_last = logits[jnp.arange(B), last_idx]
        key_sample = random.split(key_cur, B + 1)
        key_cur = key_sample[0]
        sample_keys = key_sample[1:]

        def sample_logits(logits_row):
            scaled = logits_row / jnp.maximum(temperature, 1e-6)
            return random.categorical(sample_keys[0], scaled)

        def greedy_logits(logits_row):
            return jnp.argmax(logits_row, axis=-1)

        next_tok = lax.cond(
            temperature > 0,
            lambda x: jax.vmap(sample_logits)(x),
            lambda x: jax.vmap(greedy_logits)(x),
            logits_last,
        )
        next_tok = jnp.where(done_cur, pad_id, next_tok)
        tokens_cur = tokens_cur.at[jnp.arange(B), pos_cur].set(next_tok)
        new_done = done_cur | (next_tok == eos_id)
        new_pos = jnp.where(new_done, pos_cur, pos_cur + 1)
        return (tokens_cur, new_pos, new_done, key_cur), None

    (tokens_out, _, _, _), _ = lax.scan(
        step_fn, (tokens, cur_pos, done, key), None, length=max_new
    )
    return tokens_out


def generate(
    params: Any,
    static: Any,
    tokenizer: Any,
    prompts: List[str],
    key: Array,
    *,
    max_new: int,
    temperature: float,
    pad_id: int = 50256,
    eos_id: int = 50256,
) -> List[str]:
    if len(prompts) == 0:
        return []
    pad_tok = tokenizer.pad_token_id or pad_id
    encodings = [tokenizer.encode(p) for p in prompts]
    lens = np.array([len(e) for e in encodings], dtype=np.int32)
    T0 = int(lens.max(initial=1))
    batch_ids = np.full((len(prompts), T0), pad_tok, dtype=np.int32)
    for i, enc in enumerate(encodings):
        batch_ids[i, : len(enc)] = enc
    ids = jnp.array(batch_ids, dtype=jnp.int32)
    lens = jnp.array(lens, dtype=jnp.int32)
    model = eqx.combine(params, static)
    out_ids = sample_tokens(
        model,
        ids,
        lens,
        key=key,
        max_new=max_new,
        temperature=temperature,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    out_ids_host = jax.device_get(out_ids)
    return [
        tokenizer.decode(out_ids_host[i], skip_special_tokens=True)
        for i in range(len(prompts))
    ]
