# --- sample.py ---

from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


def sample_tokens(
    model,
    prompt_ids: Int[Array, "B T0"],
    prompt_lens: Int[Array, "B"],
    *,
    key: jax.Array,
    max_new: int = 200,
    temperature: float = 0.8,
    pad_id: int = 50256,
    eos_id: int = 50256,
) -> Int[Array, "B T_total"]:
    """
    Generate tokens using temperature sampling (no device parallelism).
    Returns tokens of shape [B, T0 + max_new].
    """
    B, T0 = prompt_ids.shape
    total_len = T0 + max_new

    # Output buffer: [B, T0 + max_new], with prompt already placed.
    tokens = jnp.concatenate(
        [prompt_ids, jnp.full((B, max_new), pad_id, dtype=jnp.int32)],
        axis=1,
    )

    # Positions and done flags
    current_pos = prompt_lens.astype(jnp.int32)          # [B]
    is_done     = jnp.zeros_like(prompt_lens, dtype=bool)  # [B]

    positions = jnp.arange(total_len, dtype=jnp.int32)[None, :]

    def step(carry, _):
        tokens, current_pos, is_done, key = carry
        key, subkey = jax.random.split(key)

        # Mask is True for tokens < current_pos (causal prefix per batch element)
        attn_mask = positions < current_pos[:, None]      # [B, T_total]

        # Forward pass for each sequence in the batch
        # model: (tok_seq[T_total], mask[T_total]) -> logits[T_total, V]
        keys = jax.random.split(subkey, B)
        logits = jax.vmap(lambda ts, m, k: model(ts, m, key=k, inference=True))(tokens, attn_mask, keys)  # [B, T_total, V]

        # Get logits at the last valid position (pos-1, guarded for pos=0)
        last_idx = jnp.maximum(current_pos, 1) - 1                       # [B]
        last_logits = logits[jnp.arange(B), last_idx]                    # [B, V]

        # Sample or greedy based on temperature (compile-time boolean)
        def _sample(ops):
            lg, ks = ops
            scaled = lg / jnp.maximum(temperature, 1e-6)
            return jax.vmap(lambda _lg, _k: jax.random.categorical(_k, _lg))(scaled, ks)

        def _greedy(ops):
            lg, ks = ops
            return jnp.argmax(lg, axis=-1)

        subkeys = jax.random.split(key, B + 1)
        key = subkeys[0]
        sample_keys = subkeys[1:]

        next_tokens = jax.lax.cond(
            temperature > 0.0, _sample, _greedy, operand=(last_logits, sample_keys)
        )

        # Respect EOS
        next_tokens = jnp.where(is_done, pad_id, next_tokens)

        # Write next token at current position
        tokens = tokens.at[jnp.arange(B), current_pos].set(next_tokens)

        # Update positions and done flags
        new_done = is_done | (next_tokens == eos_id)
        new_pos  = jnp.where(new_done, current_pos, current_pos + 1)

        return (tokens, new_pos, new_done, key), None

    (tokens, _, _, _), _ = jax.lax.scan(
        step, (tokens, current_pos, is_done, key), None, length=max_new
    )

    return tokens
