import math
from typing import Tuple, Dict, Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from dna.modules import (
    Embedding,
    Dropout,
    RMSNorm,
    Attention,
    FeedForward,
    rope_cos_sin,
)


class Dense(eqx.Module):
    embed: Embedding
    dropout: Dropout
    layers: Tuple[Tuple[Attention, FeedForward], ...]
    ln_out: RMSNorm
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_mult: int,
        dropout: float,
        rope_base: float,
        key,
    ):
        k_emb, k_layers = jax.random.split(key)
        self.embed = Embedding(vocab, d_model, key=k_emb)
        self.dropout = Dropout(dropout)

        n_pairs = int(n_layers // 2)
        keys = jax.random.split(k_layers, max(0, n_pairs) * 2)

        self.layers = tuple(
            (
                Attention(d_model, n_heads, dropout, key=keys[2 * i]),
                FeedForward(d_model, mlp_mult, dropout, key=keys[2 * i + 1]),
            )
            for i in range(n_pairs)
        )

        self.ln_out = RMSNorm(d_model)
        self.n_heads = int(n_heads)
        self.rope_base = float(rope_base)

    @eqx.filter_jit
    def __call__(
        self,
        ids,
        *,
        key,
        inference: bool,
        mask: Optional[jnp.ndarray] = None,
        return_stats: bool = False, 
        **kwargs,
    ):
        # ---- token mask setup ----
        T = ids.shape[0]
        token_mask = jnp.ones((T,), dtype=bool) if mask is None else mask.astype(bool)

        # ---- embeddings + dropout ----
        h = jax.vmap(self.embed)(ids)  # (T, d)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        # ---- RoPE ----
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos, sin = rope_cos_sin(T, d_h, self.rope_base)  # each (T, d_h)

        # ---- stacked transformer blocks ----
        for attn, mlp in self.layers:
            key, k_attn, k_ff = jax.random.split(key, 3)

            a_out = attn(h, (cos, sin), token_mask, key=k_attn, inference=inference)
            h = h + a_out
            h = jnp.where(token_mask[:, None], h, 0.0) + jnp.where(
                token_mask[:, None], 0.0, jax.lax.stop_gradient(h - a_out)
            )

            m_out = mlp(h, (cos, sin), token_mask, key=k_ff, inference=inference)
            h = h + m_out
            h = jnp.where(token_mask[:, None], h, 0.0) + jnp.where(
                token_mask[:, None], 0.0, jax.lax.stop_gradient(h - m_out)
            )

        # ---- final norm + tied unembedding ----
        h = jax.vmap(self.ln_out)(h)  # (T, d)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)  # (T, V)

        return logits, ()
