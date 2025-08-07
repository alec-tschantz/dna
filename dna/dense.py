import math
from typing import Tuple, Dict, Any

import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
import jax.nn as jnn

from dna.nn import Embedding, Dropout, RMSNorm, Attention, FeedForward, rope_cos_sin


class Dense(eqx.Module):
    embed: Embedding
    dropout: Dropout
    layers: Tuple[Tuple[Attention, FeedForward], ...]
    ln: RMSNorm
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
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
        keys = jax.random.split(k_layers, n_layers * 2)
        self.layers = tuple(
            (
                Attention(d_model, n_heads, dropout, key=keys[2 * i]),
                FeedForward(d_model, mlp_mult, dropout, key=keys[2 * i + 1]),
            )
            for i in range(n_layers)
        )
        self.ln = RMSNorm(d_model)
        self.n_heads = n_heads
        self.rope_base = rope_base

    @eqx.filter_jit
    def __call__(self, ids, *, key, inference: bool):
        T = ids.shape[0]
        h = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)
        cos, sin = rope_cos_sin(
            T, self.embed.weight.shape[1] // self.n_heads, self.rope_base
        )
        for attn, mlp in self.layers:
            key, sa, sm = jax.random.split(key, 3)
            h = h + attn(h, cos, sin, key=sa, inference=inference)
            h = h + mlp(h, cos, sin, key=sm, inference=inference)
        h = jax.vmap(self.ln)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        return logits, {}
