from __future__ import annotations
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .nn import Attention, Dropout, Embedding, FeedForward, RMSNorm, rope_angles

BF16 = jnp.bfloat16
FP32 = jnp.float32


def _run_block(block, x, cos, sin, mask, key, inference):
    return block(x, cos, sin, mask, key=key, inference=inference)


_run_block_ckpt = jax.checkpoint(_run_block, static_argnums=(6,))


class TransformerBlock(eqx.Module):
    ln1: RMSNorm
    attn: Attention
    ln2: RMSNorm
    ff: FeedForward

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        *,
        key: jax.Array,
    ):
        k_attn, k_ff = jax.random.split(key, 2)
        self.ln1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout, key=k_attn)
        self.ln2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, ff_mult, dropout, key=k_ff)

    def __call__(
        self,
        x: Float[Array, "B T D"],
        cos: Float[Array, "T H"],
        sin: Float[Array, "T H"],
        mask: Optional[Bool[Array, "B T"]] = None,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B T D"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k_attn, k_ff = jax.random.split(key, 2)
        h = self.ln1(x)
        attn_out = self.attn(h, cos, sin, mask, key=k_attn, inference=inference)
        x = (x + attn_out).astype(BF16)
        h2 = self.ln2(x)
        ff_out = self.ff(h2, key=k_ff, inference=inference)
        x = (x + ff_out).astype(BF16)
        return x


class Transformer(eqx.Module):
    embed: Embedding
    drop: Dropout
    blocks: Tuple[TransformerBlock, ...]
    ln_out: RMSNorm
    vocab_size: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        rope_base: float,
        *,
        key: jax.Array,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_base = rope_base
        k_embed, k_blocks = jax.random.split(key, 2)
        self.embed = Embedding(vocab_size, d_model, key=k_embed)
        self.drop = Dropout(dropout)
        block_keys = jax.random.split(k_blocks, n_layers)
        self.blocks = tuple(
            TransformerBlock(d_model, n_heads, ff_mult, dropout, key=k)
            for k in block_keys
        )
        self.ln_out = RMSNorm(d_model)

    def __call__(
        self,
        ids: Int[Array, "B T"],
        mask: Optional[Bool[Array, "B T"]] = None,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "B T vocab_size"]:
        B, T = ids.shape
        x = self.embed(ids)
        k_drop, key = jax.random.split(key)
        x = self.drop(x, key=k_drop, inference=inference)
        cos, sin = rope_angles(T, self.d_model // self.n_heads, base=self.rope_base)
        cos = cos.astype(FP32)
        sin = sin.astype(FP32)
        for block in self.blocks:
            k_blk, key = jax.random.split(key)
            x = _run_block_ckpt(block, x, cos, sin, mask, key, inference)
        x = self.ln_out(x).astype(FP32)
        logits = x @ self.embed.weight.T.astype(FP32)
        return logits
