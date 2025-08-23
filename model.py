# model.py
from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from jaxtyping import Array, Float, Int, Bool


# ----------------- dtypes -----------------
bf16 = jnp.bfloat16  # activations + stored weights
f32 = jnp.float32  # compute/accumulation & logits


# ---------------------------------------------------------------------
# core layers (no vmaps; support leading dims)
# ---------------------------------------------------------------------
class Linear(eqx.Module):
    weight: Float[Array, "in out"]

    def __init__(self, in_dim: int, out_dim: int, *, key):
        # GPT-style init: N(0, 0.02) truncated to +/-2 sigmas, stored as bf16
        w = jax.random.truncated_normal(
            key, lower=-2.0, upper=2.0, shape=(in_dim, out_dim)
        )
        self.weight = (w * 0.02).astype(bf16)

    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
        # # Compute in fp32 for stability/throughput, cast back to bf16
        y32 = jnp.matmul(x.astype(f32), self.weight.astype(f32))
        return y32.astype(bf16)


class Embedding(eqx.Module):
    weight: Float[Array, "V D"]

    def __init__(self, vocab: int, dim: int, *, key):
        w = jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=(vocab, dim))
        self.weight = (w * 0.02).astype(bf16)

    def __call__(self, ids: Int[Array, "..."]) -> Float[Array, "... D"]:
        # jnp.take supports leading dims
        return jnp.take(self.weight, ids, axis=0).astype(bf16)


class RMSNorm(eqx.Module):
    weight: Float[Array, "D"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones((dim,), dtype=bf16)
        self.eps = float(eps)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x32 = x.astype(f32)
        var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
        y = x32 * jax.lax.rsqrt(var + self.eps)  # fp32 norm
        return (y.astype(bf16) * self.weight).astype(bf16)


class Dropout(eqx.Module):
    rate: float = eqx.field(static=True)

    def __init__(self, rate: float = 0.0):
        self.rate = float(rate)

    def __call__(
        self, x: Float[Array, "..."], *, key, inference: bool
    ) -> Float[Array, "..."]:
        if inference or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        m = jax.random.bernoulli(key, keep, x.shape)
        # Scale by keep prob (in dtype of x)
        return jnp.where(m, x / jnp.asarray(keep, dtype=x.dtype), jnp.zeros_like(x))


# ----------------- rope -----------------
def rope_cos_sin(T: int, dim: int, base: float = 10_000.0):
    assert dim % 2 == 0
    pos = jnp.arange(T, dtype=jnp.float32)[:, None]
    idx = jnp.arange(0, dim, 2, dtype=jnp.float32)[None]
    inv = base ** (-idx / dim)
    ang = pos * inv
    cos = jnp.repeat(jnp.cos(ang), 2, axis=1)
    sin = jnp.repeat(jnp.sin(ang), 2, axis=1)
    return cos, sin  # keep fp32; cast inside Attention when needed


def _rotate_half(x: Float[Array, "... h"]) -> Float[Array, "... h"]:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rope(
    x: Float[Array, "T N H"], cos: Float[Array, "T H"], sin: Float[Array, "T H"]
) -> Float[Array, "T N H"]:
    # TNH * TH -> broadcast over heads
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


# ----------------- layers -----------------
class FeedForward(eqx.Module):
    up: Linear
    gate: Linear
    down: Linear
    drop: Dropout

    def __init__(self, d_model: int, ff_mult: int, dropout: float, *, key):
        d_ff = d_model * ff_mult
        k1, k2, k3 = jax.random.split(key, 3)
        self.up = Linear(d_model, d_ff, key=k1)
        self.gate = Linear(d_model, d_ff, key=k2)
        self.down = Linear(d_ff, d_model, key=k3)
        self.drop = Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "... D"],
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "... D"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        # Compute in fp32 then cast to bf16 between layers
        gate32 = self.gate(x).astype(f32)  # (..., d_ff)
        up32 = self.up(x).astype(f32)  # (..., d_ff)
        h32 = jnn.silu(gate32) * up32  # (..., d_ff) fp32
        h = h32.astype(bf16)
        h = self.drop(h, key=k1, inference=inference)
        y = self.down(h).astype(bf16)  # (..., d_model) (down returns bf16)
        y = self.drop(y, key=k2, inference=inference)
        return y.astype(bf16)


class Attention(eqx.Module):
    qkv: Linear
    out: Linear
    drop: Dropout
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, dropout: float, *, key):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        k1, k2 = jax.random.split(key)
        self.qkv = Linear(d_model, 3 * d_model, key=k1)
        self.out = Linear(d_model, d_model, key=k2)
        self.drop = Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "T D"],  # sequence-first
        cos: Float[Array, "T d_head"],
        sin: Float[Array, "T d_head"],
        mask: Optional[Bool[Array, "T"]] = None,  # True = keep
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "T D"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        T, D = x.shape

        # Project to qkv (keep bf16 storage but upcast for attention math)
        qkv = self.qkv(x).astype(jnp.float32)  # <— fp32 here
        qkv = qkv.reshape(T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (T, N, H)

        # RoPE in fp32 as well
        q = apply_rope(
            q.astype(jnp.float32), cos.astype(jnp.float32), sin.astype(jnp.float32)
        )
        k = apply_rope(
            k.astype(jnp.float32), cos.astype(jnp.float32), sin.astype(jnp.float32)
        )
        v = v.astype(jnp.float32)

        # Build (N, T, T) boolean mask. Add a safety “at least self” guard
        attn_mask = None
        if mask is not None:
            m = mask.astype(bool)  # (T,)
            pad2d = m[:, None] & m[None, :]  # (T, T)
            pad2d = pad2d | jnp.eye(T, dtype=bool)  # <— guarantee non-empty rows
            attn_mask = jnp.broadcast_to(pad2d, (self.n_heads, T, T))

        # Use the default (XLA) path; no fused/cuDNN kernel
        out = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=attn_mask,
            is_causal=True,
            # implementation="cudnn",                     # <— key change
        )  # (T, N, H) fp32

        out = out.reshape(T, D).astype(jnp.float32)
        out = self.out(out).astype(jnp.bfloat16)
        out = self.drop(out, key=k2, inference=inference)
        return out.astype(jnp.bfloat16)


class TransformerBlock(eqx.Module):
    ln1: RMSNorm
    attn: Attention
    ln2: RMSNorm
    ff: FeedForward

    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int, dropout: float, *, key
    ):
        k1, k2 = jax.random.split(key)
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.attn = Attention(d_model, n_heads, dropout, key=k1)
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.ff = FeedForward(d_model, ff_mult, dropout, key=k2)

    def __call__(
        self,
        x: Float[Array, "T D"],
        cos: Float[Array, "T d_head"],
        sin: Float[Array, "T d_head"],
        mask: Optional[Bool[Array, "T"]] = None,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "T D"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        # Attention block with residual
        residual = x
        x = self.ln1(x)
        x = self.attn(x, cos, sin, mask, key=k1, inference=inference)
        x = (residual.astype(bf16) + x).astype(bf16)

        # FF block with residual
        residual = x
        x = self.ln2(x)
        x = self.ff(x, key=k2, inference=inference)
        x = (residual.astype(bf16) + x).astype(bf16)
        return x


# ----------------- model -----------------
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
        vocab_size: int = 50_257,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        ff_mult: int = 4,
        dropout: float = 0.1,
        rope_base: float = 10_000.0,
        *,
        key,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_base = rope_base

        keys = jax.random.split(key, n_layers + 2)
        self.embed = Embedding(vocab_size, d_model, key=keys[0])
        self.drop = Dropout(dropout)
        self.blocks = tuple(
            TransformerBlock(d_model, n_heads, ff_mult, dropout, key=keys[i + 1])
            for i in range(n_layers)
        )
        self.ln_out = RMSNorm(d_model, eps=1e-5)

    def __call__(
        self,
        ids: Int[Array, "T"],
        mask: Optional[Bool[Array, "T"]] = None,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "T V"]:
        if key is None:
            key = jax.random.PRNGKey(0)

        T = ids.shape[0]
        d_head = self.d_model // self.n_heads

        # Embedding + dropout (BF16 activations)
        x = self.embed(ids).astype(bf16)  # (T, D)
        k_drop, key = jax.random.split(key)
        x = self.drop(x, key=k_drop, inference=inference)

        # RoPE caches (BF16) for head dim
        cos, sin = rope_cos_sin(T, d_head, self.rope_base)

        # Blocks
        for block in self.blocks:
            k_blk, key = jax.random.split(key)
            x = block(x, cos, sin, mask, key=k_blk, inference=inference)

        # Final norm (BF16 → FP32) and tied output projection to vocab
        x = self.ln_out(x).astype(f32)  # (T, D) -> fp32
        logits = jnp.matmul(x, self.embed.weight.astype(f32).T)  # (T, V) fp32
        return logits  # keep fp32 for loss
