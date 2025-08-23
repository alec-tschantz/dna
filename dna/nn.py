from __future__ import annotations
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxtyping import Array, Bool, Float, Int

BF16 = jnp.bfloat16
FP32 = jnp.float32


class Linear(eqx.Module):
    weight: Float[Array, "in_dim out_dim"]

    def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array):
        w = jax.random.truncated_normal(
            key, lower=-2.0, upper=2.0, shape=(in_dim, out_dim)
        )
        self.weight = (w * 0.02).astype(BF16)

    def __call__(self, x: Float[Array, "... in_dim"]) -> Float[Array, "... out_dim"]:
        y = jnp.matmul(x.astype(FP32), self.weight.astype(FP32))
        return y.astype(BF16)


class Embedding(eqx.Module):
    weight: Float[Array, "vocab_size dim"]

    def __init__(self, vocab_size: int, dim: int, *, key: jax.Array):
        w = jax.random.truncated_normal(
            key, lower=-2.0, upper=2.0, shape=(vocab_size, dim)
        )
        self.weight = (w * 0.02).astype(BF16)

    def __call__(self, ids: Int[Array, "..."]) -> Float[Array, "... dim"]:
        return jnp.take(self.weight, ids, axis=0).astype(BF16)


class RMSNorm(eqx.Module):
    weight: Float[Array, "dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones((dim,), dtype=BF16)
        self.eps = eps

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        x_fp32 = x.astype(FP32)
        var = jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True)
        normed = x_fp32 * jax.lax.rsqrt(var + self.eps)
        return (normed.astype(BF16) * self.weight).astype(BF16)


class Dropout(eqx.Module):
    rate: float = eqx.field(static=True)

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def __call__(
        self, x: Float[Array, "..."], *, key: jax.Array, inference: bool
    ) -> Float[Array, "..."]:
        if inference or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
        return jnp.where(
            mask, x / jnp.array(keep_prob, dtype=x.dtype), jnp.zeros_like(x)
        )


def rope_angles(
    seq_len: int, dim: int, base: float = 10000.0
) -> Tuple[Float[Array, "seq_len dim"], Float[Array, "seq_len dim"]]:
    half = dim // 2
    pos = jnp.arange(seq_len, dtype=FP32)[:, None]
    idx = jnp.arange(half, dtype=FP32)[None, :]
    theta = base ** (-idx / half)
    angles = pos * theta
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    cos = jnp.repeat(cos, 2, axis=1)
    sin = jnp.repeat(sin, 2, axis=1)
    return cos, sin


def _rotate_half(x: Float[Array, "... d"]):
    half = x.shape[-1] // 2
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rope(
    x: Float[Array, "... seq_len num_heads head_dim"],
    cos: Float[Array, "seq_len head_dim"],
    sin: Float[Array, "seq_len head_dim"],
) -> Float[Array, "... seq_len num_heads head_dim"]:
    orig_shape = x.shape
    T, H = orig_shape[-3], orig_shape[-1]
    x_flat = x.reshape(-1, T, H)
    x_rotated = x_flat * cos[None, :, :] + _rotate_half(x_flat) * sin[None, :, :]
    return x_rotated.reshape(orig_shape)


class FeedForward(eqx.Module):
    w_up: Linear
    w_gate: Linear
    w_down: Linear
    drop: Dropout

    def __init__(self, d_model: int, ff_mult: int, dropout: float, *, key: jax.Array):
        d_ff = d_model * ff_mult
        k1, k2, k3 = jax.random.split(key, 3)
        self.w_up = Linear(d_model, d_ff, key=k1)
        self.w_gate = Linear(d_model, d_ff, key=k2)
        self.w_down = Linear(d_ff, d_model, key=k3)
        self.drop = Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "... d_model"],
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "... d_model"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key, 2)
        gate_out = self.w_gate(x).astype(FP32)
        up_out = self.w_up(x).astype(FP32)
        ff_out = jnn.silu(gate_out) * up_out
        ff_out = self.drop(ff_out.astype(BF16), key=k1, inference=inference)
        y = self.w_down(ff_out).astype(BF16)
        y = self.drop(y, key=k2, inference=inference)
        return y.astype(BF16)


class Attention(eqx.Module):
    w_qkv: Linear
    w_out: Linear
    drop: Dropout
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, dropout: float, *, key: jax.Array):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        k_qkv, k_out = jax.random.split(key, 2)
        self.w_qkv = Linear(d_model, 3 * d_model, key=k_qkv)
        self.w_out = Linear(d_model, d_model, key=k_out)
        self.drop = Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "... seq_len d_model"],
        cos: Float[Array, "seq_len head_dim"],
        sin: Float[Array, "seq_len head_dim"],
        mask: Optional[Bool[Array, "... seq_len"]] = None,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> Float[Array, "... seq_len d_model"]:
        if key is None:
            key = jax.random.PRNGKey(0)
        k_attn, k_out = jax.random.split(key, 2)
        B, T, D = x.shape if x.ndim == 3 else (1, *x.shape)
        qkv = self.w_qkv(x).astype(FP32)
        qkv = qkv.reshape(*qkv.shape[:-1], 3, self.n_heads, self.head_dim)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_mask = None
        if mask is not None:
            mask_bool = mask.astype(bool)
            pad_mask = mask_bool[..., None, :] & mask_bool[..., :, None]
            pad_mask = pad_mask | jnp.eye(T, dtype=bool)[None, :, :]
            pad_mask = pad_mask[:, None, :, :]
            attn_mask = pad_mask
        attn_out = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            bias=None,
            mask=attn_mask,
            is_causal=True,
        )
        attn_out = attn_out.reshape(*attn_out.shape[:-2], D).astype(FP32)
        out = self.w_out(attn_out).astype(BF16)
        out = self.drop(out, key=k_out, inference=inference)
        return out.astype(BF16)
