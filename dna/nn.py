import math
from typing import Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


def rope_cos_sin(T: int, dim: int, base: float = 10_000.0):
    pos = jnp.arange(T)[:, None]
    idx = jnp.arange(0, dim, 2)[None]
    inv = base ** (-idx / dim)
    ang = pos * inv
    cos = jnp.repeat(jnp.cos(ang), 2, axis=1)
    sin = jnp.repeat(jnp.sin(ang), 2, axis=1)
    return cos, sin


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class Embedding(eqx.Module):
    weight: jnp.ndarray

    def __init__(self, vocab: int, dim: int, *, key):
        self.weight = (
            jax.random.truncated_normal(key, lower=-2, upper=2, shape=(vocab, dim))
            * 0.02
        )

    def __call__(self, ids):
        return jnp.take(self.weight, ids, axis=0)


class RMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        var = jnp.mean(x**2, axis=-1, keepdims=True)
        return self.weight * x * jax.lax.rsqrt(var + self.eps)


class Dropout(eqx.Module):
    rate: float = eqx.field(static=True)

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def __call__(self, x, *, key, inference: bool):
        if inference or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        mask = jax.random.bernoulli(key, keep, x.shape)
        return jnp.where(mask, x / keep, 0)


class Identity(eqx.Module):
    def __call__(self, x, *_, **__):
        return jnp.zeros_like(x)


class Attention(eqx.Module):
    ln: RMSNorm
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    o: eqx.nn.Linear
    dropout: Dropout
    n_h: int = eqx.field(static=True)
    d_h: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, dropout: float, *, key):
        self.n_h, self.d_h = n_heads, d_model // n_heads
        self.ln = RMSNorm(d_model)
        self.dropout = Dropout(dropout)
        k_q, k_k, k_v, k_o = jax.random.split(key, 4)
        self.q = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_q)
        self.k = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_k)
        self.v = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_v)
        self.o = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_o)

    def __call__(self, x, cos, sin, *, key, inference: bool):
        k_attn, k_out = jax.random.split(key)
        h = self.ln(x)
        T = h.shape[0]
        q = jax.vmap(self.q)(h).reshape(T, self.n_h, self.d_h).transpose(1, 0, 2)
        k = jax.vmap(self.k)(h).reshape(T, self.n_h, self.d_h).transpose(1, 0, 2)
        v = jax.vmap(self.v)(h).reshape(T, self.n_h, self.d_h).transpose(1, 0, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        scores = jnp.einsum("hqd,hkd->hqk", q, k) / math.sqrt(self.d_h)
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None]
        scores = jnp.where(causal, scores, -1e30)
        probs = jnn.softmax(scores, axis=-1)
        probs = self.dropout(probs, key=k_attn, inference=inference)
        out = jnp.einsum("hqk,hkd->hqd", probs, v).transpose(1, 0, 2).reshape(T, -1)
        out = jax.vmap(self.o)(out)
        out = self.dropout(out, key=k_out, inference=inference)
        return x + out


class FeedForward(eqx.Module):
    ln: RMSNorm
    up: eqx.nn.Linear
    gate: eqx.nn.Linear
    down: eqx.nn.Linear
    dropout: Dropout

    def __init__(self, d_model: int, mult: int, dropout: float, *, key):
        self.ln = RMSNorm(d_model)
        self.dropout = Dropout(dropout)
        k_up, k_gate, k_down = jax.random.split(key, 3)
        d_inner = d_model * mult
        self.up = eqx.nn.Linear(d_model, d_inner, use_bias=False, key=k_up)
        self.gate = eqx.nn.Linear(d_model, d_inner, use_bias=False, key=k_gate)
        self.down = eqx.nn.Linear(d_inner, d_model, use_bias=False, key=k_down)

    def __call__(self, x, *_unused, key, inference: bool):
        k_mid, k_out = jax.random.split(key)
        h = self.ln(x)
        h = jnn.silu(jax.vmap(self.gate)(h)) * jax.vmap(self.up)(h)
        h = self.dropout(h, key=k_mid, inference=inference)
        h = jax.vmap(self.down)(h)
        h = self.dropout(h, key=k_out, inference=inference)
        return x + h
