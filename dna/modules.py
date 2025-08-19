
# ================================ modules.py ================================
from __future__ import annotations
from typing import Tuple, Optional, List
from jaxtyping import Float, Int, Bool, Array

import math
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


# RoPE utilities
def rope_cos_sin(
    T: int, dim: int, base: float = 10_000.0
) -> Tuple[Float[Array, "T dim"], Float[Array, "T dim"]]:
    """Optimized Rotary Position Embedding computation."""
    assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
    
    pos = jnp.arange(T, dtype=jnp.float32)[:, None]
    idx = jnp.arange(0, dim, 2, dtype=jnp.float32)[None, :]
    inv_freq = base ** (-idx / dim)
    angles = pos * inv_freq
    
    # Efficient repeat using broadcasting
    cos = jnp.repeat(jnp.cos(angles), 2, axis=1)
    sin = jnp.repeat(jnp.sin(angles), 2, axis=1)
    
    return cos, sin


def rotate_half(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    """Rotate half for RoPE application."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


# Core layers
class Embedding(eqx.Module):
    """Token embedding layer with improved initialization."""
    weight: Float[Array, "vocab dim"]
    
    def __init__(self, vocab: int, dim: int, *, key: jax.Array):
        # Better initialization for stability
        std = 0.02
        self.weight = jax.random.truncated_normal(
            key, lower=-3*std, upper=3*std, shape=(vocab, dim)
        ) * std
    
    def __call__(self, ids: Int[Array, ""]) -> Float[Array, "dim"]:
        return self.weight[ids]


class RMSNorm(eqx.Module):
    """RMS normalization with numerical stability."""
    weight: Float[Array, "dim"]
    eps: float = eqx.field(static=True)
    
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = jnp.ones((dim,))
        self.eps = eps
    
    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        ms = jnp.mean(x**2, axis=-1, keepdims=True)
        return self.weight * x * jax.lax.rsqrt(ms + self.eps)


class Dropout(eqx.Module):
    """Dropout with proper scaling."""
    rate: float = eqx.field(static=True)
    
    def __init__(self, rate: float = 0.0):
        self.rate = float(rate)
    
    def __call__(
        self, x: Float[Array, "..."], *, key: jax.Array, inference: bool
    ) -> Float[Array, "..."]:
        if inference or self.rate <= 0.0:
            return x
        keep_prob = 1.0 - self.rate
        mask = jax.random.bernoulli(key, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)


# Module base class
class Module(eqx.Module):
    """Base class for expert modules."""
    
    def __call__(
        self,
        x: Float[Array, "T d"],
        *args,
        key: jax.Array,
        inference: bool,
    ) -> Float[Array, "T d"]:
        raise NotImplementedError


class Identity(Module):
    """Identity module (no-op)."""
    
    def __call__(
        self,
        x: Float[Array, "T d"],
        *args,
        key: jax.Array,
        inference: bool,
    ) -> Float[Array, "T d"]:
        return x


class FeedForward(Module):
    """Optimized feedforward module with SwiGLU activation."""
    ln: RMSNorm
    up: eqx.nn.Linear
    gate: eqx.nn.Linear
    down: eqx.nn.Linear
    dropout: Dropout
    
    def __init__(self, d_model: int, mult: int, dropout: float, *, key: jax.Array):
        self.ln = RMSNorm(d_model)
        self.dropout = Dropout(dropout)
        
        d_inner = d_model * mult
        k_up, k_gate, k_down = jax.random.split(key, 3)
        
        self.up = eqx.nn.Linear(d_model, d_inner, use_bias=False, key=k_up)
        self.gate = eqx.nn.Linear(d_model, d_inner, use_bias=False, key=k_gate)
        self.down = eqx.nn.Linear(d_inner, d_model, use_bias=False, key=k_down)
    
    def __call__(
        self,
        x: Float[Array, "T d"],
        *args,
        key: jax.Array,
        inference: bool,
    ) -> Float[Array, "T d"]:
        k_mid, k_out = jax.random.split(key)
        
        h = self.ln(x)
        # SwiGLU activation
        gate_out = jax.vmap(self.gate)(h)
        up_out = jax.vmap(self.up)(h)
        h = jnn.silu(gate_out) * up_out
        
        h = self.dropout(h, key=k_mid, inference=inference)
        h = jax.vmap(self.down)(h)
        h = self.dropout(h, key=k_out, inference=inference)
        
        return x + h


class Attention(Module):
    """Multi-head attention with RoPE and optimizations."""
    ln: RMSNorm
    qkv: eqx.nn.Linear  # Combined QKV projection
    o: eqx.nn.Linear
    dropout: Dropout
    n_h: int = eqx.field(static=True)
    d_h: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    
    def __init__(self, d_model: int, n_heads: int, dropout: float, *, key: jax.Array):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        d_h = d_model // n_heads
        assert d_h % 2 == 0, "Head dimension must be even for RoPE"
        
        self.n_h = n_heads
        self.d_h = d_h
        self.scale = 1.0 / math.sqrt(d_h)
        
        self.ln = RMSNorm(d_model)
        self.dropout = Dropout(dropout)
        
        k_qkv, k_o = jax.random.split(key)
        # Combined QKV for efficiency
        self.qkv = eqx.nn.Linear(d_model, 3 * d_model, use_bias=False, key=k_qkv)
        self.o = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_o)
    
    def __call__(
        self,
        x: Float[Array, "T d"],
        cos_sin: Tuple[Float[Array, "T d_h"], Float[Array, "T d_h"]],
        attention_mask: Optional[Bool[Array, "T"]] = None,
        *args,
        key: jax.Array,
        inference: bool,
    ) -> Float[Array, "T d"]:
        cos, sin = cos_sin
        k_attn, k_out = jax.random.split(key)
        
        h = self.ln(x)
        T = h.shape[0]
        
        # Combined QKV projection and split
        qkv = jax.vmap(self.qkv)(h)
        qkv = qkv.reshape(T, 3, self.n_h, self.d_h)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Transpose for attention: (n_h, T, d_h)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        
        # Apply RoPE
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        
        # Attention scores
        scores = jnp.einsum("hqd,hkd->hqk", q, k) * self.scale
        
        # Causal mask
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None]
        
        # Apply masks
        if attention_mask is not None:
            am = attention_mask.astype(bool)
            # Combine causal and attention masks efficiently
            valid_mask = causal_mask & am[None, None, :]
            scores = jnp.where(valid_mask, scores, -1e9)
            # Query masking
            scores = jnp.where(am[None, :, None], scores, -1e9)
        else:
            scores = jnp.where(causal_mask, scores, -1e9)
        
        # Attention weights
        probs = jnn.softmax(scores, axis=-1)
        probs = self.dropout(probs, key=k_attn, inference=inference)
        
        # Attention output
        out = jnp.einsum("hqk,hkd->hqd", probs, v)
        out = out.transpose(1, 0, 2).reshape(T, -1)
        out = jax.vmap(self.o)(out)
        out = self.dropout(out, key=k_out, inference=inference)
        
        # Masked residual
        if attention_mask is not None:
            out = jnp.where(attention_mask[:, None], out, 0.0)
        
        return x + out


class ModuleRegistry:
    """Registry for creating modules."""
    
    @staticmethod
    def create_modules(
        d_model: int,
        n_heads: int,
        n_att: int,
        n_ff: int,
        n_id: int,
        mlp_mult: int,
        dropout: float,
        key: jax.Array,
    ) -> Tuple[eqx.Module, ...]:
        """Create a collection of modules."""
        total = n_att + n_ff + n_id
        if total == 0:
            return ()
        
        keys = jax.random.split(key, n_att + n_ff)
        
        modules = []
        for i in range(n_att):
            modules.append(Attention(d_model, n_heads, dropout, key=keys[i]))
        for i in range(n_ff):
            modules.append(FeedForward(d_model, mlp_mult, dropout, key=keys[n_att + i]))
        for _ in range(n_id):
            modules.append(Identity())
        
        return tuple(modules)
    
    @staticmethod
    def create_backbone(
        d_model: int,
        n_heads: int,
        mlp_mult: int,
        dropout: float,
        backbone: Tuple[str, ...],
        key: jax.Array,
    ) -> Tuple[eqx.Module, ...]:
        """Create backbone modules from specification."""
        if not backbone:
            return ()
        
        keys = jax.random.split(key, len(backbone))
        modules = []
        
        for i, layer_type in enumerate(backbone):
            if layer_type.lower() == "attention":
                modules.append(Attention(d_model, n_heads, dropout, key=keys[i]))
            elif layer_type.lower() == "feedforward":
                modules.append(FeedForward(d_model, mlp_mult, dropout, key=keys[i]))
            else:
                raise ValueError(f"Unknown backbone layer type: {layer_type}")
        
        return tuple(modules)

