# ================================ dna.py ================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List, Union
from jaxtyping import Float, Int, Bool, Array

import jax
import jax.numpy as jnp
import equinox as eqx

from dna.modules import (
    Embedding, Dropout, RMSNorm, rope_cos_sin,
    Attention, FeedForward, Identity, ModuleRegistry
)

class Dense(eqx.Module):
    """Dense transformer baseline with optimizations."""
    
    embed: Embedding
    dropout: Dropout
    layers: Tuple[Tuple[Attention, FeedForward], ...]
    ln_out: RMSNorm
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
        key: jax.Array,
    ):
        # Validate inputs
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert (d_model // n_heads) % 2 == 0, "Head dimension must be even for RoPE"
        
        k_emb, k_layers = jax.random.split(key)
        self.embed = Embedding(vocab, d_model, key=k_emb)
        self.dropout = Dropout(dropout)
        
        # Create layer pairs
        n_pairs = n_layers // 2
        if n_pairs > 0:
            keys = jax.random.split(k_layers, n_pairs * 2)
            self.layers = tuple(
                (
                    Attention(d_model, n_heads, dropout, key=keys[2*i]),
                    FeedForward(d_model, mlp_mult, dropout, key=keys[2*i+1]),
                )
                for i in range(n_pairs)
            )
        else:
            self.layers = ()
        
        self.ln_out = RMSNorm(d_model)
        self.n_heads = n_heads
        self.rope_base = rope_base
    
    @eqx.filter_jit
    def __call__(
        self,
        ids: Int[Array, "T"],
        *,
        key: jax.Array,
        inference: bool,
        mask: Optional[Bool[Array, "T"]] = None,
        return_stats: bool = False,
        **kwargs,  # Accept but ignore routing-specific kwargs
    ) -> Tuple[Float[Array, "T V"], Tuple]:
        """Forward pass through dense model."""
        T = ids.shape[0]
        token_mask = jnp.ones((T,), dtype=bool) if mask is None else mask.astype(bool)
        
        # Embeddings
        h = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)
        
        # RoPE preparation
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_sin = rope_cos_sin(T, d_h, self.rope_base)
        
        # Process through layers
        for attn, mlp in self.layers:
            key, k_attn, k_ff = jax.random.split(key, 3)
            
            # Attention block
            a_out = attn(h, cos_sin, token_mask, key=k_attn, inference=inference)
            h = h + a_out
            h = jnp.where(token_mask[:, None], h, 0.0)
            
            # MLP block
            m_out = mlp(h, cos_sin, token_mask, key=k_ff, inference=inference)
            h = h + m_out
            h = jnp.where(token_mask[:, None], h, 0.0)
        
        # Output projection
        h = jax.vmap(self.ln_out)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        
        return logits, ()
