from __future__ import annotations
from typing import Tuple, Optional, Dict, Type
from abc import ABC, abstractmethod
from jaxtyping import Float, Bool, Array

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from dna.modules import RMSNorm, Dropout


class BaseRouter(eqx.Module, ABC):
    """Abstract base class for routers."""
    
    k: int = eqx.field(static=True)
    norm_probs: bool = eqx.field(static=True)
    dropout: Dropout
    
    @abstractmethod
    def compute_logits(
        self,
        h: Float[Array, "T d"],
        key: jax.Array,
        inference: bool,
    ) -> Float[Array, "T E"]:
        """Compute routing logits."""
        pass
    
    def __call__(
        self,
        h: Float[Array, "T d"],
        *,
        key: jax.Array,
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        token_mask: Optional[Bool[Array, "T"]] = None,
    ) -> Tuple[Bool[Array, "T E"], Float[Array, "T E"], Float[Array, "T E"], Float[Array, "T E"]]:
        """Common routing logic."""
        # Compute logits
        logits_clean = self.compute_logits(h, key, inference)
        
        # Selection temperature
        t_sel = select_temp if select_temp is not None else router_temp
        logits_sel = logits_clean / jnp.maximum(t_sel, 1e-6)
        
        # Add Gumbel noise if training
        if not inference and gumbel_tau > 0:
            k_gumbel = jax.random.fold_in(key, 1)
            uniform = jax.random.uniform(k_gumbel, logits_sel.shape, minval=1e-6, maxval=1-1e-6)
            gumbel = -jnp.log(-jnp.log(uniform))
            logits_sel = logits_sel + gumbel_tau * gumbel
        
        # Apply token mask
        if token_mask is not None:
            logits_sel = jnp.where(token_mask[:, None], logits_sel, -jnp.inf)
        
        # Top-k selection
        _, top_indices = jax.lax.top_k(logits_sel, self.k)
        mask_full = jnn.one_hot(top_indices, logits_sel.shape[-1]).sum(axis=1).astype(bool)
        
        if token_mask is not None:
            mask_full = mask_full & token_mask[:, None]
        
        # Mixing probabilities
        t_mix = jnp.maximum(router_temp, 1e-6)
        probs = jnn.softmax(logits_clean / t_mix, axis=-1)
        
        if token_mask is not None:
            probs = jnp.where(token_mask[:, None], probs, 0.0)
        
        if self.norm_probs:
            masked_probs = jnp.where(mask_full, probs, 0.0)
            denom = jnp.maximum(masked_probs.sum(axis=-1, keepdims=True), 1e-9)
            probs = masked_probs / denom
            if token_mask is not None:
                probs = jnp.where(token_mask[:, None], probs, 0.0)
        
        return mask_full, probs, logits_clean, logits_sel


class LinearRouter(BaseRouter):
    """Linear projection router."""
    
    proj: eqx.nn.Linear
    
    def __init__(
        self, d_model: int, n_exp: int, k: int, dropout: float, norm_probs: bool, *, key: jax.Array
    ):
        self.k = k
        self.norm_probs = norm_probs
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)
        self.dropout = Dropout(dropout)
    
    def compute_logits(
        self, h: Float[Array, "T d"], key: jax.Array, inference: bool
    ) -> Float[Array, "T E"]:
        k_drop = jax.random.fold_in(key, 0)
        h = self.dropout(h, key=k_drop, inference=inference)
        return jax.vmap(self.proj)(h)


class CosineRouter(BaseRouter):
    """Cosine similarity router with prototypes."""
    
    prototypes: Float[Array, "E P d"]
    scale: float = eqx.field(static=True)
    n_prototypes: int = eqx.field(static=True)
    
    def __init__(
        self, d_model: int, n_exp: int, k: int, dropout: float, norm_probs: bool, *, key: jax.Array
    ):
        self.k = k
        self.norm_probs = norm_probs
        self.n_prototypes = 2
        self.scale = 10.0
        self.dropout = Dropout(dropout)
        
        # Initialize prototypes
        std = 1.0 / jnp.sqrt(d_model)
        self.prototypes = jax.random.normal(key, (n_exp, self.n_prototypes, d_model)) * std
    
    def compute_logits(
        self, h: Float[Array, "T d"], key: jax.Array, inference: bool
    ) -> Float[Array, "T E"]:
        k_drop = jax.random.fold_in(key, 0)
        h = self.dropout(h, key=k_drop, inference=inference)
        
        # L2 normalize
        h_norm = h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-6)
        p_norm = self.prototypes / jnp.maximum(
            jnp.linalg.norm(self.prototypes, axis=-1, keepdims=True), 1e-6
        )
        
        # Compute similarities
        sims = jnp.einsum("td,epd->tep", h_norm, p_norm)
        return self.scale * jax.nn.logsumexp(sims, axis=-1)


class SequenceRouter(BaseRouter):
    """Sequential routing with recurrent state."""
    
    w_in: eqx.nn.Linear
    w_rec: eqx.nn.Linear
    proj: eqx.nn.Linear
    h_norm: RMSNorm
    x_norm: RMSNorm
    h0: Float[Array, "d"]
    
    def __init__(
        self, d_model: int, n_exp: int, k: int, dropout: float, norm_probs: bool, *, key: jax.Array
    ):
        self.k = k
        self.norm_probs = norm_probs
        self.dropout = Dropout(dropout)
        
        k_in, k_rec, k_out = jax.random.split(key, 3)
        self.w_in = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_in)
        self.w_rec = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_rec)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=k_out)
        
        self.h_norm = RMSNorm(d_model)
        self.x_norm = RMSNorm(d_model)
        self.h0 = jnp.zeros((d_model,))
    
    def compute_logits(
        self, h: Float[Array, "T d"], key: jax.Array, inference: bool
    ) -> Float[Array, "T E"]:
        k_drop = jax.random.fold_in(key, 0)
        h = self.dropout(h, key=k_drop, inference=inference)
        
        # Scan through sequence
        def step(state, x):
            s_prev = state
            s_new = jnn.silu(
                self.w_rec(self.h_norm(s_prev)) + self.w_in(self.x_norm(x))
            )
            logits = self.proj(s_new)
            return s_new, logits
        
        _, logits_seq = jax.lax.scan(step, self.h0, h)
        return logits_seq


class RouterRegistry:
    """Registry for router types."""
    
    _routers: Dict[str, Type[BaseRouter]] = {
        "linear": LinearRouter,
        "cosine": CosineRouter,
        "sequence": SequenceRouter,
    }
    
    @classmethod
    def register(cls, name: str, router_cls: Type[BaseRouter]):
        """Register a new router type."""
        cls._routers[name] = router_cls
    
    @classmethod
    def get(cls, name: str) -> Type[BaseRouter]:
        """Get router class by name."""
        if name not in cls._routers:
            raise ValueError(f"Unknown router type: {name}. Available: {list(cls._routers.keys())}")
        return cls._routers[name]