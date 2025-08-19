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
from dna.routing import RouterRegistry, BaseRouter


@dataclass
class ModelKwargs:
    """Type-safe model kwargs."""
    gumbel_tau: float = 1.0
    router_temp: float = 1.0
    select_temp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gumbel_tau": self.gumbel_tau,
            "router_temp": self.router_temp,
            "select_temp": self.select_temp,
        }


class ModuleGroup:
    """Efficient grouped module execution."""
    
    def __init__(self, modules: List[eqx.Module]):
        """Group modules by signature for efficient batched execution."""
        self.groups = self._create_groups(modules)
    
    @staticmethod
    def _signature(mod: eqx.Module) -> Tuple[str, ...]:
        """Get module signature for grouping."""
        arrs = eqx.filter(mod, eqx.is_array)
        shapes_dtypes = jax.tree_map(lambda x: (tuple(x.shape), str(x.dtype)), arrs)
        return (
            type(mod).__name__,
            str(jax.tree_structure(arrs)),
            str(jax.tree_leaves(shapes_dtypes))
        )
    
    def _create_groups(self, modules: List[eqx.Module]) -> List[Dict[str, Any]]:
        """Create groups of modules with same signature."""
        buckets: Dict[Tuple, Dict[str, Any]] = {}
        
        for idx, mod in enumerate(modules):
            sig = self._signature(mod)
            entry = buckets.setdefault(sig, {"indices": [], "modules": []})
            entry["indices"].append(idx)
            entry["modules"].append(mod)
        
        groups = []
        for bucket in buckets.values():
            # Stack parameters for vectorized execution
            params = [eqx.filter(m, eqx.is_array) for m in bucket["modules"]]
            static = eqx.filter(bucket["modules"][0], lambda x: not eqx.is_array(x))
            stacked_params = jax.tree_map(lambda *xs: jnp.stack(xs, 0), *params)
            
            groups.append({
                "indices": jnp.array(bucket["indices"], dtype=jnp.int32),
                "params": stacked_params,
                "static": static,
            })
        
        # Sort by first index for consistency
        groups.sort(key=lambda g: int(g["indices"][0]))
        return groups
    
    def execute(
        self,
        inputs: Float[Array, "E C d"],
        cos_sin: Tuple[Float[Array, "C d_h"], Float[Array, "C d_h"]],
        active_mask: Bool[Array, "E C"],
        keys: jax.Array,
        inference: bool,
    ) -> Float[Array, "E C d"]:
        """Execute all modules efficiently in groups."""
        E, C, d = inputs.shape
        outputs = jnp.zeros_like(inputs)
        
        for gi, group in enumerate(self.groups):
            group_keys = jax.random.split(keys[gi], len(group["indices"]))
            group_inputs = inputs[group["indices"]]
            group_cos = cos_sin[0][group["indices"]]
            group_sin = cos_sin[1][group["indices"]]
            group_mask = active_mask[group["indices"]]
            
            # Vectorized execution
            def run_module(params, x, cos, sin, mask, key):
                module = eqx.combine(params, group["static"])
                return module(x, (cos, sin), mask, key=key, inference=inference)
            
            group_outputs = jax.vmap(run_module)(
                group["params"], group_inputs, group_cos, group_sin, group_mask, group_keys
            )
            outputs = outputs.at[group["indices"]].set(group_outputs)
        
        return outputs


@eqx.filter_jit
def capacity_selection(
    mask_te: Bool[Array, "T E"],
    score_te: Float[Array, "T E"],
    capacity: int,
) -> Tuple[Float[Array, "E C T"], Bool[Array, "E T"], Int[Array, "E C"]]:
    """Optimized capacity-based expert selection."""
    E, T = mask_te.shape[1], mask_te.shape[0]
    C = min(capacity, T)
    
    # Transpose for expert-first processing
    mask_et = mask_te.T
    score_et = jnp.where(mask_et, score_te.T, -jnp.inf)
    
    # Top-k selection per expert
    _, top_indices = jax.lax.top_k(score_et, C)
    
    # Create selection tensor efficiently
    slot = jnp.zeros((E, C, T), dtype=jnp.float32)
    expert_idx = jnp.arange(E)[:, None]
    capacity_idx = jnp.arange(C)[None, :]
    slot = slot.at[expert_idx, capacity_idx, top_indices].set(1.0)
    
    # Mask out invalid selections
    slot = slot * mask_et[:, None, :].astype(slot.dtype)
    kept = (slot.sum(axis=1) > 0)
    
    return slot, kept, top_indices


class DNA(eqx.Module):
    """Dynamic Neural Architecture with optimized routing."""
    
    embed: Embedding
    dropout: Dropout
    ln_out: RMSNorm
    routers: Tuple[BaseRouter, ...]
    backbone: Tuple[eqx.Module, ...]
    module_group: ModuleGroup
    capacity: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    norm_after_capacity: bool = eqx.field(static=True)
    
    def __init__(
        self,
        modules: Tuple[eqx.Module, ...],
        router_cls: type[BaseRouter],
        vocab: int,
        d_model: int,
        n_heads: int,
        capacity: int,
        topk: int,
        n_hops: int,
        dropout: float,
        rope_base: float,
        norm_probs: bool = False,
        norm_after_capacity: bool = False,
        backbone: Optional[Tuple[eqx.Module, ...]] = None,
        *,
        key: jax.Array 
    ):
        # Validate inputs
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert (d_model // n_heads) % 2 == 0, "Head dimension must be even for RoPE"
        
        self.capacity = capacity
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.norm_after_capacity = norm_after_capacity
        
        # Initialize embeddings and normalization
        k_embed, k_routers = jax.random.split(key)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)
        
        # Group modules for efficient execution
        self.module_group = ModuleGroup(list(modules))
        
        # Initialize backbone
        self.backbone = tuple(backbone) if backbone else ()
        
        # Initialize routers
        n_experts = len(modules)
        router_keys = jax.random.split(k_routers, n_hops)
        self.routers = tuple(
            router_cls(d_model, n_experts, topk, dropout, norm_probs, key=k)
            for k in router_keys
        )
    
    @classmethod
    def from_config(
        cls,
        vocab: int,
        d_model: int,
        n_heads: int,
        n_att: int,
        n_ff: int,
        n_id: int,
        mlp_mult: int,
        dropout: float,
        rope_base: float,
        router_type: str,
        capacity: int,
        topk: int,
        n_hops: int,
        norm_probs: bool,
        norm_after_capacity: bool,
        backbone: Tuple[str, ...],
        key: jax.Array,
    ) -> "DNA":
        """Create DNA model from configuration."""
        k_modules, k_backbone, k_model = jax.random.split(key, 3)
        
        # Create modules
        modules = ModuleRegistry.create_modules(
            d_model=d_model,
            n_heads=n_heads,
            n_att=n_att,
            n_ff=n_ff,
            n_id=n_id,
            mlp_mult=mlp_mult,
            dropout=dropout,
            key=k_modules,
        )
        
        # Create backbone
        backbone_modules = ModuleRegistry.create_backbone(
            d_model=d_model,
            n_heads=n_heads,
            mlp_mult=mlp_mult,
            dropout=dropout,
            backbone=backbone,
            key=k_backbone,
        )
        
        # Get router class
        router_cls = RouterRegistry.get(router_type)
        
        return cls(
            modules=modules,
            router_cls=router_cls,
            vocab=vocab,
            d_model=d_model,
            n_heads=n_heads,
            capacity=capacity,
            topk=topk,
            n_hops=n_hops,
            dropout=dropout,
            rope_base=rope_base,
            norm_probs=norm_probs,
            norm_after_capacity=norm_after_capacity,
            backbone=backbone_modules,
            key=k_model,
        )
    
    def _hop(
        self,
        h: Float[Array, "T d"],
        router: BaseRouter,
        cos_sin: Tuple[Float[Array, "T d_h"], Float[Array, "T d_h"]],
        key: jax.Array,
        inference: bool,
        token_mask: Bool[Array, "T"],
        gumbel_tau: float,
        router_temp: float,
        select_temp: Optional[float],
        return_stats: bool,
    ) -> Tuple[Float[Array, "T d"], Dict[str, Any]]:
        """Single routing hop with optimized computation."""
        k_route, k_exec = jax.random.split(key)
        
        # Get routing decisions
        mask_full, probs_full, logits_clean, logits_sel = router(
            h, key=k_route, inference=inference,
            gumbel_tau=gumbel_tau, router_temp=router_temp,
            select_temp=select_temp, token_mask=token_mask
        )
        
        # Capacity-based selection
        slot, kept, top_indices = capacity_selection(mask_full, logits_sel, self.capacity)
        
        # Prepare inputs for experts (optimized tensor operations)
        cos, sin = cos_sin
        xin = jnp.einsum("ect,td->ecd", slot, h)
        cosr = jnp.einsum("ect,td->ecd", slot, cos)
        sinr = jnp.einsum("ect,td->ecd", slot, sin)
        active = slot.sum(-1) > 0
        
        # Sort for sequential processing
        T = h.shape[0]
        pos_for_sort = jnp.where(active, top_indices, T + 1)
        order = jnp.argsort(pos_for_sort, axis=1, stable=True)
        
        # Reorder tensors
        xin = jnp.take_along_axis(xin, order[:, :, None], axis=1)
        cosr = jnp.take_along_axis(cosr, order[:, :, None], axis=1)
        sinr = jnp.take_along_axis(sinr, order[:, :, None], axis=1)
        active = jnp.take_along_axis(active, order, axis=1)
        slot = jnp.take_along_axis(slot, order[:, :, None], axis=1)
        
        # Execute experts
        E = xin.shape[0]
        group_keys = jax.random.split(k_exec, len(self.module_group.groups))
        expert_out = self.module_group.execute(xin, (cosr, sinr), active, group_keys, inference)
        
        # Combine outputs
        kept_t = kept.T
        combine_w = jnp.where(kept_t, probs_full, 0.0)
        
        if self.norm_after_capacity:
            denom = jnp.maximum(combine_w.sum(axis=1, keepdims=True), 1e-9)
            combine_w = combine_w / denom
        
        rho = combine_w.sum(axis=1, keepdims=True)
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)
        
        # Residual connection with routing coefficient
        h_next = h + combine - rho * h
        h_next = jnp.where(token_mask[:, None], h_next, h)
        
        # Collect stats if needed
        stats = {}
        if return_stats:
            stats = {
                "load": kept.sum(axis=1).astype(jnp.int32),
                "importance": probs_full.sum(axis=0),
                "rho": rho[:, 0],
                "entropy": self._compute_entropy(probs_full),
                "selected_edges": mask_full.astype(jnp.int32).sum(),
                "kept_edges": kept.astype(jnp.int32).sum(),
                "eff_topk": mask_full.astype(jnp.int32).sum(axis=1),
                "routing_probs": probs_full,
                "token_mask": token_mask,
            }
        
        return h_next, stats
    
    @staticmethod
    def _compute_entropy(probs: Float[Array, "T E"]) -> Float[Array, "T"]:
        """Compute normalized entropy of routing probabilities."""
        p_norm = probs / jnp.maximum(probs.sum(axis=1, keepdims=True), 1e-9)
        entropy = -(p_norm * jnp.log(p_norm + 1e-9)).sum(axis=1)
        return entropy / jnp.log(probs.shape[1] + 1e-9)
    
    @eqx.filter_jit
    def __call__(
        self,
        ids: Int[Array, "T"],
        *,
        key: jax.Array,
        inference: bool,
        mask: Optional[Bool[Array, "T"]] = None,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        return_stats: bool = False,
    ) -> Tuple[Float[Array, "T V"], Tuple[Dict[str, Any], ...]]:
        """Forward pass through DNA model."""
        T = ids.shape[0]
        token_mask = jnp.ones((T,), dtype=bool) if mask is None else mask.astype(bool)
        
        # Embedding and initial dropout
        h = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)
        
        # Prepare RoPE
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_sin = rope_cos_sin(T, d_h, self.rope_base)
        
        # Process through backbone
        for mod in self.backbone:
            key, sub = jax.random.split(key)
            out = mod(h, cos_sin, token_mask, key=sub, inference=inference)
            h = jnp.where(token_mask[:, None], out, h)
        
        # Process through routing hops
        stats_all: List[Dict[str, Any]] = []
        for router in self.routers:
            key, sub = jax.random.split(key)
            h, stats = self._hop(
                h, router, cos_sin, key=sub, inference=inference,
                token_mask=token_mask, gumbel_tau=gumbel_tau,
                router_temp=router_temp, select_temp=select_temp,
                return_stats=return_stats
            )
            stats_all.append(stats)
        
        # Final normalization and output projection
        h = jax.vmap(self.ln_out)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        
        return logits, tuple(stats_all)
