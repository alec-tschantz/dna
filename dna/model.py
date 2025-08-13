from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Bool, Array

from dna.modules import Embedding, Dropout, RMSNorm, rope_cos_sin
from dna.routing import Router


# ============================================================================
# Module signature tools for grouping identical module types/shapes
# ============================================================================


def _sig(mod: eqx.Module) -> Tuple[str, str, str]:
    """Return a structural signature for batched expert execution.

    Creates a unique signature based on:
    - Module type name
    - Array leaf structure
    - Array shapes and dtypes

    Modules with identical signatures can be vmapped together efficiently.
    """
    # Get array leaves and their properties
    arrs = eqx.filter(mod, eqx.is_array)
    arr_shapes_dtypes = jax.tree_util.tree_map(
        lambda x: (tuple(x.shape), str(x.dtype)), arrs
    )
    arr_struct = jax.tree_util.tree_structure(arrs)

    return (
        type(mod).__name__,
        str(arr_struct),
        str(jax.tree_util.tree_leaves(arr_shapes_dtypes)),
    )


def _stack(mods: List[eqx.Module]):
    """Stack parameters of structure-identical modules along a new leading axis.

    Returns
    -------
    params : PyTree
        Array leaves stacked to shape (n_mods, *leaf_shape).
    static : PyTree
        Non-array/static parts (shared across all stacked modules).
    """
    params = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *params), static


# ============================================================================
# Capacity-constrained assignment algorithm
# ============================================================================


def _capacity_select(
    mask_te: jnp.ndarray,
    score_te: jnp.ndarray,
    capacity: int,
):
    """Capacity-constrained assignment for tokens → expert slots.

    Implements top-k selection per expert with capacity constraints.
    Each expert can only process up to 'capacity' tokens.

    Parameters
    ----------
    mask_te : jnp.ndarray
        Shape (T, E) bool — hard top-k selection mask per token over experts.
    score_te : jnp.ndarray
        Shape (T, E) float — per-expert selection score per token.
    capacity : int
        Maximum tokens each expert can process (C), guaranteed > 0.

    Returns
    -------
    slot : jnp.ndarray
        Shape (E, C, T) float {0,1} — one-hot assignment of token→slot for each expert.
    kept : jnp.ndarray
        Shape (E, T) bool — whether token t was kept by expert e after capacity.
    top_idx : jnp.ndarray
        Shape (E, C) int32 — original token indices per slot.
    """
    # ===== Expert-major view to do per-expert top-k with capacity
    # m: (E, T) bool — which token→expert edges exist pre-capacity
    # g: (E, T) float — scores (masked to -inf so they never top-k)
    m = mask_te.T
    g = jnp.where(m, score_te.T, -jnp.inf)

    E, T = g.shape
    C = int(min(capacity, T))  # clamp just in case capacity > T

    # ===== Top-C selection per expert
    # top_idx: (E, C) — token indices in original sequence
    _, top_idx = jax.lax.top_k(g, C)

    # ===== Build one-hot slot tensor: (E, C, T)
    slot = jnp.zeros((E, C, T), dtype=g.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], top_idx].set(1.0)

    # ===== Strictly zero any positions that weren't in the pre-capacity mask
    slot = slot * m[:, None, :].astype(slot.dtype)

    # ===== kept[e, t] ≡ any(slot[e, :, t] == 1)
    kept = (slot.sum(axis=1) > 0).astype(bool)  # (E, T)

    return slot, kept, top_idx


# ============================================================================
# DNA Model Implementation
# ============================================================================


class Model(eqx.Module):
    """DNA model with routing-based expert execution.

    Processes sequences through multiple routing hops, where each hop
    assigns tokens to experts based on learned routing weights.
    Expert outputs are combined using router probabilities.
    """

    # Core components
    embed: Embedding
    dropout: Dropout
    ln_out: RMSNorm
    routers: Tuple[Router, ...]

    # Expert execution planning
    backbone: Tuple[eqx.Module, ...]
    groups: Tuple[Dict[str, Any], ...]

    # Configuration (static)
    capacity: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        modules: Tuple[eqx.Module, ...],
        vocab: int,
        d_model: int,
        n_heads: int,
        capacity: int,
        topk: int,
        n_hops: int,
        dropout: float,
        rope_base: float,
        backbone: Optional[Tuple[eqx.Module, ...]] = None,
        key,
    ):
        # Store configuration
        self.capacity = int(capacity)
        self.n_heads = int(n_heads)
        self.rope_base = float(rope_base)

        # Split keys
        k_embed, k_routers = jax.random.split(key, 2)

        # Initialize token processing components
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        # Group experts by structural signature for efficient batching
        buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for idx, mod in enumerate(modules):
            sig = _sig(mod)
            entry = buckets.setdefault(sig, {"idx": [], "mods": []})
            entry["idx"].append(idx)
            entry["mods"].append(mod)

        # Stack parameters for vmapped execution
        grouped = []
        for bucket in buckets.values():
            params, static = _stack(bucket["mods"])
            grouped.append(
                dict(
                    idx=jnp.array(bucket["idx"], jnp.int32),
                    params=params,
                    static=static,
                )
            )

        # Sort groups by first index for stable ordering
        grouped.sort(key=lambda d: int(d["idx"][0]))
        self.groups = tuple(grouped)

        # Store optional backbone
        self.backbone = tuple(backbone) if backbone is not None else tuple()

        # Initialize routers
        total_experts = len(modules)
        router_keys = jax.random.split(k_routers, n_hops)
        self.routers = tuple(
            Router(d_model, total_experts, topk, key=k) for k in router_keys
        )

    def _hop(
        self,
        h: Float[Array, "T d"],
        router: Router,
        cos_sin: Tuple[Float[Array, "T d_h"], Float[Array, "T d_h"]],
        *,
        key,
        inference: bool,
        token_mask: Bool[Array, "T"],
        gumbel_tau: float,
        router_temp: float,
        select_temp: Optional[float],
    ) -> Tuple[Float[Array, "T d"], Dict[str, Any]]:
        """Execute one routing hop."""
        # Split keys for routing and execution
        k_route, k_exec = jax.random.split(key)

        # Perform routing to get hard selection and soft mixing weights
        mask_full, probs_full, logits_clean, logits_sel = router(
            h,
            key=k_route,
            inference=inference,
            gumbel_tau=gumbel_tau,
            router_temp=router_temp,
            select_temp=select_temp,
        )  # shapes: (T,E), (T,E), (T,E), (T,E)

        # Mask out padding tokens
        mask_full = jnp.where(token_mask[:, None], mask_full, False)  # (T,E)
        probs_full = jnp.where(token_mask[:, None], probs_full, 0.0)  # (T,E)

        # Apply capacity constraints
        slot, kept, top_idx = _capacity_select(mask_full, logits_sel, self.capacity)

        # Gather inputs for each expert slot
        cos, sin = cos_sin
        xin = jnp.einsum("ect,td->ecd", slot, h)  # (E, C, d)
        cosr = jnp.einsum("ect,td->ecd", slot, cos)  # (E, C, d_h)
        sinr = jnp.einsum("ect,td->ecd", slot, sin)  # (E, C, d_h)

        # Determine which slots are active
        active = slot.sum(-1) > 0  # (E, C) bool

        # Sort slots by original token position for causal ordering
        pos_for_sort = jnp.where(active, top_idx, h.shape[0] + 1)  # (E, C)
        order = jnp.argsort(pos_for_sort, axis=1, stable=True)  # (E, C)

        # Apply sorting permutation
        def _take(a):
            return jnp.take_along_axis(a, order[:, :, None], axis=1)

        xin = _take(xin)  # (E, C, d)
        cosr = _take(cosr)  # (E, C, d_h)
        sinr = _take(sinr)  # (E, C, d_h)
        active = jnp.take_along_axis(active, order, axis=1)  # (E, C)
        slot = jnp.take_along_axis(slot, order[:, :, None], 1)  # (E, C, T)

        # Execute experts in batched groups
        E, C, d = xin.shape
        expert_out = jnp.zeros((E, C, d), dtype=xin.dtype)

        for gi, g in enumerate(self.groups):
            # Generate keys for this group
            sub_keys = jax.random.split(jax.random.fold_in(k_exec, gi), len(g["idx"]))

            # Extract group inputs
            inp = xin[g["idx"]]  # (n_g, C, d)
            c = cosr[g["idx"]]  # (n_g, C, d_h)
            s = sinr[g["idx"]]  # (n_g, C, d_h)
            am = active[g["idx"]]  # (n_g, C)

            # Run experts via vmap
            def _run(p, x, c1, s1, am1, k1):
                mod = eqx.combine(p, g["static"])
                return mod(x, (c1, s1), am1, key=k1, inference=inference)

            out_g = jax.vmap(_run)(g["params"], inp, c, s, am, sub_keys)  # (n_g, C, d)

            # Scatter outputs back
            expert_out = expert_out.at[g["idx"]].set(out_g)

        # Combine expert outputs weighted by router probabilities
        kept_t = kept.T  # (T, E)
        combine_w = jnp.where(kept_t, probs_full, 0.0)  # (T, E)
        rho = combine_w.sum(axis=1, keepdims=True)  # (T, 1)

        # Route outputs back to tokens
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)  # (T, d)

        # Apply residual update (Eq. 3) and preserve padding
        h_next = h + combine - rho * h
        h_next = jnp.where(token_mask[:, None], h_next, h)

        # Collect raw statistics (no aggregation here)
        stats = self._stats(
            kept=kept, probs=probs_full, mask=mask_full, rho=rho, token_mask=token_mask
        )

        return h_next, stats

    @eqx.filter_jit
    def __call__(
        self,
        ids: Int[Array, "T"],
        *,
        key,
        inference: bool,
        mask: Optional[Bool[Array, "T"]] = None,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
    ) -> Tuple[Float[Array, "T V"], Tuple[Dict[str, Any], ...]]:
        """Forward pass through DNA model.

        Parameters
        ----------
        ids : Int[Array, "T"]
            Token IDs for a single sequence.
        key : PRNGKey
            Random key for dropout/routing.
        inference : bool
            If True, disables dropout and routing exploration.
        mask : Optional[Bool[Array, "T"]]
            Valid token mask (True=valid, False=padding).
        gumbel_tau : float
            Gumbel noise scale for training-time exploration.
        router_temp : float
            Temperature for mixing probabilities.
        select_temp : Optional[float]
            Temperature for selection logits (defaults to router_temp).

        Returns
        -------
        logits : Float[Array, "T V"]
            Output logits over vocabulary.
        stats : Tuple[Dict[str, Any], ...]
            Raw statistics from each routing hop.
        """
        # Setup token mask
        T = ids.shape[0]
        token_mask: Bool[Array, "T"] = (
            jnp.ones((T,), dtype=bool) if mask is None else mask.astype(bool)
        )

        # Embed tokens and apply dropout
        h: Float[Array, "T d"] = jax.vmap(self.embed)(ids)  # (T, d)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        # Prepare RoPE positional encodings
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_sin = rope_cos_sin(T, d_h, self.rope_base)  # each (T, d_h)

        # Process through optional backbone layers
        for mod in self.backbone:
            key, sub = jax.random.split(key)
            out = mod(h, cos_sin, token_mask, key=sub, inference=inference)
            h = jnp.where(token_mask[:, None], out, h)

        # Execute routing hops
        stats_all: List[Dict[str, Any]] = []
        for router in self.routers:
            key, sub = jax.random.split(key)
            h, st = self._hop(
                h,
                router,
                cos_sin,
                key=sub,
                inference=inference,
                token_mask=token_mask,
                gumbel_tau=gumbel_tau,
                router_temp=router_temp,
                select_temp=select_temp,
            )
            stats_all.append(st)

        # Final layer norm and unembedding (tied weights)
        h = jax.vmap(self.ln_out)(h)  # (T, d)
        logits: Float[Array, "T V"] = jax.vmap(lambda t: t @ self.embed.weight.T)(h)

        return logits, tuple(stats_all)

    def _stats(
        self,
        *,
        kept: Bool[Array, "E T"],
        probs: Float[Array, "T E"],
        mask: Bool[Array, "T E"],
        rho: Float[Array, "T 1"],
        token_mask: Bool[Array, "T"],
    ) -> Dict[str, Any]:
        """Collect raw per-hop routing statistics for later aggregation."""
        # load[e] = number of tokens kept by expert e (E,)
        load = kept.sum(axis=1).astype(jnp.int32)

        # importance[e] = sum of routing probs to expert e across tokens (E,)
        importance = probs.sum(axis=0)

        # entropy[t] = entropy of per-token routing probs (normalized) (T,)
        p = probs
        p_sum = p.sum(axis=1, keepdims=True) + 1e-9
        p_norm = p / p_sum
        entropy = -(p_norm * jnp.log(p_norm + 1e-9)).sum(axis=1)
        entropy = entropy / jnp.log(p.shape[1] + 1e-9)

        # selected_edges = total hard edges before capacity (scalar)
        selected_edges = mask.astype(jnp.int32).sum()

        # kept_edges = total hard edges after capacity (scalar)
        kept_edges = kept.astype(jnp.int32).sum()

        # eff_topk[t] = number of selected experts for token t before capacity (T,)
        eff_topk = mask.astype(jnp.int32).sum(axis=1)

        # rho[t] = total mixing weight kept for token t after capacity (T,)
        # routing_probs = per-token full distribution over experts (T,E)
        # token_mask[t] = valid token flag (T,)
        return dict(
            load=load,  # (E,) int32 — tokens kept per expert
            importance=importance,  # (E,) float — sum of probs per expert
            rho=rho[:, 0],  # (T,) float — total kept mixing weight
            entropy=entropy,  # (T,) float — normalized routing entropy
            selected_edges=selected_edges,  # () int32 — edges pre-capacity
            kept_edges=kept_edges,  # () int32 — edges post-capacity
            eff_topk=eff_topk,  # (T,) int32 — selected experts per token (pre-cap)
            routing_probs=probs,  # (T,E) float — softmax over experts
            token_mask=token_mask,  # (T,) bool — valid tokens
        )
