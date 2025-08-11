# dna/model.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Bool, Array

from dna.modules import Embedding, Dropout, RMSNorm, rope_cos_sin
from dna.routing import Router


# ============================================================================
# Helper: module signature tools to group identical module types/shapes
# ============================================================================


def _sig(mod: eqx.Module) -> Tuple[str, str, str]:
    """Return a structural signature for batched expert execution.

    We want a stable signature that clusters modules that can be vmapped together.
    We key on:
      - type name
      - array-leaf structure
      - array-leaf shapes+dtypes
    """
    # Array leaves → (shape, dtype)
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
        Leaves stacked to shape ``(n_mods, *leaf_shape)``.
    static : PyTree
        The non-array/static part (reused for all stacked modules).
    """
    params = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *params), static


# ============================================================================
# Capacity-constrained assignment
# ============================================================================


def _capacity_select(
    mask_te: jnp.ndarray,
    score_te: jnp.ndarray,
    capacity: int,
):
    """Capacity-constrained assignment for tokens → expert slots.

    Parameters
    ----------
    mask_te : jnp.ndarray
        Shape ``(T, E)`` bool — hard top-k selection mask *per token over experts*.
        (Identity experts, if any, are treated as normal experts.)
    score_te : jnp.ndarray
        Shape ``(T, E)`` float — *per-expert selection score per token* used
        to rank tokens for a given expert. **Use clean logits**, not per-token
        probabilities, so ranking is invariant across tokens.
    capacity : int
        Per-expert capacity (number of token slots kept at this hop).

    Returns
    -------
    slot : jnp.ndarray
        Shape ``(E, C, T)`` one-hot over tokens per *slot* per expert; i.e.,
        ``slot[e, c, t] = 1.0`` if token ``t`` is assigned to expert ``e``'s slot ``c``.
    kept : jnp.ndarray
        Shape ``(E, T)`` bool — token t kept by expert e (assigned any slot).
    top_idx : jnp.ndarray
        Shape ``(E, C)`` int32 — original token indices selected by each expert.
    """
    # Switch to expert-major for ranking tokens per expert.
    # Only consider tokens the router *hard-selected* for that expert.
    m = mask_te.T  # (E, T) bool
    g = jnp.where(m, score_te.T, -jnp.inf)  # (E, T) float

    cap = int(min(capacity, g.shape[1]))  # C ≤ T
    # Top-C tokens per expert by score
    _, top_idx = jax.lax.top_k(g, cap)  # (E, C)
    E, C = top_idx.shape
    T = g.shape[1]

    # Build slot assignment tensor (E, C, T)
    slot = jnp.zeros((E, C, T), dtype=g.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], top_idx].set(1.0)

    # Safety: clear any indices that weren't actually selected (ties/NaNs etc.)
    slot = slot * m[:, None, :].astype(slot.dtype)

    # kept[e, t] = True iff token t was picked by expert e for some slot
    kept = (slot.sum(1) > 0).astype(bool)  # (E, T)
    return slot, kept, top_idx


# ============================================================================
# DNA model
# ============================================================================


class Model(eqx.Module):
    """Routing-and-execute core for a single (unbatched) sequence.

    Experts must implement: ``__call__(x, (cos, sin), attention_mask, *, key, inference)``
    """

    # ---- Embedding / norm / dropout / routers ----
    embed: Embedding
    dropout: Dropout
    ln_out: RMSNorm
    routers: Tuple[Router, ...]

    # ---- Execution planning ----
    backbone: Tuple[eqx.Module, ...]
    groups: Tuple[Dict[str, Any], ...]

    # ---- Static metadata ----
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
        # ====== Meta/config ======
        self.capacity = int(capacity)
        self.n_heads = int(n_heads)
        self.rope_base = float(rope_base)

        # ====== Keys ======
        k_embed, k_routers = jax.random.split(key, 2)

        # ====== Token path: embed → dropout → output norm ======
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        # ====== Bucket identical experts for batched execution ======
        buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for idx, mod in enumerate(modules):
            s = _sig(mod)
            entry = buckets.setdefault(s, {"idx": [], "mods": []})
            entry["idx"].append(idx)
            entry["mods"].append(mod)

        grouped = []
        for b in buckets.values():
            params, static = _stack(b["mods"])  # stack arrays; capture shared static
            grouped.append(
                dict(idx=jnp.array(b["idx"], jnp.int32), params=params, static=static)
            )
        # Sort by the first original index so interleaved groups keep stable ordering
        grouped.sort(key=lambda d: int(d["idx"][0]))
        self.groups = tuple(grouped)

        # ====== Optional backbone ======
        self.backbone = tuple(backbone) if backbone is not None else tuple()

        # ====== Routers ======
        total_experts = len(modules)
        router_keys = jax.random.split(k_routers, n_hops)
        self.routers = tuple(
            Router(d_model, total_experts, topk, key=k) for k in router_keys
        )

    # ---------------------------------------------------------------------
    # One routing + execution hop
    # ---------------------------------------------------------------------
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
        router_temperature: float,
        select_temperature: Optional[float],
    ) -> Tuple[Float[Array, "T d"], Dict[str, Any]]:
        r"""
        Implements the K-ribbon step with capacity. After routing and execution,
        we combine outputs as in Eq. (3):

            ``h_{s+1} = h_s + Σ_e ρ_e ( M_e(h_s^e) - h_s )``

        where the sum is over the *kept* experts for each token, with weights
        ``ρ_e`` from the router softmax (after temperature and capacity).
        """

        # -------- Randomness split (independent RNG for routing vs execution) ------
        # key -> k_route (routing noise, e.g., Gumbel), k_exec (expert dropout, etc.)
        k_route, k_exec = jax.random.split(key)

        # -------- Routing: hard top-k selection + soft mixing over all experts ------
        # router(...) returns:
        #   mask_full: (T, E) bool  hard top-k mask per token
        #   probs_full: (T, E) float  softmax over experts (mixing weights)
        #   logits_clean: (T, E) float  pre-noise logits (used for capacity ranking)
        mask_full, probs_full, logits_clean = router(
            h,
            key=k_route,
            inference=inference,
            gumbel_tau=gumbel_tau,
            router_temperature=router_temperature,  # mixing temperature
            select_temperature=select_temperature,  # selection temperature (opt.)
        )  # -> (T,E), (T,E), (T,E)

        # -------- Respect padding: masked tokens select nothing / mix nothing -------
        # token_mask: (T,) bool   True=valid, False=pad
        # Broadcast along expert axis to zero out invalid rows.
        mask_full = jnp.where(token_mask[:, None], mask_full, False)  # (T,E)
        probs_full = jnp.where(token_mask[:, None], probs_full, 0.0)  # (T,E)

        # -------- Capacity assignment per expert (tokens -> expert slots) -----------
        # Rank tokens for each expert using *clean logits* (not per-token probs).
        # Returns:
        #   slot: (E, C, T) one-hot, slot[e,c,t]=1 iff token t assigned to (e,c)
        #   kept: (E, T) bool, True iff e kept t in some slot
        #   top_idx: (E, C) int, original token indices per (e,c)
        slot, kept, top_idx = _capacity_select(mask_full, logits_clean, self.capacity)

        # -------- Gather per-slot inputs (einsum via one-hot slot tensor) -----------
        # h:   (T, d)      -> xin:  (E, C, d)
        # cos: (T, d_h)    -> cosr: (E, C, d_h)
        # sin: (T, d_h)    -> sinr: (E, C, d_h)
        cos, sin = cos_sin
        xin = jnp.einsum("ect,td->ecd", slot, h)  # (E, C, d)
        cosr = jnp.einsum("ect,td->ecd", slot, cos)  # (E, C, d_h)
        sinr = jnp.einsum("ect,td->ecd", slot, sin)  # (E, C, d_h)

        # -------- Active slot mask (which rows in C dimension are populated) --------
        # active[e,c] = True if slot[e,c,:] selects any token
        active = slot.sum(-1) > 0  # (E, C) bool

        # -------- Per-expert causal ordering (sort by original token positions) -----
        # We need the expert’s internal time axis to be chronological for causal ops.
        # pos_for_sort: (E, C) use token index for active rows; T+1 sentinel otherwise.
        # order: (E, C) permutation along C that sorts ascending by position.
        pos_for_sort = jnp.where(active, top_idx, h.shape[0] + 1)  # (E, C)
        order = jnp.argsort(pos_for_sort, axis=1, stable=True)  # (E, C)

        # Apply the same permutation to all per-slot tensors.
        def _take(a):
            return jnp.take_along_axis(a, order[:, :, None], axis=1)

        xin = _take(xin)  # (E, C, d)
        cosr = _take(cosr)  # (E, C, d_h)
        sinr = _take(sinr)  # (E, C, d_h)
        active = jnp.take_along_axis(active, order, axis=1)  # (E, C)
        slot = jnp.take_along_axis(slot, order[:, :, None], 1)  # (E, C, T)

        # -------- Execute experts in grouped batches (same-structure vmapping) ------
        # xin/cosr/sinr/active are now time-ordered per expert along C.
        # We vmap across experts in each structural group, then scatter back.
        E, C, d = xin.shape
        expert_out = jnp.zeros((E, C, d), dtype=xin.dtype)

        for gi, g in enumerate(self.groups):
            # sub_keys: (n_g,) RNGs, one per expert instance in this group
            sub_keys = jax.random.split(jax.random.fold_in(k_exec, gi), len(g["idx"]))

            # Slice group views
            inp = xin[g["idx"]]  # (n_g, C, d)
            c = cosr[g["idx"]]  # (n_g, C, d_h)
            s = sinr[g["idx"]]  # (n_g, C, d_h)
            am = active[g["idx"]]  # (n_g, C) bool

            # _run: rebuild module from (params, static) and run it
            def _run(p, x, c1, s1, am1, k1):
                mod = eqx.combine(p, g["static"])
                return mod(x, (c1, s1), am1, key=k1, inference=inference)

            out_g = jax.vmap(_run)(g["params"], inp, c, s, am, sub_keys)  # (n_g, C, d)

            # Scatter group outputs back to expert rows
            expert_out = expert_out.at[g["idx"]].set(out_g)

        # -------- Combine expert outputs back to original token positions ----------
        # kept_t:   (T, E) bool    transpose of kept
        # combine_w:(T, E) float   router probs masked by capacity-kept edges
        # rho:      (T, 1) float   total kept mass per token
        kept_t = kept.T
        combine_w = jnp.where(kept_t, probs_full, 0.0)  # (T, E)
        rho = combine_w.sum(axis=1, keepdims=True)  # (T, 1)

        # Route (E,C,d) -> (T,d) via slot[e,c,t] and weight by combine_w[t,e]:
        # combine[t] = Σ_e Σ_c expert_out[e,c] * slot[e,c,t] * combine_w[t,e]
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)  # (T, d)

        # -------- Residual update (Eq. 3) and pad freezing -------------------------
        # TODO: normalise rho & combine w?
        # h_next = (1 - rho) * h + combine
        h_next = h + combine - rho * h
        h_next = jnp.where(token_mask[:, None], h_next, h)

        # -------- Hop-level stats (load, entropy, rho, capacity drops, etc.) -------
        st = self._stats(
            kept=kept,
            probs=probs_full,
            mask=mask_full,
            rho=rho,
            token_mask=token_mask,
        )
        return h_next, st

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    @eqx.filter_jit
    def __call__(
        self,
        ids: Int[Array, "T"],
        *,
        key,
        inference: bool,
        attention_mask: Optional[Bool[Array, "T"]] = None,
        gumbel_tau: float = 1.0,
        router_temperature: float = 1.0,
        select_temperature: Optional[float] = None,
    ) -> Tuple[Float[Array, "T V"], Tuple[Dict[str, Any], ...]]:
        r"""
        Parameters
        ----------
        ids : Int[Array, "T"]
            Token ids for a *single* sequence (batch via `vmap` externally).
        key : PRNGKey
            For dropout/routing/module calls.
        inference : bool
            If ``True``, disables dropout and Gumbel in routers.
        attention_mask : Optional[Bool[Array, "T"]]
            1/True for valid tokens, 0/False for padding; if ``None``, all valid.
        gumbel_tau : float, default=1.0
            Scale for training-time Gumbel exploration in routers (selection only).
        router_temperature : float, default=1.0
            Temperature for *mixing* probabilities.
        select_temperature : Optional[float], default=None
            Temperature for *selection* logits (top-k). If None, uses `router_temperature`.

        Returns
        -------
        logits : Float[Array, "T V"]
            Token logits over vocabulary (tied unembedding).
        stats : Tuple[Dict[str, Any], ...]
            Tuple of hop statistics (length = number of routers).
        """
        # ====== Token validity mask ======
        T = ids.shape[0]
        token_mask: Bool[Array, "T"] = (
            jnp.ones((T,), dtype=bool)
            if attention_mask is None
            else attention_mask.astype(bool)
        )

        # ====== Embed + dropout ======
        h: Float[Array, "T d"] = jax.vmap(self.embed)(ids)  # (T, d)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        # ====== RoPE caches (cos, sin) ======
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_sin = rope_cos_sin(T, d_h, self.rope_base)  # each (T, d_h)

        # ====== Optional backbone (pre-routing) ======
        for mod in self.backbone:
            key, sub = jax.random.split(key)
            out = mod(h, cos_sin, token_mask, key=sub, inference=inference)
            # Module decides its own residual; we keep pads unchanged
            h = jnp.where(token_mask[:, None], out, h)

        # ====== Routing hops ======
        stats_all: List[Dict[str, Any]] = []
        for R in self.routers:
            key, sub = jax.random.split(key)
            h, st = self._hop(
                h,
                R,
                cos_sin,
                key=sub,
                inference=inference,
                token_mask=token_mask,
                gumbel_tau=gumbel_tau,
                router_temperature=router_temperature,
                select_temperature=select_temperature,
            )
            stats_all.append(st)

        # ====== Output: norm + tied unembedding ======
        h = jax.vmap(self.ln_out)(h)  # (T, d)
        logits: Float[Array, "T V"] = jax.vmap(lambda t: t @ self.embed.weight.T)(h)

        return logits, tuple(stats_all)

    # ---------------------------------------------------------------------
    # Stats collector
    # ---------------------------------------------------------------------
    def _stats(
        self,
        *,
        kept: Bool[Array, "E T"],
        probs: Float[Array, "T E"],
        mask: Bool[Array, "T E"],
        rho: Float[Array, "T 1"],
        token_mask: Bool[Array, "T"],
    ) -> Dict[str, Any]:
        """Collect interpretable hop statistics."""
        # ---- Loads & importance ----
        load = kept.sum(axis=1).astype(jnp.int32)  # (E,)
        importance_sum = probs.sum(axis=0)  # (E,)
        valid_count = token_mask.astype(jnp.int32).sum()
        importance_mean = jnp.where(
            valid_count > 0,
            importance_sum / valid_count,
            jnp.zeros_like(importance_sum),
        )

        # ---- Edge accounting ----
        selected_edges = mask.astype(jnp.int32).sum()  # scalar
        kept_edges = kept.astype(jnp.int32).sum()  # scalar
        cap_drop_frac_edges = jnp.where(
            selected_edges > 0,
            (selected_edges - kept_edges) / selected_edges,
            0.0,
        )

        # ---- Entropy per token (normalized by log E) ----
        p = probs
        p_sum = p.sum(axis=1, keepdims=True) + 1e-9
        p_norm = p / p_sum
        ent = -(p_norm * jnp.log(p_norm + 1e-9)).sum(axis=1)  # (T,)
        ent = ent / jnp.log(p.shape[1] + 1e-9)  # (T,)

        # ---- Effective top-k on valid tokens only ----
        eff_k_all = mask.astype(jnp.int32).sum(axis=1).astype(jnp.float32)  # (T,)
        valid_mask_f = token_mask.astype(jnp.float32)  # (T,)

        eff_topk_mean = jnp.where(
            valid_count > 0, (eff_k_all * valid_mask_f).sum() / valid_count, 0.0
        )
        eff_k_for_min = jnp.where(token_mask, eff_k_all, jnp.inf)
        eff_k_for_max = jnp.where(token_mask, eff_k_all, -jnp.inf)
        eff_topk_min = jnp.where(valid_count > 0, jnp.min(eff_k_for_min), 0.0)
        eff_topk_max = jnp.where(valid_count > 0, jnp.max(eff_k_for_max), 0.0)

        # ---- Capacity utilization (fraction per expert) ----
        cap = jnp.asarray(self.capacity, dtype=jnp.float32)
        cap_util = load.astype(jnp.float32) / jnp.maximum(cap, 1.0)

        # ---- Dispersion & utilization ----
        util = (load > 0).mean()
        load_norm = load / (jnp.sum(load) + 1e-9)
        load_std = jnp.std(load_norm)

        return dict(
            load=load,
            importance_sum=importance_sum,
            importance_mean=importance_mean,
            rho_mean=rho.mean(),
            rho_min=rho.min(),
            rho_max=rho.max(),
            entropy_mean=ent.mean(),
            entropy_min=ent.min(),
            entropy_max=ent.max(),
            util=util,
            load_std=load_std,
            cap_drop_frac_edges=cap_drop_frac_edges,
            selected_edges=jnp.asarray(selected_edges, jnp.int32),
            kept_edges=jnp.asarray(kept_edges, jnp.int32),
            eff_topk_mean=eff_topk_mean,
            eff_topk_min=eff_topk_min,
            eff_topk_max=eff_topk_max,
            cap_util_mean=cap_util.mean(),
            cap_util_min=cap_util.min(),
            cap_util_max=cap_util.max(),
        )
