# -----------------------------------------------------------------------------
# modules.py
# -----------------------------------------------------------------------------
# Generic Mixture-of-Experts execution with capacity-constrained slotting.
#
# This file purposely does **no routing** (no top-k / sampling). It receives
# (T, E) boolean masks & (T, E) weights from an external router, packs
# token-major inputs (T, ...) into expert/slot-major (E, C, ...) using a slot
# assignment, executes experts in vmapped groups, and combines outputs back to
# (T, ...) with the provided weights. It also returns capacity bookkeeping that
# the model can use for stats (kept edges, etc.).
#
# Key ideas:
# - Any array arg/kwarg whose leading dimension equals T will be packed to (E,C,...)
#   via the slot tensor. This lets you pass `hidden`, `cos`, `sin`, or a mask
#   in token-major form **without** writing expert-specific code.
# - Experts are heterogenous; modules with identical param-tree structures are
#   grouped and run with vmap for speed.
# - We inspect each group's __call__ signature once (outside JIT) to filter kwargs
#   so we never pass unsupported kwargs to modules (e.g., Attention may accept
#   `attention_mask`, FeedForward may not).
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import inspect

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------- helpers: signature grouping & stacking ----------


def _array_tree(mod):
    """Filter arrays (params/buffers) out of the module pytree."""
    return eqx.filter(mod, eqx.is_array)


def _static_tree(mod):
    """Filter statics (non-arrays) out of the module pytree."""
    return eqx.filter(mod, lambda x: not eqx.is_array(x))


def _sig_key(mod) -> Tuple[type, str]:
    """Lightweight signature key: (type, array-tree treedef).
    Modules with the same type & array shape structure will share a vmap group.
    """
    arr_tree = _array_tree(mod)
    return (type(mod), str(jax.tree_util.tree_structure(arr_tree)))


def _stack_group(mods: Sequence[eqx.Module]):
    """Stack params of same-shaped modules -> (E_g, ...), keep shared statics."""
    params = [_array_tree(m) for m in mods]
    static0 = _static_tree(mods[0])
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params)
    return stacked, static0


# ---------- slot building (capacity-aware token packing) ----------


def _build_slots(
    mask_te: jnp.ndarray, weight_te: jnp.ndarray, capacity: Optional[int]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Given routing mask (T,E) and weights (T,E), build slots (E,C,T) and 'kept' (E,T).

    #---------
    Shapes
      mask_te:   (T, E) bool    — selection mask from router.
      weight_te: (T, E) float32 — combination weights per token/expert (0 where unrouted).
      capacity:  None or int    — max tokens per expert (C).

    Returns
      slot_ect:  (E, C, T) float32 in {0,1}  — token-to-slot assignment one-hots.
      kept_et:   (E, T) bool                 — token kept by expert (not dropped by capacity).
      top_idx:   (E, C) int32                — token indices chosen for each slot (for sorting).
    """
    assert mask_te.dtype == jnp.bool_, "mask_te must be boolean (T,E)."
    T, E = mask_te.shape
    C = T if (capacity is None) else int(min(capacity, T))

    # Expert-major views
    m = mask_te.T                                      # (E, T)
    w = jnp.where(m, weight_te.T, -jnp.inf)            # (E, T) => invalid -> -inf

    # Select up to C tokens per expert by weight
    _, idx_top = jax.lax.top_k(w, C)                   # (E, C) token indices in [0, T)

    # Build raw slots then mask out invalid picks (if any)
    slot = jnp.zeros((E, C, T), dtype=w.dtype)         # (E, C, T)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], idx_top].set(1.0)
    slot = slot * m[:, None, :]                        # drop any selection that wasn't actually routed

    kept = slot.sum(axis=1).astype(bool)               # (E, T)

    # For causality-preserving execution, we sort each expert's slots by original time index
    # Compute per-slot token positions; inactive slots get sentinel T+1 then sorted stably
    pos_for_sort = jnp.where(slot.astype(bool), jnp.arange(T)[None, None, :], T + 1)  # (E,C,T)
    # First, get the position for each slot (argmax over T is safe since only one 1 per slot)
    slot_pos = jnp.argmax(pos_for_sort, axis=-1)       # (E, C)
    order = jnp.argsort(slot_pos, axis=1, stable=True) # (E, C)
    slot = jnp.take_along_axis(slot, order[:, :, None], axis=1)
    idx_sorted = jnp.take_along_axis(idx_top, order, axis=1)  # (E, C)

    return slot, kept, idx_sorted


# ---------- packing & combining ----------


def _dispatch_args(slot_ect: jnp.ndarray, *args, **kwargs):
    """For any array arg/kwarg whose leading dimension is T, pack to (E,C,...) via slot.

    #---------
    slot: (E, C, T) with 0/1 values.
    For non-arrays or arrays not starting with T, pass through unchanged.
    """

    def _pack(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1:
            T = slot_ect.shape[-1]
            if x.shape[0] == T:
                # (E,C,T) x (T,...) -> (E,C,...) ; tokens -> expert/slot
                return jnp.einsum("ect,t...->ec...", slot_ect, x)
        return x

    packed_args = tuple(_pack(a) for a in args)
    packed_kwargs = {k: _pack(v) for k, v in kwargs.items()}
    return packed_args, packed_kwargs


def _combine_tree(
    slot_ect: jnp.ndarray, weight_te: jnp.ndarray, expert_out_tree: Any
) -> Any:
    """Combine expert outputs back to token-major (T, ...), weighted by weight_te.

    For each array leaf with shape (E,C,...) combine via:
      y[t,...] = sum_e sum_c slot[e,c,t] * weight[t,e] * out[e,c,...]
    """

    def _combine_leaf(out_ecX: jnp.ndarray):
        # Flatten trailing dims for a clean einsum, then restore.
        rest = out_ecX.shape[2:]
        out_ecf = out_ecX.reshape(out_ecX.shape[:2] + (-1,))  # (E, C, F)
        tf = jnp.einsum("ecf,ect,te->tf", out_ecf, slot_ect, weight_te)  # (T, F)
        return tf.reshape((slot_ect.shape[-1],) + rest)

    return jax.tree_util.tree_map(_combine_leaf, expert_out_tree)


# ---------- kwarg policy (filter unsupported kwargs per group) ----------


def _call_kw_policy(mod_type: type):
    """Inspect mod_type.__call__ and decide which kwargs are allowed."""
    sig = inspect.signature(mod_type.__call__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    allowed = tuple(
        p.name
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return has_var_kw, set(allowed)


def _filter_kwargs_for_group(gkwargs: Dict[str, Any], has_var_kw: bool, allowed: set[str]) -> Dict[str, Any]:
    if has_var_kw:
        return gkwargs
    return {k: v for k, v in gkwargs.items() if k in allowed}


# ---------- Mixture of Experts (generic, single hop) ----------


class Modules(eqx.Module):
    """Single-hop, routing-driven execution over a *collection of modules*.

    Usage:
        mods = Modules(experts, capacity=64)
        out, aux = mods(
            *token_major_args,
            route_mask=mask_te,        # (T,E) bool
            route_weight=weight_te,    # (T,E) float
            key=key,
            inference=False,
            **token_major_kwargs,      # e.g., cos, sin, attention_mask
        )

    Notes:
      - This class is routing-agnostic. Provide mask/weights from a Router.
      - Capacity is enforced here. Kept assignments are exposed via `aux["kept"]`.
      - Any token-major arrays in args/kwargs (leading dim T) are packed automatically.
      - Per-group kwargs are filtered so modules never receive unsupported kwargs.
      - Randomness: one subkey per expert; if your module needs per-slot randomness,
        handle splitting internally from that expert key.
    """

    # Original expert list (kept for clarity / inspection)
    experts: Tuple[eqx.Module, ...]
    # Grouped execution metadata
    _group_idx: Tuple[jnp.ndarray, ...]     # per-group expert indices (E_g,)
    _group_params: Tuple[Any, ...]          # per-group stacked params (E_g, ...)
    _group_static: Tuple[Any, ...]          # per-group static trees
    _group_kw_has_var: Tuple[bool, ...]     # per-group kw policy
    _group_kw_allowed: Tuple[Tuple[str, ...], ...]
    capacity: Optional[int] = eqx.field(static=True)

    def __init__(self, experts: Sequence[eqx.Module], *, capacity: Optional[int] = None):
        # --------- store experts & capacity ---------
        self.experts = tuple(experts)
        self.capacity = capacity

        # --------- bucket modules by (type, param-tree-structure) for vmapped execution ---------
        buckets: Dict[Tuple[type, str], List[int]] = {}
        for i, m in enumerate(self.experts):
            buckets.setdefault(_sig_key(m), []).append(i)

        group_idx: List[jnp.ndarray] = []
        group_params: List[Any] = []
        group_static: List[Any] = []
        group_kw_has_var: List[bool] = []
        group_kw_allowed: List[Tuple[str, ...]] = []

        for (mtype, _), idxs in sorted(buckets.items(), key=lambda kv: kv[1][0]):
            mods = [self.experts[i] for i in idxs]
            p, s = _stack_group(mods)
            group_idx.append(jnp.asarray(idxs, dtype=jnp.int32))
            group_params.append(p)
            group_static.append(s)
            has_var, allowed = _call_kw_policy(mtype)
            group_kw_has_var.append(has_var)
            group_kw_allowed.append(tuple(sorted(allowed)))

        self._group_idx = tuple(group_idx)
        self._group_params = tuple(group_params)
        self._group_static = tuple(group_static)
        self._group_kw_has_var = tuple(group_kw_has_var)
        self._group_kw_allowed = tuple(group_kw_allowed)

    @eqx.filter_jit
    def __call__(
        self,
        *args,
        route_mask: jnp.ndarray,
        route_weight: Optional[jnp.ndarray] = None,
        key: Optional[jax.Array] = None,
        inference: bool = True,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Args:
        *args/**kwargs: token-major inputs; any array with leading dim T is packed.
        route_mask:     (T,E) bool — which token is sent to which expert.
        route_weight:   (T,E) float — combination weights per token/expert (0 where unrouted).
                        If None, defaults to route_mask.astype(float).
        key:            PRNGKey used to derive per-expert keys inside each group.
        inference:      forwarded flag (filtered to modules that accept it).

        Returns:
        combined: pytree of arrays with token-major leading dim T (mirrors experts' outputs).
        aux:      dict with:
                    - "slot": (E,C,T) float in {0,1}, token-to-slot assignment
                    - "kept": (E,T) bool, tokens kept by capacity
                    - "idx":  (E,C) int32, token indices per slot (sorted by time)
        """
        # --------- validate & sanitize inputs --------------------------------------
        mask_te = route_mask.astype(bool)                        # (T, E)
        T, E = mask_te.shape
        w_te = mask_te.astype(jnp.float32) if route_weight is None else route_weight
        w_te = jnp.where(mask_te, w_te, 0.0)                     # zero where unrouted

        # --------- capacity-limited slot construction ------------------------------
        # slot_ect ∈ {0,1}: (E, C, T)  kept_et: (E, T)  idx_ec: (E, C)
        slot_ect, kept_et, idx_ec = _build_slots(mask_te, w_te, self.capacity)

        # --------- pack token-major args/kwargs to expert/slot-major ---------------
        packed_args, packed_kwargs = _dispatch_args(slot_ect, *args, **kwargs)

        # Seed for per-group/per-expert key splitting
        base_key = jax.random.PRNGKey(0) if key is None else key

        # --------- execute each group with vmap over experts in that group ---------
        group_outs = []
        for gi, (gparams, gstatic, idxs, has_var, allowed) in enumerate(
            zip(
                self._group_params,
                self._group_static,
                self._group_idx,
                self._group_kw_has_var,
                self._group_kw_allowed,
            )
        ):
            E_g = idxs.shape[0]

            # Slice inputs for this group (only arrays with leading dim E get sliced)
            def _slice_in(x):
                if isinstance(x, jnp.ndarray) and x.ndim >= 2 and x.shape[0] == E:
                    return x[idxs]                                 # (E_g, C, ...)
                return x                                           # passthrough

            g_args = tuple(_slice_in(a) for a in packed_args)
            g_kwargs = {k: _slice_in(v) for k, v in packed_kwargs.items()}

            # Determine whether this group's modules accept a 'key' kwarg
            want_key = ("key" in allowed) or has_var
            if want_key:
                # Per-expert keys as a **positional** vmapped arg (shape (E_g, 2))
                sub_keys = jax.random.split(jax.random.fold_in(base_key, gi), E_g)
                g_args = g_args + (sub_keys,)

            # Always provide 'inference' if supported
            if ("inference" in allowed) or has_var:
                g_kwargs = dict(g_kwargs, inference=inference)

            # Filter kwargs to only what's accepted by this group's __call__
            g_kwargs = _filter_kwargs_for_group(g_kwargs, has_var, set(allowed))

            # Build in_axes for vmap:
            # - params always mapped over axis 0
            # - args mapped if they have leading dim E_g
            in_axes_params = (0,)
            in_axes_args: Tuple[Optional[int], ...] = tuple(
                0 if (isinstance(a, jnp.ndarray) and a.ndim >= 1 and a.shape[0] == E_g) else None
                for a in g_args
            )

            # Per-expert call: if we appended keys, the **last positional** is the key
            if want_key:
                def _call_one(param_i, *aa, **kk):
                    *aa_no_key, key_i = aa
                    mod = eqx.combine(param_i, gstatic)
                    return mod(*aa_no_key, key=key_i, **kk)
            else:
                def _call_one(param_i, *aa, **kk):
                    mod = eqx.combine(param_i, gstatic)
                    return mod(*aa, **kk)

            out_tree = jax.vmap(
                lambda p, *aa: _call_one(p, *aa, **g_kwargs),
                in_axes=in_axes_params + in_axes_args,
                out_axes=0,
            )(gparams, *g_args)                                    # leaves: (E_g, C, ...)

            group_outs.append((idxs, out_tree))

        # --------- stitch group outputs back to full (E,C,...) layout --------------
        def _zeros_like_group(gout):
            return jax.tree_util.tree_map(lambda x: jnp.zeros((E,) + x.shape[1:], x.dtype), gout)

        accum = None
        for idxs, gout in group_outs:
            if accum is None:
                accum = _zeros_like_group(gout)

            def _scatter(dst, src):
                return dst.at[idxs].set(src)

            accum = jax.tree_util.tree_map(_scatter, accum, gout)  # (E, C, ...)

        # --------- combine back to token-major using slots + weights ----------------
        combined = _combine_tree(slot_ect, w_te, accum)            # (T, ...)

        aux = {"slot": slot_ect, "kept": kept_et, "idx": idx_ec}
        return combined, aux
