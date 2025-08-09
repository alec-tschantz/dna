from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Given routing mask (T,E) and weights (T,E), build slots (E,C,T) and 'kept' (E,T).

    - For each expert e, pick up to C tokens with largest weight (ties: stable).
    - Returns:
        slot[e,c,t] in {0,1} assigns token t to slot c for expert e.
        kept[e,t]   indicates token t was kept for expert e (after capacity).
    """
    assert mask_te.dtype == jnp.bool_, "mask_te must be boolean (T,E)."
    T, E = mask_te.shape
    C = T if (capacity is None) else int(min(capacity, T))

    m = mask_te.T  # (E,T)
    w = jnp.where(m, weight_te.T, -jnp.inf)  # (E,T)

    # pick top-C tokens per expert by weight
    # if C==T, jax.lax.top_k still works but is a bit heavier; short-circuit
    if C == T:
        # one slot per token, ordered by time (stable)
        # put tokens in ascending index order
        score = jnp.where(m, jnp.arange(T)[None, :], T + 1)  # (E,T)
        order = jnp.argsort(score, axis=1, stable=True)  # (E,T)
        ar = jnp.arange(T)
        slot = jnp.zeros((E, T, T), dtype=w.dtype)
        slot = slot.at[jnp.arange(E)[:, None], ar[None, :], order].set(1.0)
        kept = m
        return slot, kept

    # capacity < T: choose top-C by weight, then sort by original time index for causality
    _, idx_top = jax.lax.top_k(w, C)  # (E, C), indices in [0,T)
    # build one-hot slots
    slot = jnp.zeros((E, C, T), dtype=w.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], idx_top].set(1.0)
    # drop any that weren't actually routed (mask)
    slot = slot * m[:, None, :]
    kept = slot.sum(axis=1).astype(bool)  # (E,T)
    # stable order by token time index
    pos = jnp.where(kept, jnp.argmax(slot, axis=1), T + 1)  # (E,T) time indices
    order = jnp.argsort(pos, axis=1, stable=True)[:, :C]  # (E,C)
    slot = jnp.take_along_axis(slot, order[:, :, None], axis=1)
    return slot, kept


def _dispatch_args(slot_ect: jnp.ndarray, *args, **kwargs):
    """For any array arg/kwarg whose leading dimension is T, pack to (E,C,...) via slot.

    slot: (E,C,T) with 0/1 values.
    For non-arrays or arrays not starting with T, pass through unchanged.
    """

    def _pack(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1:
            T = slot_ect.shape[-1]
            if x.shape[0] == T:
                # einsum: (E,C,T) x (T,...) -> (E,C,...)
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
        # (E,C,rest...) -> (T,rest...)
        # First gather (E,C,T) weighting term -> multiply and sum over E,C.
        # We multiply slot and weight separately: slot[e,c,t] * weight[t,e].
        # Reshape rest to flat for a clean einsum, then restore.
        rest = out_ecX.shape[2:]
        out_ecf = out_ecX.reshape(out_ecX.shape[:2] + (-1,))  # (E,C,F)
        # term1: sum over e,c of out * slot -> (T,F)
        t1 = jnp.einsum("ecf,ect->tf", out_ecf, slot_ect)
        # term2: weight across experts per token -> multiply along e then sum:
        # we need sum_e weight[t,e]* (sum_c out*slot for that e?). The previous t1 already summed over e,c.
        # But t1 did a full sum over e,c; to apply per-expert weights correctly, combine in one shot:
        # Do it properly: bring weight in before summing over e.
        # Recompute with a single einsum:
        tf = jnp.einsum("ecf,ect,te->tf", out_ecf, slot_ect, weight_te)
        return tf.reshape((slot_ect.shape[-1],) + rest)

    return jax.tree_util.tree_map(_combine_leaf, expert_out_tree)


# ---------- Mixture of Experts (single hop) ----------


class MixtureOfExperts(eqx.Module):
    """Single-hop, routing-driven MoE.

    Usage:
        moe = MixtureOfExperts(experts, capacity=64)
        # route_mask: (T,E) bool, route_weight: (T,E) float (0 for unrouted).
        y = moe(*args, route_mask=mask, route_weight=weight, key=key, inference=False)

    Notes:
      - You control routing externally; we never sample or pick top-k for you.
      - We *do* enforce per-expert capacity to keep compute tight (optional).
      - Any array in *args/**kwargs with leading dim T is packed and dispatched.
      - Experts may be heterogeneous; ones with the same param-tree layout are vmapped together.
      - Outputs can be arbitrary pytrees of arrays; all leaves are combined token-wise.
    """

    # raw experts (kept for clarity / rebuild)
    experts: Tuple[eqx.Module, ...]
    # vmapped groups
    _group_idx: Tuple[jnp.ndarray, ...]  # per-group expert indices (E_g,)
    _group_params: Tuple[Any, ...]  # per-group stacked params (E_g, ...)
    _group_static: Tuple[Any, ...]  # per-group static trees
    capacity: Optional[int] = eqx.field(static=True)

    def __init__(
        self, experts: Sequence[eqx.Module], *, capacity: Optional[int] = None
    ):
        self.experts = tuple(experts)
        self.capacity = capacity

        # group by signature for vectorized execution
        buckets: Dict[Tuple[type, str], List[int]] = {}
        for i, m in enumerate(self.experts):
            buckets.setdefault(_sig_key(m), []).append(i)

        group_idx, group_params, group_static = [], [], []
        for _, idxs in sorted(buckets.items(), key=lambda kv: kv[1][0]):
            mods = [self.experts[i] for i in idxs]
            p, s = _stack_group(mods)
            group_idx.append(jnp.asarray(idxs, dtype=jnp.int32))
            group_params.append(p)
            group_static.append(s)

        self._group_idx = tuple(group_idx)
        self._group_params = tuple(group_params)
        self._group_static = tuple(group_static)

    @eqx.filter_jit
    def __call__(
        self,
        *args,
        route_mask: jnp.ndarray,
        route_weight: Optional[jnp.ndarray] = None,
        key: Optional[jax.Array] = None,
        inference: bool = True,
        **kwargs,
    ):
        """
        Args:
          *args/**kwargs: token-major inputs; any array with leading dim T is dispatched.
          route_mask: bool (T,E), which token is sent to which expert.
          route_weight: float (T,E), combination weights per token/expert (0 for unrouted).
                        If None, defaults to route_mask.astype(float).
          key: forwarded to experts that need randomness.
          inference: forwarded flag.

        Returns:
          Combined outputs with token-major leading dim T (pytree mirrors experts' outputs).
        """
        mask_te = route_mask.astype(bool)
        T, E = mask_te.shape
        w_te = mask_te.astype(jnp.float32) if route_weight is None else route_weight
        # Safety: zero-out weights where mask==False (lets you pass logits probs directly)
        w_te = jnp.where(mask_te, w_te, 0.0)

        # Build (E,C,T) assignment slots and kept mask per expert.
        slot_ect, _kept_et = _build_slots(mask_te, w_te, self.capacity)  # slot ∈ {0,1}

        # Pack all inputs for only the active tokens.
        packed_args, packed_kwargs = _dispatch_args(slot_ect, *args, **kwargs)

        # Execute all groups in parallel (vmap over experts within a group).
        def _run_group(gparams, gstatic, idxs):
            # Select slice of per-group inputs for those experts.
            # For packed arrays: we need (E_g, C, ...) -> pass as-is.
            # For non-arrays: pass through (same value to each expert).
            def _slice_in(x):
                if (
                    isinstance(x, jnp.ndarray)
                    and x.ndim >= 2
                    and x.shape[0] == slot_ect.shape[0]
                ):
                    return x[idxs]  # (E_g, C, ...)
                return x

            g_args = tuple(_slice_in(a) for a in packed_args)
            g_kwargs = {k: _slice_in(v) for k, v in packed_kwargs.items()}

            def _call_one(params_i, *aa, **kk):
                m = eqx.combine(params_i, gstatic)
                # Experts can accept arbitrary signatures; forward.
                return m(*aa, **kk)

            # vmap over experts in the group; C stays as a batch/time-like axis inside args.
            out_tree = jax.vmap(
                lambda p, *aa, **kk: _call_one(p, *aa, **kk),
                in_axes=(0,) + (0,) * len(g_args),
                out_axes=0,
            )(gparams, *g_args, **g_kwargs)
            # out_tree leaves have shape (E_g, C, ...)
            return out_tree

        # Run each group and stitch back to full (E,C,...) layout
        group_outs = []
        for gparams, gstatic, idxs in zip(
            self._group_params, self._group_static, self._group_idx
        ):
            gout = _run_group(gparams, gstatic, idxs)
            group_outs.append((idxs, gout))

        # Allocate empty (E,C,...) pytree and scatter group outputs into it
        # Create a template by running a tiny fake combine on zeros (lazy-free).
        def _zeros_like_group(gout):
            return jax.tree_util.tree_map(
                lambda x: jnp.zeros((E,) + x.shape[1:], x.dtype), gout
            )

        accum = None
        for idxs, gout in group_outs:
            if accum is None:
                accum = _zeros_like_group(gout)

            def _scatter(dst, src):
                return dst.at[idxs].set(src)

            accum = jax.tree_util.tree_map(
                _scatter, accum, gout
            )  # (E,C,...) across leaves

        # Combine back to token-major using slots + weights.
        combined = _combine_tree(slot_ect, w_te, accum)
        return combined
