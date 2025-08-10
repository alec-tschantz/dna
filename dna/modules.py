# modules.py
# -----------------------------------------------------------------------------
# Generic, routing-agnostic execution with capacity-constrained slotting.
#
# 
# Key properties:
#   • Any token-major arg/kwarg (leading dim = T) is automatically packed to (E, C, ...)
#   • Heterogeneous experts are grouped by (type, param-structure) and vmapped per group
#   • Capacity is enforced here; dropped assignments are reflected in `aux["kept"]`
#   • Zero branching API: one path for both train/eval; modules may receive `inference`
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import inspect

import jax
import jax.numpy as jnp
import equinox as eqx


# ---- grouping utilities ----------------------------------------------------


def _array_tree(mod):
    return eqx.filter(mod, eqx.is_array)


def _static_tree(mod):
    return eqx.filter(mod, lambda x: not eqx.is_array(x))


def _sig_key(mod) -> Tuple[type, str]:
    arr_tree = _array_tree(mod)
    return (type(mod), str(jax.tree_util.tree_structure(arr_tree)))


def _stack_group(mods: Sequence[eqx.Module]):
    params = [_array_tree(m) for m in mods]
    static0 = _static_tree(mods[0])
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params)
    return stacked, static0


# ---- capacity-aware slot construction -------------------------------------


def _build_slots(mask_te: jnp.ndarray, weight_te: jnp.ndarray, capacity: Optional[int]):
    """Construct (E, C, T) one-hot slots and keep mask.

    Args:
        mask_te:   (T, E) bool
        weight_te: (T, E) float (0 where unrouted)
        capacity:  int | None
    Returns:
        slot_ect:  (E, C, T) float in {0,1}
        kept_et:   (E, T) bool
        idx_ec:    (E, C) int32 → token indices per slot (sorted by time)
    """
    assert mask_te.dtype == jnp.bool_, "route_mask must be boolean"
    T, E = mask_te.shape
    C = T if capacity is None else int(min(capacity, T))

    m_et = mask_te.T  # (E, T)
    w_et = jnp.where(m_et, weight_te.T, -jnp.inf)  # (E, T)

    # Top‑C by weight per expert
    _, idx_top = jax.lax.top_k(w_et, C)  # (E, C)

    # Scatter into slots
    slot = jnp.zeros((E, C, T), dtype=w_et.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], idx_top].set(1.0)
    slot = slot * m_et[:, None, :]  # mask out non‑selected tokens

    kept_et = slot.sum(axis=1).astype(bool)  # (E, T)

    # Stable, causal order by original time index 
    pos = jnp.where(slot.astype(bool), jnp.arange(T)[None, None, :], T + 1)  # (E, C, T)
    slot_pos = jnp.argmax(pos, axis=-1)  # (E, C)
    order = jnp.argsort(slot_pos, axis=1, stable=True)  # (E, C)
    slot = jnp.take_along_axis(slot, order[:, :, None], axis=1)
    idx_sorted = jnp.take_along_axis(idx_top, order, axis=1)
    return slot, kept_et, idx_sorted


# ---- pack & combine --------------------------------------------------------


def _dispatch_args(slot_ect: jnp.ndarray, *args, **kwargs):
    """Pack token-major arrays (leading dim T) to (E, C, ...)."""

    def _pack(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 1:
            T = slot_ect.shape[-1]
            if x.shape[0] == T:
                return jnp.einsum("ect,t...->ec...", slot_ect, x)
        return x

    packed_args = tuple(_pack(a) for a in args)
    packed_kwargs = {k: _pack(v) for k, v in kwargs.items()}
    return packed_args, packed_kwargs


def _combine_tree(
    slot_ect: jnp.ndarray, weight_te: jnp.ndarray, expert_out_tree: Any
) -> Any:
    """Combine (E, C, ...) expert outputs back to token-major (T, ...)."""

    def _combine_leaf(out_ecX: jnp.ndarray):
        rest = out_ecX.shape[2:]
        out_ecf = out_ecX.reshape(out_ecX.shape[:2] + (-1,))
        tf = jnp.einsum("ecf,ect,te->tf", out_ecf, slot_ect, weight_te)
        return tf.reshape((slot_ect.shape[-1],) + rest)

    return jax.tree_util.tree_map(_combine_leaf, expert_out_tree)


# ---- per-group kwargs policy ----------------------------------------------


def _call_kw_policy(mod_type: type):
    sig = inspect.signature(mod_type.__call__)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    allowed = tuple(
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return has_var_kw, set(allowed)


def _filter_kwargs_for_group(
    gkwargs: Dict[str, Any], has_var_kw: bool, allowed: set[str]
) -> Dict[str, Any]:
    return gkwargs if has_var_kw else {k: v for k, v in gkwargs.items() if k in allowed}


# ---- main class ------------------------------------------------------------


class ModuleGroup(eqx.Module):
    """Capacity-aware execution over a *collection of experts*.

    Usage:
        group = ModuleGroup(experts, capacity=64)
        y, aux = group(
            x_td,                     # token-major inputs (T, ...)
            route_mask=mask_te,       # (T, E) bool
            route_weight=weight_te,   # (T, E) float (0 where unrouted)
            key=key,
            inference=False,
            # plus any token-major kwargs your modules support, e.g. cos, sin, attention_mask
        )

    Returns `y` with token-major leading dimension and `aux` with slot bookkeeping.
    """

    experts: Tuple[eqx.Module, ...]
    _group_idx: Tuple[jnp.ndarray, ...]
    _group_params: Tuple[Any, ...]
    _group_static: Tuple[Any, ...]
    _group_kw_has_var: Tuple[bool, ...]
    _group_kw_allowed: Tuple[Tuple[str, ...], ...]
    capacity: Optional[int] = eqx.field(static=True)

    def __init__(
        self, experts: Sequence[eqx.Module], *, capacity: Optional[int] = None
    ):
        self.experts = tuple(experts)
        self.capacity = capacity

        # Bucket by (type, param‑shapes) so vmaps are shape‑stable & fast
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
        route_mask: jnp.ndarray,  # (T, E) bool
        route_weight: Optional[jnp.ndarray] = None,  # (T, E) float
        key: Optional[jax.Array] = None,
        inference: bool = True,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        mask_te = route_mask.astype(bool)
        T, E = mask_te.shape
        w_te = (
            mask_te.astype(jnp.float32)
            if route_weight is None
            else jnp.where(mask_te, route_weight, 0.0)
        )

        # Build slots (capacity enforcement lives here)
        slot_ect, kept_et, idx_ec = _build_slots(mask_te, w_te, self.capacity)

        # Pack token-major inputs into expert/slot-major
        packed_args, packed_kwargs = _dispatch_args(slot_ect, *args, **kwargs)

        base_key = jax.random.PRNGKey(0) if key is None else key
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
            E_g = int(idxs.shape[0])

            def _slice_in(x):
                if isinstance(x, jnp.ndarray) and x.ndim >= 2 and x.shape[0] == E:
                    return x[idxs]  # (E_g, C, ...)
                return x

            g_args = tuple(_slice_in(a) for a in packed_args)
            g_kwargs = {k: _slice_in(v) for k, v in packed_kwargs.items()}

            want_key = ("key" in allowed) or has_var
            if want_key:
                subkeys = jax.random.split(jax.random.fold_in(base_key, gi), E_g)
                g_args = g_args + (subkeys,)

            if ("inference" in allowed) or has_var:
                g_kwargs = dict(g_kwargs, inference=inference)

            g_kwargs = _filter_kwargs_for_group(g_kwargs, has_var, set(allowed))

            in_axes_params = (0,)
            in_axes_args = tuple(
                (
                    0
                    if (
                        isinstance(a, jnp.ndarray) and a.ndim >= 1 and a.shape[0] == E_g
                    )
                    else None
                )
                for a in g_args
            )

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
            )(
                gparams, *g_args
            )  # leaves shaped (E_g, C, ...)

            group_outs.append((idxs, out_tree))

        # Stitch groups back to (E, C, ...)
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

            accum = jax.tree_util.tree_map(_scatter, accum, gout)

        # Combine to token-major
        combined = _combine_tree(slot_ect, w_te, accum)
        aux = {"slot": slot_ect, "kept": kept_et, "idx": idx_ec}
        return combined, aux

