from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Bool, Array

from dna.modules import Embedding, Dropout, RMSNorm, rope_cos_sin
from dna.routing import LinearRouter


def _sig(mod: eqx.Module) -> Tuple[str, str, str]:
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
    params = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *params), static


def _capacity_select(
    mask_te: jnp.ndarray,
    score_te: jnp.ndarray,
    capacity: int,
):
    m = mask_te.T
    g = jnp.where(m, score_te.T, -jnp.inf)
    E, T = g.shape
    C = int(min(capacity, T))
    _, top_idx = jax.lax.top_k(g, C)
    slot = jnp.zeros((E, C, T), dtype=g.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], top_idx].set(1.0)
    slot = slot * m[:, None, :].astype(slot.dtype)
    kept = (slot.sum(axis=1) > 0).astype(bool)
    return slot, kept, top_idx


class DNA(eqx.Module):
    embed: Embedding
    dropout: Dropout
    ln_out: RMSNorm
    routers: Tuple[LinearRouter, ...]
    backbone: Tuple[eqx.Module, ...]
    groups: Tuple[Dict[str, Any], ...]
    capacity: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    norm_after_capacity: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        modules: Tuple[eqx.Module, ...],
        router_cls: type[LinearRouter],
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
        key,
    ):
        self.capacity = int(capacity)
        self.n_heads = int(n_heads)
        self.rope_base = float(rope_base)
        k_embed, k_routers = jax.random.split(key, 2)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for idx, mod in enumerate(modules):
            sig = _sig(mod)
            entry = buckets.setdefault(sig, {"idx": [], "mods": []})
            entry["idx"].append(idx)
            entry["mods"].append(mod)

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
        grouped.sort(key=lambda d: int(d["idx"][0]))
        self.groups = tuple(grouped)
        self.backbone = tuple(backbone) if backbone is not None else tuple()

        total_experts = len(modules)
        router_keys = jax.random.split(k_routers, n_hops)
        self.routers = tuple(
            router_cls(d_model, total_experts, topk, dropout, norm_probs, key=k)
            for k in router_keys
        )
        self.norm_after_capacity = norm_after_capacity

    def _hop(
        self,
        h: Float[Array, "T d"],
        router: LinearRouter,
        cos_sin: Tuple[Float[Array, "T d_h"], Float[Array, "T d_h"]],
        *,
        key,
        inference: bool,
        token_mask: Bool[Array, "T"],
        gumbel_tau: float,
        router_temp: float,
        select_temp: Optional[float],
        return_stats: bool,
    ) -> Tuple[Float[Array, "T d"], Dict[str, Any]]:
        k_route, k_exec = jax.random.split(key)
        mask_full, probs_full, logits_clean, logits_sel = router(
            h,
            key=k_route,
            inference=inference,
            gumbel_tau=gumbel_tau,
            router_temp=router_temp,
            select_temp=select_temp,
            token_mask=token_mask,
        )

        slot, kept, top_idx = _capacity_select(mask_full, logits_sel, self.capacity)
        cos, sin = cos_sin
        xin = jnp.einsum("ect,td->ecd", slot, h)
        cosr = jnp.einsum("ect,td->ecd", slot, cos)
        sinr = jnp.einsum("ect,td->ecd", slot, sin)
        active = slot.sum(-1) > 0
        pos_for_sort = jnp.where(active, top_idx, h.shape[0] + 1)
        order = jnp.argsort(pos_for_sort, axis=1, stable=True)

        def _take(a):
            return jnp.take_along_axis(a, order[:, :, None], axis=1)

        xin = _take(xin)
        cosr = _take(cosr)
        sinr = _take(sinr)
        active = jnp.take_along_axis(active, order, axis=1)
        slot = jnp.take_along_axis(slot, order[:, :, None], 1)

        E, C, d = xin.shape
        expert_out = jnp.zeros((E, C, d), dtype=xin.dtype)
        for gi, g in enumerate(self.groups):
            sub_keys = jax.random.split(jax.random.fold_in(k_exec, gi), len(g["idx"]))
            inp = xin[g["idx"]]
            c = cosr[g["idx"]]
            s = sinr[g["idx"]]
            am = active[g["idx"]]

            def _run(p, x, c1, s1, am1, k1):
                mod = eqx.combine(p, g["static"])
                return mod(x, (c1, s1), am1, key=k1, inference=inference)

            out_g = jax.vmap(_run)(g["params"], inp, c, s, am, sub_keys)
            expert_out = expert_out.at[g["idx"]].set(out_g)

        kept_t = kept.T
        combine_w = jnp.where(kept_t, probs_full, 0.0)
        if self.norm_after_capacity:
            denom = combine_w.sum(axis=1, keepdims=True)
            combine_w = combine_w / jnp.clip(denom, 1e-9, None)

        rho = combine_w.sum(axis=1, keepdims=True)
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)
        h_next = h + combine - rho * h
        h_next = jnp.where(token_mask[:, None], h_next, h)

        stats = {}
        if return_stats:
            stats = self._stats(
                kept=kept,
                probs=probs_full,
                mask=mask_full,
                rho=rho,
                token_mask=token_mask,
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
        return_stats: bool = False,
    ) -> Tuple[Float[Array, "T V"], Tuple[Dict[str, Any], ...]]:
        T = ids.shape[0]
        token_mask: Bool[Array, "T"] = (
            jnp.ones((T,), dtype=bool) if mask is None else mask.astype(bool)
        )
        h: Float[Array, "T d"] = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_sin = rope_cos_sin(T, d_h, self.rope_base)

        for mod in self.backbone:
            key, sub = jax.random.split(key)
            out = mod(h, cos_sin, token_mask, key=sub, inference=inference)
            h = jnp.where(token_mask[:, None], out, h)

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
                return_stats=return_stats,
            )
            stats_all.append(st)

        h = jax.vmap(self.ln_out)(h)
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
        load = kept.sum(axis=1).astype(jnp.int32)
        importance = probs.sum(axis=0)
        p = probs
        p_sum = p.sum(axis=1, keepdims=True) + 1e-9
        p_norm = p / p_sum
        entropy = -(p_norm * jnp.log(p_norm + 1e-9)).sum(axis=1)
        entropy = entropy / jnp.log(p.shape[1] + 1e-9)
        selected_edges = mask.astype(jnp.int32).sum()
        kept_edges = kept.astype(jnp.int32).sum()
        eff_topk = mask.astype(jnp.int32).sum(axis=1)
        return dict(
            load=load,
            importance=importance,
            rho=rho[:, 0],
            entropy=entropy,
            selected_edges=selected_edges,
            kept_edges=kept_edges,
            eff_topk=eff_topk,
            routing_probs=probs,
            token_mask=token_mask,
        )
