import math
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax.nn as jnn

import equinox as eqx

from dna.nn import (
    Embedding,
    Dropout,
    RMSNorm,
    Attention,
    FeedForward,
    Identity,
    rope_cos_sin,
    rotate_half,
)


def _softmax_topk(logits, k):
    soft = jnn.softmax(logits, axis=-1)
    _, top = jax.lax.top_k(logits, k)
    hard = jnn.one_hot(top, logits.shape[-1]).sum(axis=-2)
    gate = jax.lax.stop_gradient(hard - soft) + soft  # straight‑through estimator
    mask = hard.astype(bool)
    gate = jnp.where(mask, gate, 0.0)
    gate = gate / jnp.clip(gate.sum(-1, keepdims=True), 1e-9)
    return mask, gate


def _dispatch(mask, weight, n_exp, capacity):
    one_hot = mask.T.astype(weight.dtype)  # [E, T]
    # TODO
    rank = jnp.cumsum(one_hot, axis=1) - 1
    within_cap = rank < capacity
    slot = jax.nn.one_hot(jnp.clip(rank, 0, capacity - 1), capacity, dtype=weight.dtype)
    slot *= within_cap[..., None]
    slot = slot.transpose(0, 2, 1)  # [E, C, T]
    dispatch = slot * (weight.T * within_cap)[:, None, :]
    return slot, dispatch  # both [E, C, T]


def _sig(m):
    st = jax.tree_util.tree_structure(eqx.filter(m, eqx.is_array))
    static = str(eqx.filter(m, lambda x: not eqx.is_array(x)))
    return type(m).__name__, str(st), static


def _stack(mods):
    arr = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *arr), static


# -----------------------------------------------------------------------------
#  Router
# -----------------------------------------------------------------------------


class Router(eqx.Module):
    proj: eqx.nn.Linear
    id_bias: float = eqx.field(static=True)
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, id_bias: float, key):
        self.k = k
        self.id_bias = id_bias
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)

    def __call__(self, h):
        logits = jax.vmap(self.proj)(h)  # [T, E]
        # Paper §2.3: ‘identity expert receives an additive bias β’ (identity_bias)
        logits = logits.at[:, -1].add(self.id_bias)
        return _softmax_topk(logits, self.k)  # mask, weight


# -----------------------------------------------------------------------------
#  DNA model
# -----------------------------------------------------------------------------


class DNA(eqx.Module):
    embed: Embedding
    dropout: Dropout
    backbone: Tuple[eqx.Module, ...]
    routers: Tuple[Router, ...]
    groups: Tuple[Dict[str, Any], ...]
    ln_out: RMSNorm

    # Static fields
    n_modules: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    # ---------------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------------

    def __init__(
        self,
        vocab: int,
        d_model: int,
        n_heads: int,
        n_modules: int,
        capacity: int,
        topk: int,
        n_hops: int,
        mlp_mult: int,
        dropout: float,
        rope_base: float,
        num_backbone: int,
        identity_bias: float,
        key,
    ):
        self.n_modules = n_modules
        self.capacity = capacity
        self.n_heads = n_heads
        self.rope_base = rope_base

        # ---------- global components ----------
        k_embed, k_modules, k_routers, k_backbone = jax.random.split(key, 4)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        # ---------- build trainable experts ----------
        module_keys = jax.random.split(k_modules, n_modules)
        experts = []
        for i, k_i in enumerate(module_keys):
            if i % 2 == 0:
                experts.append(Attention(d_model, n_heads, dropout, key=k_i))
            else:
                experts.append(FeedForward(d_model, mlp_mult, dropout, key=k_i))

        # ---------- group by signature ----------
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for idx, mod in enumerate(experts):
            s = _sig(mod)
            entry = buckets.setdefault(s, {"idx": [], "mods": []})
            entry["idx"].append(idx)
            entry["mods"].append(mod)
        grouped = []
        for b in buckets.values():
            params, st = _stack(b["mods"])
            grouped.append(
                dict(idx=jnp.array(b["idx"], jnp.int32), params=params, static=st)
            )
        grouped.sort(key=lambda d: int(d["idx"][0]))
        self.groups = tuple(grouped)

        # ---------- backbone ----------
        if num_backbone > 0:
            bb_keys = jax.random.split(k_backbone, num_backbone)
            backbone_tmp = []
            for i in range(num_backbone):
                if i % 2 == 0:
                    backbone_tmp.append(
                        Attention(d_model, n_heads, dropout, key=bb_keys[i])
                    )
                else:
                    backbone_tmp.append(
                        FeedForward(d_model, mlp_mult, dropout, key=bb_keys[i])
                    )
            self.backbone = tuple(backbone_tmp)
        else:
            self.backbone = tuple()

        # ---------- routers (one per hop) ----------
        router_keys = jax.random.split(k_routers, n_hops)
        total_experts = n_modules + 1  # +1 for identity expert
        self.routers = tuple(
            Router(d_model, total_experts, topk, id_bias=identity_bias, key=k)
            for k in router_keys
        )

    # ---------------------------------------------------------------------
    # Forward pass (Section 2 Alg. 1)
    # ---------------------------------------------------------------------

    def _hop(self, h, router: Router, cos, sin, *, key, inference):
        mask, weight = router(h)  # [T, E]
        n_exp_total = weight.shape[-1]
        # [E, C, T]
        slot, dispatch = _dispatch(mask, weight, n_exp_total, self.capacity)

        # Gather token blocks for all experts
        xin = jnp.einsum("ect,td->ecd", dispatch, h)  # [E, C, d]
        cos_r = jnp.einsum("ect,td->ecd", slot, cos)
        sin_r = jnp.einsum("ect,td->ecd", slot, sin)

        # Apply *trainable* experts (indices 0…n_modules‑1)
        outs = []
        for g in self.groups:
            n_g = len(g["idx"])
            sub_keys = jax.random.split(key, n_g)
            inp = xin[g["idx"]]
            c = cos_r[g["idx"]]
            s = sin_r[g["idx"]]
            out_g = jax.vmap(
                lambda p, x, c1, s1, k1: eqx.combine(p, g["static"])(
                    x, c1, s1, key=k1, inference=inference
                )
            )(g["params"], inp, c, s, sub_keys)
            outs.append(out_g)
        expert_out = jnp.concatenate(outs, axis=0)  # [n_modules, C, d]

        # Combine back to tokens, *excluding* identity expert (last row)
        combine = jnp.einsum("ecd,ect->td", expert_out, dispatch[:-1])

        # ρ in Eq. (8) is the *sum of trainable‑expert weights* (skip expert excluded)
        rho = weight[:, :-1].sum(axis=1, keepdims=True)
        h_next = h + combine - rho * h

        stats = dict(
            load=mask[:, :-1].sum(axis=0),  # per‑expert load (no identity)
            importance=weight[:, :-1].sum(axis=0),
            dropped=(mask[:, :-1].sum(axis=1) == 0).sum(),
            selections=mask[:, :-1].astype(jnp.int32).sum(axis=1),
        )
        return h_next, stats

    # ------------------------------------------------------------------
    # __call__ – public API
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, ids, *, key, inference: bool):
        T = ids.shape[0]
        h = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        cos, sin = rope_cos_sin(
            T, self.embed.weight.shape[1] // self.n_heads, self.rope_base
        )

        for mod in self.backbone:
            key, sub = jax.random.split(key)
            h = h + mod(h, cos, sin, key=sub, inference=inference)

        stats_all = []
        for R in self.routers:
            key, sub = jax.random.split(key)
            h, st = self._hop(h, R, cos, sin, key=sub, inference=inference)
            stats_all.append(st)

        h = jax.vmap(self.ln_out)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        return logits, tuple(stats_all)
