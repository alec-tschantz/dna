# dna_model.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List

import math
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from jaxtyping import Array, Float, Int, Bool


# ---------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------
bf16 = jnp.bfloat16  # parameters & activations
f32 = jnp.float32  # accumulations / logits / softmax


# ---------------------------------------------------------------------
#  utils
# ---------------------------------------------------------------------
def rope_cos_sin(
    T: int, dim: int, base: float = 10_000.0
) -> Tuple[Float[Array, "T dim"], Float[Array, "T dim"]]:
    assert dim % 2 == 0
    pos = jnp.arange(T, dtype=f32)[:, None]
    idx = jnp.arange(0, dim, 2, dtype=f32)[None]
    inv = base ** (-idx / dim)
    ang = pos * inv
    cos = jnp.repeat(jnp.cos(ang), 2, axis=1).astype(bf16)
    sin = jnp.repeat(jnp.sin(ang), 2, axis=1).astype(bf16)
    return cos, sin


def _rotate_half(x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


# ---------------------------------------------------------------------
# core layers
# ---------------------------------------------------------------------
class Linear(eqx.Module):
    weight: Float[Array, "in out"]

    def __init__(self, in_dim: int, out_dim: int, *, key):
        w = jax.random.truncated_normal(
            key, lower=-2.0, upper=2.0, shape=(in_dim, out_dim)
        )
        self.weight = (w * 0.02).astype(bf16)

    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
        return jnp.matmul(x.astype(bf16), self.weight.astype(bf16))


class Embedding(eqx.Module):
    weight: Float[Array, "V D"]

    def __init__(self, vocab: int, dim: int, *, key):
        w = jax.random.truncated_normal(key, lower=-2.0, upper=2.0, shape=(vocab, dim))
        self.weight = (w * 0.02).astype(bf16)

    def __call__(self, ids: Int[Array, "..."]) -> Float[Array, "... D"]:
        return jnp.take(self.weight.astype(bf16), ids, axis=0)


class RMSNorm(eqx.Module):
    weight: Float[Array, "D"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones((dim,), dtype=bf16)
        self.eps = float(eps)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x32 = x.astype(f32)
        var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
        y = x32 * jax.lax.rsqrt(var + self.eps)
        return (y.astype(bf16) * self.weight.astype(bf16)).astype(bf16)


class Dropout(eqx.Module):
    rate: float = eqx.field(static=True)

    def __init__(self, rate: float = 0.0):
        self.rate = float(rate)

    def __call__(
        self, x: Float[Array, "..."], *, key, inference: bool
    ) -> Float[Array, "..."]:
        if inference or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        m = jax.random.bernoulli(key, keep, x.shape)
        return jnp.where(m, x / keep, jnp.zeros_like(x))


# ---------------------------------------------------------------------
# experts: (x, mask, *, key, inference)
# ---------------------------------------------------------------------
class Expert(eqx.Module):
    def __call__(
        self,
        x: Float[Array, "T D"],
        mask: Optional[Bool[Array, "T"]],
        pos: Optional[tuple[Float[Array, "T d_h"], Float[Array, "T d_h"]]],
        *,
        key,
        inference: bool,
    ) -> Float[Array, "T D"]:
        raise NotImplementedError


class Identity(Expert):
    def __call__(self, x, mask, pos, *, key, inference):
        return x


class FeedForward(Expert):
    ln: RMSNorm
    up: Linear
    gate: Linear
    down: Linear
    drop: Dropout

    def __init__(self, d_model: int, mult: int, dropout: float, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.ln = RMSNorm(d_model)
        self.up = Linear(d_model, d_model * mult, key=k1)
        self.gate = Linear(d_model, d_model * mult, key=k2)
        self.down = Linear(d_model * mult, d_model, key=k3)
        self.drop = Dropout(dropout)

    def __call__(self, x, mask, pos, *, key, inference):
        k1, k2 = jax.random.split(key)
        h = self.ln(x)
        a = self.gate(h).astype(f32)
        u = (jnn.silu(a).astype(bf16) * self.up(h)).astype(bf16)
        u = self.drop(u, key=k1, inference=inference)
        u = self.down(u)
        u = self.drop(u, key=k2, inference=inference)
        y = x + u
        return jnp.where(mask[:, None], y, x) if mask is not None else y


class Attention(Expert):
    ln: RMSNorm
    qkv: Linear
    out: Linear
    drop: Dropout
    n_h: int = eqx.field(static=True)
    d_h: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, dropout: float, *, key):
        assert d_model % n_heads == 0
        d_h = d_model // n_heads
        assert d_h % 2 == 0
        self.n_h, self.d_h = n_heads, d_h
        k1, k2 = jax.random.split(key, 2)
        self.ln = RMSNorm(d_model)
        self.qkv = Linear(d_model, 3 * d_model, key=k1)
        self.out = Linear(d_model, d_model, key=k2)
        self.drop = Dropout(dropout)

    def __call__(self, x, mask, pos, *, key, inference):
        cos, sin = pos
        T, D = x.shape
        k1, k2 = jax.random.split(key)

        h = self.ln(x)
        qkv = self.qkv(h).astype(bf16).reshape(T, 3, self.n_h, self.d_h)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [T,H,dh]

        q = q * cos[:, None, :] + _rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + _rotate_half(k) * sin[:, None, :]

        qf = q.astype(f32).transpose(1, 0, 2)  # [H,T,dh]
        kf = k.astype(f32).transpose(1, 0, 2)
        vf = v.astype(f32).transpose(1, 0, 2)

        scores = jnp.einsum("htd,hsd->hts", qf, kf) / math.sqrt(self.d_h)
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, :, :]

        if mask is not None:
            m = mask.astype(bool)
            qmask = m[None, :, None]
            kmask = m[None, None, :]
            allow = causal & kmask
            neg = jnp.finfo(f32).min
            scores = jnp.where(allow, scores, neg)
            scores = jnp.where(qmask, scores, neg)
        else:
            scores = jnp.where(causal, scores, jnp.finfo(f32).min)

        p = jnn.softmax(scores, axis=-1).astype(bf16)
        p = self.drop(p, key=k1, inference=inference).astype(f32)

        o = (
            jnp.einsum("hts,hsd->htd", p, vf)
            .transpose(1, 0, 2)
            .reshape(T, D)
            .astype(bf16)
        )
        o = self.out(o)
        o = self.drop(o, key=k2, inference=inference)
        y = x + o
        return jnp.where(mask[:, None], y, x) if mask is not None else y


# ---------------------------------------------------------------------
# router -> (mask [T,E], probs [T,E])
# ---------------------------------------------------------------------
def _topk_mask(logits: Float[Array, "T E"], k: int) -> Bool[Array, "T E"]:
    _, idx = jax.lax.top_k(logits, int(k))
    return (
        jnp.zeros_like(logits, dtype=bool)
        .at[jnp.arange(logits.shape[0])[:, None], idx]
        .set(True)
    )


def _maybe_gumbel(x, *, key, inference: bool, tau):
    if inference:
        return x

    def no_noise(_):
        return x

    def add_noise(_):
        u = jax.random.uniform(key, x.shape, minval=1e-6, maxval=1.0 - 1e-6)
        g = -jnp.log(-jnp.log(u))
        return x + tau * g

    return jax.lax.cond(tau > 0.0, add_noise, no_noise, operand=None)


class Router(eqx.Module):
    ln: RMSNorm
    proj: Linear
    drop: Dropout
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, dropout: float, *, key):
        k1, k2 = jax.random.split(key)
        self.ln = RMSNorm(d_model)
        self.proj = Linear(d_model, n_exp, key=k1)
        self.drop = Dropout(dropout)
        self.k = int(k)

    def __call__(
        self,
        h: Float[Array, "T D"],
        mask: Optional[Bool[Array, "T"]],
        *,
        key,
        inference: bool,
        router_temp: float = 1.0,
        gumbel_tau: float = 0.0,
    ) -> Tuple[Bool[Array, "T E"], Float[Array, "T E"]]:
        k1, k2 = jax.random.split(key)
        x = self.ln(h)
        x = self.drop(x, key=k1, inference=inference)
        logits = self.proj(x).astype(f32)  # [T,E]
        if mask is not None:
            logits = jnp.where(mask[:, None], logits, jnp.finfo(f32).min)

        sel = logits / jnp.clip(router_temp, 1e-6, None)
        sel = _maybe_gumbel(sel, key=k2, inference=inference, tau=gumbel_tau)
        hard = _topk_mask(sel, self.k)

        probs = jnn.softmax(
            logits / jnp.clip(router_temp, 1e-6, None), axis=-1
        )  # [T,E]
        if mask is not None:
            probs = jnp.where(mask[:, None], probs, 0.0)
        probs = jnp.where(hard, probs, 0.0)  # no re-norm
        return hard, probs


# ---------------------------------------------------------------------
# expert grouping
# ---------------------------------------------------------------------
def _sig(mod: eqx.Module) -> Tuple[str, str, str]:
    arrs = eqx.filter(mod, eqx.is_array)
    arr_shapes_dtypes = jax.tree.map(lambda x: (tuple(x.shape), str(x.dtype)), arrs)
    arr_struct = jax.tree.structure(arrs)
    return (
        type(mod).__name__,
        str(arr_struct),
        str(jax.tree.leaves(arr_shapes_dtypes)),
    )


def _stack_params(mods: List[eqx.Module]):
    params = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params)
    return stacked, static


class GroupStack(eqx.Module):
    idx: Int[Array, "E"]
    params: Any
    proto: Any

    def __call__(
        self,
        x: Float[Array, "T D"],
        mask: Bool[Array, "E T"],
        pos: Optional[Tuple[Float[Array, "T dh"], Float[Array, "T dh"]]],
        *,
        key,
        inference: bool,
    ) -> Float[Array, "E T D"]:
        E = int(mask.shape[0])
        keys = jax.random.split(key, E)

        def _one(p, s, k):
            mod: Expert = eqx.combine(p, self.proto)
            return mod(x, s, pos, key=k, inference=inference)  # [T,D]

        return jax.vmap(_one)(self.params, mask, keys)  # [E,T,D]


# ---------------------------------------------------------------------
# DNA model
# ---------------------------------------------------------------------
class DNA(eqx.Module):
    embed: Embedding
    drop: Dropout
    ln_out: RMSNorm
    backbone: Tuple[Expert, ...]
    groups: Tuple[GroupStack, ...]
    routers: Tuple[Router, ...]
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    vocab: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        modules: Tuple[Expert, ...],
        routers: Tuple[Router, ...],
        vocab: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        rope_base: float,
        backbone: Optional[Tuple[Expert, ...]] = None,
        key,
    ):
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.rope_base = float(rope_base)
        k_embed = key

        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.drop = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for idx, m in enumerate(modules):
            sig = _sig(m)
            e = buckets.setdefault(sig, {"idx": [], "mods": []})
            e["idx"].append(idx)
            e["mods"].append(m)

        grouped: List[GroupStack] = []
        for b in buckets.values():
            params, proto = _stack_params(b["mods"])
            grouped.append(
                GroupStack(
                    idx=jnp.array(b["idx"], dtype=jnp.int32), params=params, proto=proto
                )
            )
        grouped.sort(key=lambda g: int(g.idx[0]))
        self.groups = tuple(grouped)

        self.backbone = tuple(backbone) if backbone is not None else tuple()
        self.routers = tuple(routers)

    # ----------------------------- hop (core) -----------------------------
    def _hop(
        self,
        h: Float[Array, "T D"],
        router: Router,
        pos: tuple[Float[Array, "T dh"], Float[Array, "T dh"]],
        *,
        key,
        inference: bool,
        mask: Bool[Array, "T"],
        router_temp: float,
        gumbel_tau: float,
    ) -> Float[Array, "T D"]:

        kr, ke = jax.random.split(key)

        hard_te, probs_te = router(
            h,
            mask,
            key=kr,
            inference=inference,
            router_temp=router_temp,
            gumbel_tau=gumbel_tau,
        )

        T, D = h.shape
        delta = jnp.zeros((T, D), dtype=f32)

        for gi, grp in enumerate(self.groups):
            subkey = jax.random.fold_in(ke, gi)

            cols = grp.idx  # [e_g]
            hard_g = jnp.take(hard_te, cols, axis=1)  # [t,e_g]
            prob_g = jnp.take(probs_te, cols, axis=1)  # [t,e_g]

            # per-expert token mask (valid & selected)
            sel_et = (hard_g & mask[:, None]).T  # [e_g,t]

            # run experts in this group only on masked tokens; returns [e_g,t,d]
            out_etd = grp(h, sel_et, pos, key=subkey, inference=inference)
            out_etd = out_etd.astype(f32)

            # weights per (expert, token); zero for non-selected
            w_et1 = jnp.where(hard_g, prob_g, 0.0).T[..., None]  # [e_g,t,1]

            # Eq. (3): accumulate sum_e ρ_e * (M_e(h) - h)
            delta = delta + jnp.sum(
                w_et1 * (out_etd - h[None, ...].astype(f32)), axis=0
            )

        y = (h.astype(f32) + delta).astype(bf16)
        return jnp.where(mask[:, None], y, h)

    # ----------------------------- forward (T) -----------------------------
    def __call__(
        self,
        ids: Int[Array, "T"],
        *,
        key,
        inference: bool,
        mask: Optional[Bool[Array, "T"]] = None,
        router_temp: float = 1.0,
        gumbel_tau: float = 0.0,
    ) -> Float[Array, "T V"]:
        T = int(ids.shape[0])
        valid = mask if mask is not None else jnp.ones((T,), dtype=bool)

        h = jax.vmap(self.embed)(ids).astype(bf16)
        k0, key = jax.random.split(key)
        h = self.drop(h, key=k0, inference=inference)

        dh = self.d_model // self.n_heads
        pos = rope_cos_sin(T, dh, self.rope_base)

        for m in self.backbone:
            key, sub = jax.random.split(key)
            h = m(h, valid, key=sub, inference=inference, pos=pos)

        for r in self.routers:
            key, sub = jax.random.split(key)
            h = self._hop(
                h,
                r,
                pos,
                key=sub,
                inference=inference,
                mask=valid,
                router_temp=router_temp,
                gumbel_tau=gumbel_tau,
            )

        h = jax.vmap(self.ln_out)(h)
        logits = jnp.matmul(h.astype(f32), self.embed.weight.astype(f32).T)
        return logits  # [T,V] f32
