# dna/dna.py
#
# DNA hop faithful to the paper and MoE best practices:
# - Router logits + optional identity-only bias (from trainer, outside grad).
# - Softmax temperature for gates (probs) and optional Gumbel-Top-k sampling
#   for the discrete selection (top-k on logits + noise).
# - No STE. Gradients flow through the softmax probs of selected experts.
# - Capacity packing for JIT-friendly per-expert execution; no post-drop renorm.
# - Eq. (3)-style residual, consistent with modules that return x + f(x).
#
# JIT-safe: no boolean advanced indexing; use jnp.where and lax.top_k.

from typing import Tuple, Dict, Any, Optional

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
    rope_cos_sin,
)

# -------------------------- routing helpers -------------------------- #

def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Boolean mask of top-k indices per row of `logits` (shape [T, E])."""
    _, idx = jax.lax.top_k(logits, k)
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)
    return hard.astype(bool)


def _capacity_select(mask_te: jnp.ndarray, gate_te: jnp.ndarray, capacity: int):
    """
    Fixed per-expert capacity across tokens (JIT-friendly packing).
    Inputs:
      mask_te: [T, E] bool   (non-identity top-k per token)
      gate_te: [T, E] float  (softmax probs, non-identity slice)
    Returns:
      slot: [E, C, T] binary selection matrix (slots per expert)
      kept: [E, T] bool (whether expert e kept token t)
    """
    m = mask_te.T                         # [E, T]
    g = jnp.where(m, gate_te.T, -jnp.inf) # [E, T]

    cap = int(min(capacity, g.shape[1]))
    _, top_idx = jax.lax.top_k(g, cap)    # [E, C]

    E, C = top_idx.shape
    T = g.shape[1]
    slot = jnp.zeros((E, C, T), dtype=g.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], top_idx].set(1.0)

    # Avoid phantom picks by intersecting with original mask
    slot = slot * m[:, None, :]           # [E, C, T]
    kept = slot.sum(1).astype(bool)       # [E, T]
    return slot, kept


def _sig(mod):
    st = jax.tree_util.tree_structure(eqx.filter(mod, eqx.is_array))
    static = str(eqx.filter(mod, lambda x: not eqx.is_array(x)))
    return type(mod).__name__, str(st), static


def _stack(mods):
    params = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *params), static


# ------------------------------ Router ------------------------------ #

class Router(eqx.Module):
    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        self.k = k
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)

    def __call__(
        self,
        h: jnp.ndarray,
        bias_row: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.Array] = None,
        sample_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        temp: float = 1.0,
    ):
        """
        h: [T, d]
        bias_row: [E] or [ ] None. Trainer-supplied per-hop bias. We apply ONLY the
                  identity component (last index) to logits before selection.
                  We also stop_gradient so it's outside autograd.
        key: RNG key used only if sample_gumbel=True.
        sample_gumbel: if True, add Gumbel noise to logits for *top-k decision only*.
        gumbel_tau: Gumbel noise scale (τ).
        temp: router softmax temperature (applies to probs only).

        Returns:
          mask_full: [T, E] bool  (hard top-k, possibly Gumbel-sampled)
          probs:     [T, E] float (softmax over temperature-scaled clean logits)
          logits:    [T, E] float (clean logits after identity bias)
        """
        logits = jax.vmap(self.proj)(h)                 # [T, E]
        E = logits.shape[-1]

        # Identity-only bias (last column). Ignore non-identity entries if provided.
        if bias_row is not None:
            b_id = jax.lax.stop_gradient(bias_row[-1])
            id_mask = (jnp.arange(E) == (E - 1))[None, :]  # [1,E]
            logits = logits + b_id * id_mask

        # Probabilities from temperature-scaled CLEAN logits
        probs = jnn.softmax(logits / jnp.clip(temp, 1e-6, None), axis=-1)

        # Discrete decision: optionally add Gumbel noise to selection path
        if sample_gumbel:
            assert key is not None, "Router requires key when sample_gumbel=True"
            u = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
            gumbel = -jnp.log(-jnp.log(u))
            logits_for_topk = logits + gumbel_tau * gumbel
        else:
            logits_for_topk = logits

        mask_full = _topk_mask(logits_for_topk, self.k)
        return mask_full, probs, logits


# ------------------------------- DNA -------------------------------- #

class DNA(eqx.Module):
    embed: Embedding
    dropout: Dropout
    backbone: Tuple[eqx.Module, ...]
    routers: Tuple[Router, ...]
    groups: Tuple[Dict[str, Any], ...]
    ln_out: RMSNorm

    # Static fields
    n_modules: int = eqx.field(static=True)   # trainable experts
    capacity: int = eqx.field(static=True)    # per-expert capacity (fixed C for JIT)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

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
        key,
    ):
        self.n_modules = n_modules
        self.capacity = capacity
        self.n_heads = n_heads
        self.rope_base = rope_base

        # Global components
        k_embed, k_modules, k_routers, k_backbone = jax.random.split(key, 4)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        # Build trainable experts (alternate Attn/FFN)
        module_keys = jax.random.split(k_modules, n_modules)
        experts = []
        for i, k_i in enumerate(module_keys):
            if i % 2 == 0:
                experts.append(Attention(d_model, n_heads, dropout, key=k_i))
            else:
                experts.append(FeedForward(d_model, mlp_mult, dropout, key=k_i))

        # Group identical signatures
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

        # Optional dense backbone
        if num_backbone > 0:
            bb_keys = jax.random.split(k_backbone, num_backbone)
            bb = []
            for i in range(num_backbone):
                if i % 2 == 0:
                    bb.append(Attention(d_model, n_heads, dropout, key=bb_keys[i]))
                else:
                    bb.append(FeedForward(d_model, mlp_mult, dropout, key=bb_keys[i]))
            self.backbone = tuple(bb)
        else:
            self.backbone = tuple()

        # Routers (one per hop). Last expert index is identity.
        router_keys = jax.random.split(k_routers, n_hops)
        total_experts = n_modules + 1
        self.routers = tuple(
            Router(d_model, total_experts, topk, key=k) for k in router_keys
        )

    # --------------------- one hop --------------------- #

    def _hop(
        self,
        h,
        router: Router,
        cos,
        sin,
        *,
        key,
        inference,
        bias_row,
        sample_gumbel: bool,
        gumbel_tau: float,
        temp: float,
    ):
        """One hop with optional Gumbel-Top-k for the discrete selection."""
        k_route, k_exec = jax.random.split(key)

        # Route (include identity in routing; identity is last column)
        mask_full, probs_full, logits_full = router(
            h,
            bias_row,
            key=k_route,
            sample_gumbel=sample_gumbel,
            gumbel_tau=gumbel_tau,
            temp=temp,
        )  # [T,E], [T,E], [T,E]

        probs_tr = probs_full[:, :-1]                   # exclude identity
        mask_tr = mask_full[:, :-1]                     # exclude identity
        id_in_topk = mask_full[:, -1]                   # [T]

        # Token budget per expert across tokens
        slot, kept = _capacity_select(mask_tr, probs_tr, self.capacity)  # [E,C,T],[E,T]

        # Gather token blocks for experts
        xin = jnp.einsum("ect,td->ecd", slot, h)         # [E,C,d]
        cos_r = jnp.einsum("ect,td->ecd", slot, cos)     # [E,C,d_h]
        sin_r = jnp.einsum("ect,td->ecd", slot, sin)     # [E,C,d_h]

        # Execute trainable experts (modules return x+f(x))
        outs = []
        for g in self.groups:
            n_g = len(g["idx"])
            sub_keys = jax.random.split(k_exec, n_g)
            k_exec = sub_keys[0]  # advance deterministically
            inp = xin[g["idx"]]    # [n_g, C, d]
            c = cos_r[g["idx"]]    # [n_g, C, d_h]
            s = sin_r[g["idx"]]    # [n_g, C, d_h]
            out_g = jax.vmap(
                lambda p, x, c1, s1, k1: eqx.combine(p, g["static"])(
                    x, c1, s1, key=k1, inference=inference
                )
            )(g["params"], inp, c, s, sub_keys)
            outs.append(out_g)
        expert_out = jnp.concatenate(outs, axis=0)       # [E, C, d] = M_e(h_in)

        # Combine (no renorm), identity not dispatched
        kept_t = kept.T                                   # [T,E]
        combine_w = jnp.where(kept_t, probs_tr, 0.0)      # [T,E] realized weights
        rho = combine_w.sum(axis=1, keepdims=True)        # [T,1] realized non-id mass
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)

        # Eq. (3): h_next = h + Σ_e ρ_e (M_e(h) - h). Since M_e returns x+f(x),
        # this equals h + Σ_e ρ_e f_e(h).
        h_next = h + combine - rho * h

        # ---------------- stats ----------------
        load = kept.sum(axis=1).astype(jnp.int32)         # [E]
        importance = probs_tr.sum(axis=0)                 # [E]
        selected_edges = mask_tr.sum()
        kept_edges = kept.sum()
        cap_drop_frac_edges = jnp.where(
            selected_edges > 0, (selected_edges - kept_edges) / selected_edges, 0.0
        )

        p_tr = probs_tr
        p_tr_sum = p_tr.sum(axis=1, keepdims=True) + 1e-9
        p_tr_norm = p_tr / p_tr_sum
        ent = -(p_tr_norm * jnp.log(p_tr_norm + 1e-9)).sum(axis=1)
        ent = ent / jnp.log(p_tr.shape[1] + 1e-9)

        util = (load > 0).mean()
        load_norm = load / (jnp.sum(load) + 1e-9)
        load_std = jnp.std(load_norm)

        rho_mean = rho.mean(); rho_min = rho.min(); rho_max = rho.max()

        k = router.k
        T = h.shape[0]
        total_routes = jnp.asarray(k * T, jnp.int32)
        id_topk_count = id_in_topk.sum().astype(jnp.int32)

        stats = dict(
            load=load,
            importance=importance,
            rho_mean=rho_mean, rho_min=rho_min, rho_max=rho_max,
            id_topk_rate=id_in_topk.mean(),
            id_topk_count=id_topk_count,
            total_routes=total_routes,
            entropy_mean=ent.mean(),
            entropy_min=ent.min(),
            entropy_max=ent.max(),
            util=util,
            load_std=load_std,
            cap_drop_frac_edges=cap_drop_frac_edges,
            selected_edges=jnp.asarray(selected_edges, jnp.int32),
            kept_edges=jnp.asarray(kept_edges, jnp.int32),
            id_logit_mean=jnp.mean(logits_full[:, -1]),
            nonid_logit_mean=jnp.mean(logits_full[:, :-1]),
        )
        return h_next, stats

    # --------------------- public API --------------------- #

    @eqx.filter_jit
    def __call__(
        self,
        ids: jnp.ndarray,             # [T]
        *,
        key,
        inference: bool,
        biases: Optional[jnp.ndarray] = None,      # [n_hops, E_total] or [n_hops]
        gumbel: bool = False,
        gumbel_tau: float = 1.0,
        temp: float = 1.0,
    ):
        """
        Forward pass (no batching; vmap externally).
        `biases`: if 2D, we read the last column only (identity); if 1D, use scalar per hop.
        """
        T = ids.shape[0]
        h = jax.vmap(self.embed)(ids)             # [T, d]
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        d_h = self.embed.weight.shape[1] // self.n_heads
        cos, sin = rope_cos_sin(T, d_h, self.rope_base)  # [T, d_h]

        for mod in self.backbone:
            key, sub = jax.random.split(key)
            h = h + mod(h, cos, sin, key=sub, inference=inference)

        stats_all = []
        for hop_idx, R in enumerate(self.routers):
            key, sub = jax.random.split(key)
            bias_row = None
            if biases is not None:
                if biases.ndim == 1:
                    # scalar per hop -> apply to identity column
                    E_total = self.n_modules + 1
                    zero = jnp.zeros((E_total,), dtype=jnp.float32)
                    bias_row = zero.at[-1].set(biases[hop_idx])
                else:
                    bias_row = biases[hop_idx]  # [E_total]; router will mask to identity
            h, st = self._hop(
                h, R, cos, sin,
                key=sub,
                inference=inference,
                bias_row=bias_row,
                sample_gumbel=gumbel and (not inference),
                gumbel_tau=gumbel_tau,
                temp=temp,
            )
            stats_all.append(st)

        h = jax.vmap(self.ln_out)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        return logits, tuple(stats_all)
