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


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Return boolean mask of top-k indices per token: [T, E]."""
    _, idx = jax.lax.top_k(logits, k)
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)
    return hard.astype(bool)


def _capacity_select(mask_te: jnp.ndarray, gate_te: jnp.ndarray, capacity: int):
    """
    Select up to `capacity` tokens per expert using gate as score.
    Inputs:
      mask_te: [T, E] bool – token chose expert (top-k pre-capacity), excludes identity
      gate_te: [T, E] float – softmax probs for experts, excludes identity
    Returns:
      slot: [E, C, T] binary selection matrix
      kept: [E, T] bool  – whether expert e kept token t
    """
    m = mask_te.T
    g = jnp.where(m, gate_te.T, -jnp.inf)  # [E, T]

    cap = jnp.minimum(capacity, g.shape[1])
    _, top_idx = jax.lax.top_k(g, cap)  # [E, C]

    E, C = top_idx.shape
    T = g.shape[1]
    slot = jnp.zeros((E, C, T), dtype=g.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], top_idx].set(1.0)

    kept = slot.sum(1).astype(bool)  # [E, T]
    return slot, kept


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

    def __call__(self, h: jnp.ndarray, is_valid: jnp.ndarray):
        """
        h: [T, d], is_valid: [T] bool (True=real token, False=pad)
        Returns: (mask [T,E], probs [T,E]) over E = n_modules + 1 (last is identity)
        """
        logits = jax.vmap(self.proj)(h)  # [T, E]
        T, E = logits.shape

        # Encourage identity: additive bias on identity column (JIT-safe)
        id_col_mask = (jnp.arange(E) == (E - 1))[None, :]  # [1,E] True at identity
        logits = logits + jnp.where(id_col_mask, self.id_bias, 0.0)

        # Make pads route to identity only, w/out boolean indexing
        pad = (~is_valid)[:, None]        # [T,1]
        non_id_col_mask = ~id_col_mask    # [1,E]

        # Non-identity cols for pads => -inf
        logits = jnp.where(pad & non_id_col_mask, jnp.full_like(logits, -1e30), logits)
        # Identity col for pads => +big
        logits = logits + jnp.where(pad & id_col_mask, 1e30, 0.0)

        probs = jnn.softmax(logits, axis=-1)  # differentiable; no ST

        # Hard top-k on logits
        mask = _topk_mask(logits, self.k)     # [T,E] bool

        # For pads force the mask to identity-only (avoid random -inf picks when k>1)
        id_rows = id_col_mask.astype(bool).repeat(T, axis=0)  # [T,E]
        mask = jnp.where(pad, id_rows, mask)

        return mask, probs


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
    n_modules: int = eqx.field(static=True)   # trainable experts
    capacity: int = eqx.field(static=True)    # per-expert capacity
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
        total_experts = n_modules + 1  # +1 identity (last column)
        self.routers = tuple(
            Router(d_model, total_experts, topk, id_bias=identity_bias, key=k)
            for k in router_keys
        )

    # ---------------------------------------------------------------------
    # Forward pass (identity excluded from capacity; JIT-safe masking)
    # ---------------------------------------------------------------------

    def _hop(self, h, is_valid, router: Router, cos, sin, *, key, inference):
        mask_full, probs_full = router(h, is_valid)   # [T,E], [T,E]
        probs_tr = probs_full[:, :-1]                 # exclude identity
        mask_tr = mask_full[:, :-1]                   # exclude identity

        # Capacity selection per expert (over tokens)
        slot, kept = _capacity_select(mask_tr, probs_tr, self.capacity)  # [E,C,T], [E,T]

        # Gather token blocks for experts
        xin = jnp.einsum("ect,td->ecd", slot, h)     # [E, C, d]
        cos_r = jnp.einsum("ect,td->ecd", slot, cos) # [E, C, d_h]
        sin_r = jnp.einsum("ect,td->ecd", slot, sin) # [E, C, d_h]

        # Apply trainable experts grouped by signature
        outs = []
        for g in self.groups:
            n_g = len(g["idx"])
            sub_keys = jax.random.split(key, n_g)
            inp = xin[g["idx"]]    # [n_g, C, d]
            c = cos_r[g["idx"]]    # [n_g, C, d_h]
            s = sin_r[g["idx"]]    # [n_g, C, d_h]
            out_g = jax.vmap(
                lambda p, x, c1, s1, k1: eqx.combine(p, g["static"])(
                    x, c1, s1, key=k1, inference=inference
                )
            )(g["params"], inp, c, s, sub_keys)
            outs.append(out_g)
        expert_out = jnp.concatenate(outs, axis=0)  # [E, C, d], E=n_modules

        # Realized (post-capacity) per-token expert weights: no renorm
        kept_t = kept.T                              # [T,E]
        combine_w = jnp.where(kept_t, probs_tr, 0.0) # [T,E] unnormalized
        rho = combine_w.sum(axis=1, keepdims=True)   # [T,1] realized non-identity mass

        # Scatter back to tokens with combine weights
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)  # [T,d]
        h_next = h + combine - rho * h  # preserve identity mass

        # Stats
        per_expert_load = kept.sum(axis=1)  # [E]
        selected_any = (mask_tr.sum(axis=1) > 0)        # [T]
        kept_any = (kept_t.sum(axis=1) > 0)             # [T]
        cap_drop_rate = jnp.where(selected_any.any(), (selected_any & (~kept_any)).mean(), 0.0)

        stats = dict(
            load=per_expert_load,                # post-capacity loads (no identity)
            importance=probs_tr.sum(axis=0),     # pre-capacity mass per expert
            cap_drop_rate=cap_drop_rate,
            selections=mask_tr.astype(jnp.int32).sum(axis=1),
        )
        return h_next, stats

    # ------------------------------------------------------------------
    # __call__ – public API
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, ids, attn_mask: jnp.ndarray = None, *, key, inference: bool):
        T = ids.shape[0]
        h = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        if attn_mask is None:
            is_valid = jnp.ones((T,), dtype=bool)
        else:
            is_valid = attn_mask.astype(bool)

        cos, sin = rope_cos_sin(
            T, self.embed.weight.shape[1] // self.n_heads, self.rope_base
        )

        for mod in self.backbone:
            key, sub = jax.random.split(key)
            h = h + mod(h, cos, sin, key=sub, inference=inference)

        stats_all = []
        for R in self.routers:
            key, sub = jax.random.split(key)
            h, st = self._hop(h, is_valid, R, cos, sin, key=sub, inference=inference)
            stats_all.append(st)

        h = jax.vmap(self.ln_out)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        return logits, tuple(stats_all)
