# -----------------------------------------------------------------------------
# model.py
# -----------------------------------------------------------------------------
# End-to-end model wiring:
#   tokens → Embedding → (optional backbone) → repeat n_hops:
#     Router decides (mask, weight) over experts
#     Modules executes non-identity experts with capacity & combines back
#     Residual: h <- h + combine - rho * h   (rho = sum_e kept_weight[t,e])
#   → RMSNorm → tied output projection
#
# Stats are collected per hop (utilization, entropy, identity rate, capacity drops, etc.).
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from dna.nn import Embedding, Dropout, RMSNorm, rope_cos_sin  # primitives
from dna.nn import Attention, FeedForward                     # typical modules
from .modules import Modules
from .router import RouterBase, TopKRouter, RouterOutput


# ---------- stats helper ----------


def _hop_stats(
    *,
    kept_et: jnp.ndarray,          # (E, T) bool
    probs_tr: jnp.ndarray,         # (T, E) float — non-identity expert probs
    mask_tr: jnp.ndarray,          # (T, E) bool  — non-identity hard mask
    rho_t1: jnp.ndarray,           # (T, 1) float — sum_e kept_weight[t,e]
    logits_full: jnp.ndarray,      # (T, E_total) float
    id_in_topk_t: jnp.ndarray,     # (T,) bool
    token_mask_t: jnp.ndarray,     # (T,) bool
    k_routes: int,
) -> Dict[str, Any]:
    # --------- loads & counts ---------
    load_e = kept_et.sum(axis=1).astype(jnp.int32)                      # (E,)
    importance_e = probs_tr.sum(axis=0)                                  # (E,)

    selected_edges = mask_tr.sum()                                       # scalar
    kept_edges = kept_et.sum()                                           # scalar
    cap_drop_frac_edges = jnp.where(
        selected_edges > 0, (selected_edges - kept_edges) / selected_edges, 0.0
    )

    # --------- routing entropy (normalized) ---------
    p = probs_tr
    p_sum = p.sum(axis=1, keepdims=True) + 1e-9
    p_norm = p / p_sum
    ent_t = -(p_norm * jnp.log(p_norm + 1e-9)).sum(axis=1)              # (T,)
    ent_t = ent_t / jnp.log(p.shape[1] + 1e-9)

    # --------- utilization & dispersion ---------
    util = (load_e > 0).mean()
    load_norm = load_e / (jnp.sum(load_e) + 1e-9)
    load_std = jnp.std(load_norm)

    # --------- combine weight summary ---------
    rho_mean = rho_t1.mean()
    rho_min = rho_t1.min()
    rho_max = rho_t1.max()

    # --------- identity expert selection rate ---------
    valid_T = token_mask_t.sum().astype(jnp.int32)
    total_routes = jnp.asarray(k_routes * valid_T, jnp.int32)
    id_topk_count = jnp.sum(jnp.where(token_mask_t, id_in_topk_t, False)).astype(jnp.int32)
    id_rate = jnp.where(valid_T > 0, id_topk_count / valid_T, 0.0)

    return dict(
        load=load_e,
        importance=importance_e,
        rho_mean=rho_mean,
        rho_min=rho_min,
        rho_max=rho_max,
        id_topk_rate=id_rate,
        id_topk_count=id_topk_count,
        total_routes=total_routes,
        entropy_mean=ent_t.mean(),
        entropy_min=ent_t.min(),
        entropy_max=ent_t.max(),
        util=util,
        load_std=load_std,
        cap_drop_frac_edges=cap_drop_frac_edges,
        selected_edges=jnp.asarray(selected_edges, jnp.int32),
        kept_edges=jnp.asarray(kept_edges, jnp.int32),
        id_logit_mean=jnp.mean(logits_full[:, -1]),
        nonid_logit_mean=jnp.mean(logits_full[:, :-1]),
    )


# ---------- end-to-end model ----------


class Model(eqx.Module):
    """Distributed model composed of:
      - Token embedding + dropout
      - Optional dense backbone stack
      - A generic Modules executor (experts collection + capacity)
      - A stack of Routers (n_hops)
      - Output RMSNorm + tied unembedding
    """

    embed: Embedding
    dropout: Dropout
    ln_out: RMSNorm

    # Execution core
    modules: Modules
    routers: Tuple[RouterBase, ...]
    backbone: Tuple[eqx.Module, ...]  # optional dense stack

    # Meta (static)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab: int,
        d_model: int,
        n_heads: int,
        rope_base: float,
        modules: Modules,
        routers: Sequence[RouterBase],
        backbone: Optional[Sequence[eqx.Module]] = None,
        dropout_p: float = 0.0,
        key,
    ):
        self.n_heads = n_heads
        self.rope_base = rope_base

        k_embed, k_drop = jax.random.split(key, 2)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout_p)
        self.ln_out = RMSNorm(d_model)

        self.modules = modules
        self.routers = tuple(routers)
        self.backbone = tuple(backbone) if backbone is not None else tuple()

    # --------- one hop (router → modules → residual) ---------
    def _hop(
        self,
        h_td: jnp.ndarray,                 # (T, d)
        router: RouterBase,
        cos_tdh: jnp.ndarray,              # (T, d_head)
        sin_tdh: jnp.ndarray,              # (T, d_head)
        token_mask_t: jnp.ndarray,         # (T,) bool
        *,
        key,
        inference: bool,
        bias_row: Optional[jnp.ndarray],
        sample_gumbel: bool,
        gumbel_tau: float,
        temp: float,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        # --------- router selection (over *all* experts, incl. identity) ---------
        rkey, mkey = jax.random.split(key)
        r_out: RouterOutput = router(
            h_td,
            key=rkey,
            temp=temp,
            sample_gumbel=sample_gumbel,
            gumbel_tau=gumbel_tau,
            bias_row=bias_row,
        )
        mask_full_te = r_out.mask            # (T, E_total)
        probs_full_te = r_out.weight         # (T, E_total)  (soft weights)
        logits_full_te = r_out.logits        # (T, E_total)

        # Split identity (last col) from non-identity experts
        mask_tr = jnp.where(token_mask_t[:, None], mask_full_te[:, :-1], False)  # (T, E)
        probs_tr = jnp.where(token_mask_t[:, None], probs_full_te[:, :-1], 0.0)  # (T, E)
        id_in_topk_t = jnp.where(token_mask_t, mask_full_te[:, -1], False)       # (T,)

        # --------- modules execution (non-identity experts only) ---------
        # Forward any token-major extras generically: hidden state, rope caches, masks, etc.
        # These will be packed to (E,C,...) inside Modules via slot.
        combined_td, aux = self.modules(
            h_td,                      # typically passed as first arg to experts
            cos=cos_tdh,
            sin=sin_tdh,
            attention_mask=token_mask_t,    # will be filtered per module signature if unsupported
            route_mask=mask_tr,
            route_weight=probs_tr,
            key=mkey,
            inference=inference,
        )
        kept_et = aux["kept"]         # (E, T) kept by capacity

        # --------- residual update: h <- h + combine - rho*h ---------
        kept_tE = kept_et.T                                         # (T, E)
        kept_weight_tE = jnp.where(kept_tE, probs_tr, 0.0)          # (T, E)
        rho_t1 = kept_weight_tE.sum(axis=1, keepdims=True)          # (T, 1)
        h_next = h_td + combined_td - rho_t1 * h_td                 # (T, d)
        h_next = jnp.where(token_mask_t[:, None], h_next, h_td)     # keep pads unchanged

        # --------- stats ---------
        stats = _hop_stats(
            kept_et=kept_et,
            probs_tr=probs_tr,
            mask_tr=mask_tr,
            rho_t1=rho_t1,
            logits_full=logits_full_te,
            id_in_topk_t=id_in_topk_t,
            token_mask_t=token_mask_t,
            k_routes=getattr(router, "k", 0),
        )
        return h_next, stats

    # --------- forward ---------
    @eqx.filter_jit
    def __call__(
        self,
        ids_t: jnp.ndarray,                         # (T,) int32 — single sequence; caller can vmap over batch
        *,
        key,
        inference: bool,
        attention_mask_t: Optional[jnp.ndarray] = None,  # (T,) 1/0
        biases: Optional[jnp.ndarray] = None,             # None | (n_hops,) | (n_hops, E_total)
        gumbel: bool = False,
        gumbel_tau: float = 1.0,
        temp: float = 1.0,
    ) -> Tuple[jnp.ndarray, Tuple[Dict[str, Any], ...]]:
        # --------- token mask ---------
        T = ids_t.shape[0]
        token_mask_t = jnp.ones((T,), dtype=bool) if attention_mask_t is None else attention_mask_t.astype(bool)

        # --------- embed + dropout ---------
        h_td = jax.vmap(self.embed)(ids_t)              # (T, d)
        key, sub = jax.random.split(key)
        h_td = self.dropout(h_td, key=sub, inference=inference)

        # --------- rope caches ---------
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_tdh, sin_tdh = rope_cos_sin(T, d_h, self.rope_base)   # (T, d_head) each

        # --------- optional backbone ---------
        for i, block in enumerate(self.backbone):
            key, sub = jax.random.split(key)
            out_td = block(h_td, cos_tdh, sin_tdh, key=sub, inference=inference)  # generic call
            # masked residual
            out_td = jnp.where(token_mask_t[:, None], out_td - h_td, 0.0) + h_td
            h_td = out_td

        # --------- router+modules hops ---------
        stats_all: list[Dict[str, Any]] = []
        for hop_idx, router in enumerate(self.routers):
            key, sub = jax.random.split(key)

            # Per-hop bias row handling:
            if biases is None:
                bias_row = None
            elif biases.ndim == 1:
                E_total = probs_cols = (getattr(router, "proj").out_features if hasattr(router, "proj") else None)
                # If we can't infer E_total from router, build a row that biases only identity at the end.
                # Here we create a row of zeros with only last col set; shape will be checked inside router.
                if probs_cols is None:
                    bias_row = None
                else:
                    zero = jnp.zeros((probs_cols,), dtype=jnp.float32)
                    bias_row = zero.at[-1].set(biases[hop_idx])
            else:
                bias_row = biases[hop_idx]  # (E_total,)

            h_td, st = self._hop(
                h_td,
                router,
                cos_tdh,
                sin_tdh,
                token_mask_t,
                key=sub,
                inference=inference,
                bias_row=bias_row,
                sample_gumbel=(gumbel and (not inference)),
                gumbel_tau=gumbel_tau,
                temp=temp,
            )
            stats_all.append(st)

        # --------- output head ---------
        h_td = jax.vmap(self.ln_out)(h_td)                              # (T, d)
        logits_tv = jax.vmap(lambda t: t @ self.embed.weight.T)(h_td)   # (T, V)

        return logits_tv, tuple(stats_all)


# ---------- convenience builder (mirrors your previous default pattern) ----------


def build_default_model(
    *,
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
) -> Model:
    """Factory that replicates your alternating Attention/FFN recipe, but separated."""
    k_mods, k_routers, k_backbone, k_embed = jax.random.split(key, 4)

    # Build expert set (alternating Attention / FeedForward)
    mod_keys = jax.random.split(k_mods, n_modules)
    experts = tuple(
        Attention(d_model, n_heads, dropout, key=mod_keys[i]) if (i % 2 == 0)
        else FeedForward(d_model, mlp_mult, dropout, key=mod_keys[i])
        for i in range(n_modules)
    )
    modules = Modules(experts, capacity=capacity)

    # Routers (n_modules + identity)
    total_exp = n_modules + 1
    routers = tuple(
        TopKRouter(d_model, total_exp, topk, key=k) for k in jax.random.split(k_routers, n_hops)
    )

    # Optional backbone
    if num_backbone > 0:
        bb_keys = jax.random.split(k_backbone, num_backbone)
        backbone = tuple(
            Attention(d_model, n_heads, dropout, key=bb_keys[i]) if (i % 2 == 0)
            else FeedForward(d_model, mlp_mult, dropout, key=bb_keys[i])
            for i in range(num_backbone)
        )
    else:
        backbone = tuple()

    # Model
    k_model = jax.random.PRNGKey(0)  # only for embedding/dropout inside Model init
    return Model(
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        rope_base=rope_base,
        modules=modules,
        routers=routers,
        backbone=backbone,
        dropout_p=dropout,
        key=k_model,
    )
