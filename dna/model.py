from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from dna.nn import Embedding, Dropout, RMSNorm, rope_cos_sin  # primitives
from dna.nn import Attention, FeedForward                     # typical modules
from .modules import ModuleGroup
from .router import TopKRouter, RouterOutput


# ---- hop stats -------------------------------------------------------------

def _hop_stats(*, kept_et: jnp.ndarray, probs_tr: jnp.ndarray, mask_tr: jnp.ndarray,
               rho_t1: jnp.ndarray, logits_full: jnp.ndarray,
               token_mask_t: jnp.ndarray, id_col: Optional[int], k_routes: int) -> Dict[str, Any]:
    """Small but telling set of per‑hop diagnostics.
    Shapes:
        kept_et:  (E, T) bool for non‑identity experts
        probs_tr: (T, E) float for non‑identity experts
        mask_tr:  (T, E) bool  for non‑identity experts
        rho_t1:   (T, 1)  sum of *kept* weights per token
        logits_full: (T, E_total) float (clean)
        token_mask_t: (T,) bool indicating valid tokens (pads=False)
        id_col:   index of identity expert if present, else None
    """
    load_e = kept_et.sum(axis=1).astype(jnp.int32)                # (E,)
    selected_edges = mask_tr.sum()                                 # scalar
    kept_edges = kept_et.sum()                                     # scalar
    cap_drop_frac = jnp.where(selected_edges > 0, (selected_edges - kept_edges) / selected_edges, 0.0)

    # Entropy over the *non‑identity* distribution per token, normalized by log E
    p = probs_tr
    p_sum = p.sum(axis=1, keepdims=True) + 1e-9
    p_norm = p / p_sum
    ent_t = -(p_norm * jnp.log(p_norm + 1e-9)).sum(axis=1)
    ent_t = ent_t / jnp.log(p.shape[1] + 1e-9)

    util_frac = (load_e > 0).mean()                               # fraction of used experts
    load_cv = jnp.std(load_e.astype(jnp.float32) + 1e-9) / (jnp.mean(load_e.astype(jnp.float32)) + 1e-9)

    rho_mean, rho_min, rho_max = rho_t1.mean(), rho_t1.min(), rho_t1.max()
    avg_selected_k = mask_tr.sum(axis=1).mean()                   # should be close to k_routes

    stats = dict(
        tokens_per_expert=load_e,
        kept_per_expert=load_e,             # synonymous here; explicit name for clarity
        util_frac=util_frac,
        load_cv=load_cv,
        entropy_mean=ent_t.mean(),
        entropy_min=ent_t.min(),
        entropy_max=ent_t.max(),
        rho_mean=rho_mean,
        rho_min=rho_min,
        rho_max=rho_max,
        avg_selected_k=avg_selected_k,
        capacity_drop_frac=cap_drop_frac,
        selected_edges=jnp.asarray(selected_edges, jnp.int32),
        kept_edges=jnp.asarray(kept_edges, jnp.int32),
        logits_mean=jnp.mean(logits_full),
    )

    if id_col is not None:
        id_rate = jnp.where(token_mask_t.any(), logits_full.shape[0] > 0, False)
        # Actual identity selection rate (using the hard mask from the caller)
        # The caller passes mask_full to compute this quickly.
        # We store a placeholder here; it will be overwritten by the caller for accuracy.
        stats["identity_selected_rate"] = jnp.array(0.0)

    return stats


# ---- model -----------------------------------------------------------------

class Model(eqx.Module):
    """Distributed model composed of:
      • Token embedding + dropout
      • Optional dense backbone stack
      • A `ModuleGroup` executor (experts collection + capacity)
      • A stack of Routers (n_hops)
      • Output RMSNorm + tied unembedding
    """

    embed: Embedding
    dropout: Dropout
    ln_out: RMSNorm

    group: ModuleGroup
    routers: Tuple[TopKRouter, ...]
    backbone: Tuple[eqx.Module, ...]  # optional dense stack

    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab: int,
        d_model: int,
        n_heads: int,
        rope_base: float,
        group: ModuleGroup,
        routers: Sequence[TopKRouter],
        backbone: Optional[Sequence[eqx.Module]] = None,
        dropout_p: float = 0.0,
        key,
    ):
        self.n_heads = int(n_heads)
        self.rope_base = float(rope_base)

        k_embed, k_drop = jax.random.split(key, 2)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout_p)
        self.ln_out = RMSNorm(d_model)

        self.group = group
        self.routers = tuple(routers)
        self.backbone = tuple(backbone) if backbone is not None else tuple()

    # -- one hop: modules + residual (router already executed outside) -------
    def _apply_hop(
        self,
        h_td: jnp.ndarray,                 # (T, d)
        r_out: RouterOutput,               # routing result over ALL experts
        cos_tdh: jnp.ndarray,              # (T, d_head)
        sin_tdh: jnp.ndarray,              # (T, d_head)
        token_mask_t: jnp.ndarray,         # (T,) bool
        *,
        key,
        inference: bool,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        # Split identity (last column) from non‑identity experts
        mask_full_te = r_out.mask
        probs_full_te = r_out.weight
        logits_full_te = r_out.logits

        E_total = probs_full_te.shape[-1]
        assert E_total >= 1, "Expect at least one expert; conventionally last is identity"
        id_col = E_total - 1

        mask_tr = jnp.where(token_mask_t[:, None], mask_full_te[:, :id_col], False)  # (T, E_nonid)
        probs_tr = jnp.where(token_mask_t[:, None], probs_full_te[:, :id_col], 0.0)  # (T, E_nonid)
        id_selected_t = jnp.where(token_mask_t, mask_full_te[:, id_col], False)      # (T,)

        # Execute non‑identity experts via ModuleGroup
        combined_td, aux = self.group(
            h_td,
            cos=cos_tdh,
            sin=sin_tdh,
            attention_mask=token_mask_t,
            route_mask=mask_tr,
            route_weight=probs_tr,
            key=key,
            inference=inference,
        )
        kept_et = aux["kept"]                                # (E_nonid, T)

        # Residual update: h <- h + combined − rho*h, where rho is *kept* weight mass
        kept_tE = kept_et.T                                   # (T, E_nonid)
        kept_weight_tE = jnp.where(kept_tE, probs_tr, 0.0)    # (T, E_nonid)
        rho_t1 = kept_weight_tE.sum(axis=1, keepdims=True)    # (T, 1)
        h_next = h_td + combined_td - rho_t1 * h_td           # (T, d)
        h_next = jnp.where(token_mask_t[:, None], h_next, h_td)

        # Stats (minimal, informative)
        stats = _hop_stats(
            kept_et=kept_et,
            probs_tr=probs_tr,
            mask_tr=mask_tr,
            rho_t1=rho_t1,
            logits_full=logits_full_te,
            token_mask_t=token_mask_t,
            id_col=id_col,
            k_routes=self.routers[0].k if len(self.routers) else 0,
        )
        # Fill identity selection rate using the real mask
        valid_T = token_mask_t.sum().astype(jnp.int32)
        id_topk_count = jnp.sum(jnp.where(token_mask_t, id_selected_t, False)).astype(jnp.int32)
        id_rate = jnp.where(valid_T > 0, id_topk_count / valid_T, 0.0)
        stats["identity_selected_rate"] = id_rate
        return h_next, stats

    # -- forward --------------------------------------------------------------
    @eqx.filter_jit
    def __call__(
        self,
        ids_t: jnp.ndarray,                         # (T,) — caller can vmap over batch
        *,
        key,
        inference: bool,
        attention_mask_t: Optional[jnp.ndarray] = None,  # (T,) bool
        biases: Optional[jnp.ndarray] = None,             # None | (n_hops, E_total)
    ) -> Tuple[jnp.ndarray, Tuple[Dict[str, Any], ...]]:
        # Token mask
        T = ids_t.shape[0]
        token_mask_t = jnp.ones((T,), dtype=bool) if attention_mask_t is None else attention_mask_t.astype(bool)

        # Embed + dropout
        h_td = jax.vmap(self.embed)(ids_t)
        key, sub = jax.random.split(key)
        h_td = self.dropout(h_td, key=sub, inference=inference)

        # RoPE caches
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos_tdh, sin_tdh = rope_cos_sin(T, d_h, self.rope_base)

        # Optional backbone (masked residual)
        for block in self.backbone:
            key, sub = jax.random.split(key)
            out_td = block(h_td, cos_tdh, sin_tdh, key=sub, inference=inference)
            out_td = jnp.where(token_mask_t[:, None], out_td - h_td, 0.0) + h_td
            h_td = out_td

        # Hops: route outside, then apply
        stats_all: list[Dict[str, Any]] = []
        for hop_idx, router in enumerate(self.routers):
            key, rkey, mkey = jax.random.split(key, 3)

            # Optional per-hop expert bias [E_total]
            bias_row = None
            if biases is not None:
                assert biases.ndim == 2 and biases.shape[0] == len(self.routers), \
                    "biases must be (n_hops, E_total)"
                bias_row = biases[hop_idx]

            r_out = router(h_td, key=rkey, inference=inference, bias_e=bias_row)
            h_td, st = self._apply_hop(h_td, r_out, cos_tdh, sin_tdh, token_mask_t, key=mkey, inference=inference)
            stats_all.append(st)

        # Output head
        h_td = jax.vmap(self.ln_out)(h_td)
        logits_tv = jax.vmap(lambda t: t @ self.embed.weight.T)(h_td)  # tied unembedding
        return logits_tv, tuple(stats_all)


# ---- convenience builder ---------------------------------------------------

def build_default_model(
    *,
    vocab: int,
    d_model: int,
    n_heads: int,
    n_experts: int,
    capacity: int,
    topk: int,
    n_hops: int,
    mlp_mult: int,
    dropout: float,
    rope_base: float,
    num_backbone: int,
    key,
) -> Model:
    """Factory: alternating Attention/FFN experts with a matching router stack."""
    k_mods, k_routers, k_backbone, k_model = jax.random.split(key, 4)

    # Build expert set (alternating Attention / FeedForward)
    mod_keys = jax.random.split(k_mods, n_experts)
    experts = tuple(
        Attention(d_model, n_heads, dropout, key=mod_keys[i]) if (i % 2 == 0)
        else FeedForward(d_model, mlp_mult, dropout, key=mod_keys[i])
        for i in range(n_experts)
    )
    group = ModuleGroup(experts, capacity=capacity)

    # Routers (non‑identity experts + one identity at the end)
    total_exp = n_experts + 1
    routers = tuple(TopKRouter(d_model, total_exp, topk, key=k) for k in jax.random.split(k_routers, n_hops))

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

    return Model(
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        rope_base=rope_base,
        group=group,
        routers=routers,
        backbone=backbone,
        dropout_p=dropout,
        key=k_model,
    )