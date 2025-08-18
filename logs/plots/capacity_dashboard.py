# logs/plots/capacity_dashboard.py
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import wandb

from .utils import has_routing, _forward_batched_for_stats, _expert_color_table


def _sum_over_bt(x_bte: np.ndarray, valid_bt: np.ndarray) -> np.ndarray:
    """Sum over valid (B,T) positions, return (E,) counts."""
    b, t = np.where(valid_bt)
    if b.size == 0:
        return np.zeros((x_bte.shape[-1],), dtype=np.float64)
    X = x_bte[b, t, :].astype(np.float64)  # (N,E)
    return X.sum(axis=0)  # (E,)


def _build_selected_from_probs(
    probs_bte: np.ndarray,  # (B,T,E)
    effk_bt: np.ndarray,  # (B,T)
    token_mask_bt: np.ndarray,  # (B,T) bool
) -> np.ndarray:
    """Rebuild selection mask per token from routing_probs and eff_topk."""
    B, T, E = probs_bte.shape
    selected = np.zeros((B, T, E), dtype=bool)
    effk_bt = np.clip(effk_bt.astype(int), 0, E)
    for b in range(B):
        for t in range(T):
            if not token_mask_bt[b, t]:
                continue
            k = int(effk_bt[b, t])
            if k <= 0:
                continue
            row = probs_bte[b, t]
            idx = np.argpartition(row, -k)[-k:]
            selected[b, t, idx] = True
    return selected


def _approx_kept_from_capacity(
    selected_bte: np.ndarray,  # (B,T,E) bool
    scores_bte: np.ndarray,  # (B,T,E) float (use probs as proxy)
    token_mask_bt: np.ndarray,  # (B,T) bool
    capacity: Optional[int],  # tokens per expert per sequence
) -> np.ndarray:
    """Approximate capacity filter: for each expert & batch item, keep top-C tokens by score."""
    B, T, E = selected_bte.shape
    if not capacity or capacity <= 0:
        return selected_bte.copy()
    kept = np.zeros_like(selected_bte, dtype=bool)
    for b in range(B):
        valid_t = token_mask_bt[b]
        for e in range(E):
            pos = np.where(selected_bte[b, :, e] & valid_t)[0]
            if pos.size == 0:
                continue
            if pos.size <= capacity:
                kept[b, pos, e] = True
            else:
                vals = scores_bte[b, pos, e]
                top = np.argpartition(vals, -capacity)[-capacity:]
                kept[b, pos[top], e] = True
    return kept


def log_capacity_saturation_dashboard(
    model,
    batch: Dict[str, Any],
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    capacity: int,
    top_experts: int = 36,
):
    if not has_routing(model):
        return

    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B = int(ids.shape[0])
    fkeys = jax.random.split(jax.random.fold_in(key, 331), B)
    stats_all = _forward_batched_for_stats(
        model,
        ids,
        mask,
        fkeys,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if not isinstance(stats_all, (tuple, list)) or len(stats_all) == 0:
        return

    # Determine E, H from routing_probs
    first = stats_all[0]
    probs0 = np.asarray(first["routing_probs"])
    if probs0.ndim == 2:  # (T,E) -> (B,T,E) safety
        probs0 = probs0[None, ...]
    E = int(probs0.shape[-1])
    H = len(stats_all)

    drop_frac_EH = np.zeros((E, H), dtype=np.float64)
    kept_tot_H = np.zeros((H,), dtype=np.float64)
    drop_tot_H = np.zeros((H,), dtype=np.float64)
    load_mean_H = np.zeros((H,), dtype=np.float64)
    usage_total = np.zeros((E,), dtype=np.float64)

    for h, hop in enumerate(stats_all):
        token_mask_bt = np.asarray(hop["token_mask"]).astype(bool)  # (B,T)
        probs_bte = np.asarray(hop["routing_probs"], dtype=np.float32)  # (B,T,E)
        if probs_bte.ndim == 2:
            probs_bte = probs_bte[None, ...]
        eff = hop.get("eff_topk", None)
        if eff is None:
            # fallback: guess from router.k if present
            try:
                k_guess = int(getattr(model.routers[0], "k", 1))
            except Exception:
                k_guess = 1
            effk_bt = np.full(token_mask_bt.shape, k_guess, dtype=int)
        else:
            effk_bt = np.asarray(eff).astype(int)
            if effk_bt.ndim == 1:  # safety
                effk_bt = effk_bt[None, :]

        # Rebuild selected and approx kept
        selected_bte = _build_selected_from_probs(probs_bte, effk_bt, token_mask_bt)
        kept_bte = _approx_kept_from_capacity(
            selected_bte, probs_bte, token_mask_bt, capacity
        )

        sel_e = _sum_over_bt(selected_bte, token_mask_bt)  # (E,)
        kep_e = _sum_over_bt(kept_bte, token_mask_bt)  # (E,)
        dro_e = np.clip(sel_e - kep_e, 0.0, None)

        kept_tot_H[h] = float(kep_e.sum())
        drop_tot_H[h] = float(dro_e.sum())
        usage_total += sel_e

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(sel_e > 0, dro_e / sel_e, 0.0)
        drop_frac_EH[:, h] = frac

        load_mean_H[h] = float(np.mean(kep_e / max(capacity, 1)))

    # Sort experts by total usage and keep top K
    order = np.argsort(usage_total)[::-1]
    keep_idx = order[: min(top_experts, E)]
    K = keep_idx.size
    df_sub = drop_frac_EH[keep_idx, :]  # (K,H)

    # Colors per expert (legend-friendly). If you have true type ids, pass them here.
    meta_for_colors = {
        "n_experts": K,
        "expert_type_ids": np.zeros((K,), dtype=np.int32),
        "id_to_type": ["Expert"],
    }
    _ = _expert_color_table(
        meta_for_colors
    )  # we don't strictly need colors for this dashboard

    # --- Figure layout ---
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12.5, 9.5), dpi=160)
    gs = gridspec.GridSpec(
        3,
        2,
        height_ratios=[2.0, 1.0, 1.0],
        width_ratios=[1.4, 1.0],
        hspace=0.32,
        wspace=0.28,
    )

    # Heatmap: drop fraction per expert × hop
    ax0 = fig.add_subplot(gs[0, :])
    im = ax0.imshow(df_sub, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax0.set_title(
        f"Capacity Drop Fraction per Expert × Hop • top {K}/{E} • step {step}",
        fontsize=12,
        weight="bold",
    )
    ax0.set_xlabel("Hop")
    ax0.set_ylabel("Expert (usage-sorted)")
    cbar = plt.colorbar(im, ax=ax0, shrink=0.9)
    cbar.ax.set_ylabel("drop fraction", rotation=90, va="center")

    # Stacked bars: kept vs dropped totals per hop
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.bar(np.arange(H), kept_tot_H, label="kept", alpha=0.85)
    ax1.bar(np.arange(H), drop_tot_H, bottom=kept_tot_H, label="dropped", alpha=0.65)
    ax1.set_title("Total kept vs dropped per hop")
    ax1.set_xlabel("Hop")
    ax1.set_ylabel("count")
    ax1.legend(frameon=False)

    # Line: average capacity utilization per hop
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(np.arange(H), load_mean_H, marker="o")
    ax2.axhline(1.0, color="black", lw=1.0, alpha=0.4)
    ax2.set_ylim(0, max(1.15, float(load_mean_H.max()) + 0.05))
    ax2.set_title("Average capacity utilization (mean kept / capacity)")
    ax2.set_xlabel("Hop")
    ax2.set_ylabel("utilization")

    # Histogram of expert drop fractions (aggregated)
    ax3 = fig.add_subplot(gs[2, :])
    hist = df_sub.flatten()
    ax3.hist(hist, bins=40)
    ax3.set_title("Distribution of per-expert drop fractions (all hops)")
    ax3.set_xlabel("drop fraction")
    ax3.set_ylabel("experts×hops")

    fig.suptitle(
        "Capacity Pressure Dashboard (approx from probs + eff_topk)",
        y=0.98,
        fontsize=13,
        weight="bold",
    )
    wandb.log({"routing/dashboard": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)
