# logs/plots/phase_portrait.py
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import wandb

from .utils import has_routing, _forward_batched_for_stats, _expert_color_table


def _build_selected_from_probs(
    probs_bte: np.ndarray,          # (B,T,E)
    effk_bt: np.ndarray,            # (B,T)
    token_mask_bt: np.ndarray,      # (B,T) bool
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
    selected_bte: np.ndarray,       # (B,T,E) bool
    scores_bte: np.ndarray,         # (B,T,E) float (use probs as proxy)
    token_mask_bt: np.ndarray,      # (B,T) bool
    capacity: Optional[int],        # tokens per expert per batch item
) -> np.ndarray:
    """Approximate capacity: for each (b,e), keep top-C tokens by score among selected."""
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


def log_expert_phase_portrait(
    model,
    batch: Dict[str, Any],
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    top_experts: int = 42,
):
    if not has_routing(model):
        return

    # Forward to get per-hop batched stats
    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B = int(ids.shape[0])
    fkeys = jax.random.split(jax.random.fold_in(key, 7771), B)
    stats_all = _forward_batched_for_stats(
        model, ids, mask, fkeys,
        gumbel_tau=gumbel_tau, router_temp=router_temp, select_temp=select_temp
    )
    if not isinstance(stats_all, (tuple, list)) or len(stats_all) == 0:
        return

    # Determine E from routing_probs
    first = stats_all[0]
    probs0 = np.asarray(first["routing_probs"])
    if probs0.ndim == 2:  # (T,E) -> (B,T,E) safety
        probs0 = probs0[None, ...]
    E = int(probs0.shape[-1])

    # Accumulators across hops
    selected = np.zeros((E,), dtype=np.float64)
    kept = np.zeros((E,), dtype=np.float64)
    top1_count = np.zeros((E,), dtype=np.float64)
    top1_prob_sum = np.zeros((E,), dtype=np.float64)

    capacity = int(getattr(model, "capacity", 0) or 0)

    for hop in stats_all:
        token_mask_bt = np.asarray(hop["token_mask"]).astype(bool)     # (B,T)
        p_bte = np.asarray(hop["routing_probs"]).astype(np.float32)    # (B,T,E) or (T,E)
        if p_bte.ndim == 2:
            p_bte = p_bte[None, ...]
        # eff_topk might be (B,T). If missing, fall back to router.k
        eff = hop.get("eff_topk", None)
        if eff is None:
            try:
                k_guess = int(getattr(model.routers[0], "k", 1))
            except Exception:
                k_guess = 1
            effk_bt = np.full(token_mask_bt.shape, k_guess, dtype=int)
        else:
            effk_bt = np.asarray(eff).astype(int)
            if effk_bt.ndim == 1:
                effk_bt = effk_bt[None, :]

        # Rebuild selection and approximate kept
        M_bte = _build_selected_from_probs(p_bte, effk_bt, token_mask_bt)          # selected
        K_bte = _approx_kept_from_capacity(M_bte, p_bte, token_mask_bt, capacity)  # kept (approx)

        # Flatten valid positions
        b, t = np.where(token_mask_bt)
        if b.size == 0:
            continue
        P = p_bte[b, t, :]        # (N,E)
        M = M_bte[b, t, :]        # (N,E)
        K = K_bte[b, t, :]        # (N,E)

        selected += M.sum(axis=0)
        kept += K.sum(axis=0)

        # Top-1 expert per token (by routing_probs)
        idx = np.argmax(P, axis=-1)                 # (N,)
        vals = P[np.arange(P.shape[0]), idx]        # (N,)
        cnt = np.bincount(idx, minlength=E)         # (E,)
        top1_count += cnt
        add = np.zeros((E,), dtype=np.float64)
        np.add.at(add, idx, vals.astype(np.float64))
        top1_prob_sum += add

    usage = selected.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        kept_rate = np.where(selected > 0, kept / selected, 0.0)
        mean_top1_prob = np.where(top1_count > 0, top1_prob_sum / top1_count, 0.0)

    # Pick top experts by usage
    keep = np.argsort(usage)[::-1][:min(top_experts, E)]
    K = keep.size
    x = kept_rate[keep]
    y = mean_top1_prob[keep]
    s = usage[keep]

    # Normalize bubble sizes
    s_norm = s / (s.max() + 1e-9)
    sizes = 150.0 * np.sqrt(np.clip(s_norm, 0.0, 1.0)) + 12.0

    # Colors by type if available
    try:
        from .utils import _expert_meta
        meta = _expert_meta(model)
    except Exception:
        meta = None

    if meta is not None:
        sub_meta = {
            "n_experts": K,
            "expert_type_ids": np.asarray(meta["expert_type_ids"])[keep],
            "id_to_type": list(meta["id_to_type"]),
        }
        cols = _expert_color_table(sub_meta)
    else:
        cols = np.tile(np.array([[0.2, 0.5, 0.9, 1.0]]), (K, 1))

    # Figure
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10.5, 7.5), dpi=170)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.5], width_ratios=[1.0, 1.0],
                           hspace=0.25, wspace=0.20)

    ax = fig.add_subplot(gs[0, :])
    ax.scatter(x, y, s=sizes, c=cols, edgecolors="white", linewidths=0.8)
    ax.set_xlabel("kept rate (= kept / selected)")
    ax.set_ylabel("mean top-1 probability (when expert wins)")
    ax.set_title(f"Expert Phase Portrait • top {K}/{E} • step {step}", fontsize=12, weight="bold")
    ax.grid(alpha=0.25)

    # Annotate a few large-usage experts
    top_show = np.argsort(s)[::-1][:min(10, K)]
    for i in top_show:
        ax.text(x[i], y[i], str(keep[i]), fontsize=8, weight="bold",
                color="#222", ha="center", va="center")

    # Marginal histograms
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(x, bins=30)
    ax2.set_title("Kept rate distribution")
    ax2.set_xlabel("kept rate")
    ax2.set_ylabel("experts")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(y, bins=30)
    ax3.set_title("Mean top-1 probability distribution")
    ax3.set_xlabel("mean top-1 prob")
    ax3.set_ylabel("experts")

    fig.suptitle("Routing Effectiveness vs Confidence (approx from probs + eff_topk)",
                 y=0.98, fontsize=13, weight="bold")
    wandb.log({"routing/expert_phase": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)
