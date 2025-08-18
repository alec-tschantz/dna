# logs/plots/co_usage.py
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Circle
import jax
import jax.numpy as jnp
import wandb

from .utils import (
    has_routing,
    _expert_color_table,
    evaluate_for_visuals,      # meta + light batch stats
    _forward_batched_for_stats # full per-hop batched stats
)

def _flatten_valid(mask_bt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return linear indices (B,T) where token is valid."""
    return np.where(mask_bt.astype(bool))

def _aggregate_cooc(
    selected_bte: np.ndarray,
    kept_bte: np.ndarray,
    valid_bt: Tuple[np.ndarray, np.ndarray],
):
    """Return (E,E) co-occur matrices for selected and kept."""
    b_idx, t_idx = valid_bt
    Xsel = selected_bte[b_idx, t_idx, :].astype(np.float32)  # (N,E)
    Xkep = kept_bte[b_idx, t_idx, :].astype(np.float32)      # (N,E)
    co_sel = Xsel.T @ Xsel
    co_kep = Xkep.T @ Xkep
    np.fill_diagonal(co_sel, 0.0)
    np.fill_diagonal(co_kep, 0.0)
    usage = Xsel.sum(axis=0)  # how often expert selected
    return co_sel, co_kep, usage

def _circle_layout(n: int, radius: float = 1.0, start_angle: float = np.pi / 2) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ang = start_angle + np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([radius*np.cos(ang), radius*np.sin(ang)], axis=-1)

def _bezier_arc(p0, p1, bend: float = 0.3):
    c = np.array([0.0, 0.0])
    v0 = (p0 - c); v1 = (p1 - c)
    m = (p0 + p1) / 2.0
    n = (v0/np.linalg.norm(v0) + v1/np.linalg.norm(v1))
    if np.linalg.norm(n) < 1e-6:
        n = (p0 - p1)[::-1] * np.array([1, -1])
    n = n / (np.linalg.norm(n) + 1e-9)
    ctrl = m + bend * n
    return np.array([p0, ctrl, ctrl, p1], dtype=np.float32)

def _draw_graph(ax, pos, node_sizes, node_colors, edges, title: str):
    # edges: list of (i, j, weight, kept_weight)
    maxw = max((w for _, _, w, _ in edges), default=1.0)
    for i, j, w, k in edges:
        if w <= 0:
            continue
        p0, p1 = pos[i], pos[j]
        bez = _bezier_arc(p0, p1, bend=0.28)
        width = 0.6 + 5.4 * (w / (maxw + 1e-9))
        keep_alpha = 0.2 + 0.75 * (k / (w + 1e-9))
        path = MplPath(bez, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, lw=width, facecolor="none", edgecolor="black", alpha=0.12))
        ax.add_patch(PathPatch(path, lw=max(0.8, 0.6*width), facecolor="none", edgecolor=(0,0,0,keep_alpha)))
    # nodes
    for i, (xy, sz, col) in enumerate(zip(pos, node_sizes, node_colors)):
        r = 0.03 + 0.06 * sz
        circ = Circle(xy, r, facecolor=col, edgecolor="white", linewidth=1.0, zorder=3)
        ax.add_patch(circ)
        ax.text(xy[0], xy[1], str(i), ha="center", va="center", fontsize=7.5,
                color="white", weight="bold", zorder=4)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title, fontsize=12, weight="bold")

# ---------------------------
# Selection/keeping rebuilds
# ---------------------------

def _build_selected_from_probs(
    probs_bte: np.ndarray,          # (B,T,E)
    effk_bt: np.ndarray,            # (B,T) effective top-k per token
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
            # top-k indices for this (b,t)
            idx = np.argpartition(row, -k)[-k:]
            selected[b, t, idx] = True
    return selected

def _approx_kept_from_capacity(
    selected_bte: np.ndarray,       # (B,T,E) bool
    scores_bte: np.ndarray,         # (B,T,E) float (use probs as proxy)
    token_mask_bt: np.ndarray,      # (B,T) bool
    capacity: Optional[int],        # tokens per expert per sequence
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

# ---------------------------
# Public logger
# ---------------------------

def log_expert_co_usage_graph(
    model,
    batch: Dict[str, Any],
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    top_experts: int = 24,
    min_edge_frac: float = 0.04,
):
    """Circular network of expert co-usage across all hops (selected vs kept α)."""
    if not has_routing(model):
        return

    # Meta via evaluate_for_visuals (types, labels, counts, etc.)
    batch_stats, _ = evaluate_for_visuals(
        model,
        batch,
        key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
        num_examples=2,
    )
    if batch_stats is None:
        return

    E = int(batch_stats["n_experts"])

    # Get full per-hop batched stats (B,T,E stuff)
    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B = int(ids.shape[0])
    fkeys = jax.random.split(jax.random.fold_in(key, 9871), B)
    stats_all = _forward_batched_for_stats(
        model, ids, mask, fkeys,
        gumbel_tau=gumbel_tau, router_temp=router_temp, select_temp=select_temp
    )
    if not isinstance(stats_all, (tuple, list)) or len(stats_all) == 0:
        return

    # Aggregate co-usage over hops
    co_sel = np.zeros((E, E), dtype=np.float64)
    co_kep = np.zeros((E, E), dtype=np.float64)
    usage = np.zeros((E,), dtype=np.float64)

    capacity = int(getattr(model, "capacity", 0) or 0)

    for hop in stats_all:
        token_mask_bt = np.asarray(hop["token_mask"]).astype(bool)      # (B,T)
        probs_bte = np.asarray(hop["routing_probs"], dtype=np.float32)  # (B,T,E)

        # eff_topk might be (B,T) after vmap; if not present, fallback to k from router
        eff = hop.get("eff_topk", None)
        if eff is None:
            # fallback: use router.k if available (assumes constant across hops)
            try:
                k_guess = int(getattr(model.routers[0], "k", 1))
            except Exception:
                k_guess = 1
            effk_bt = np.full(token_mask_bt.shape, k_guess, dtype=int)
        else:
            effk_bt = np.asarray(eff).astype(int)
            if effk_bt.ndim == 1:  # safety
                effk_bt = effk_bt[None, :]

        # Rebuild selected (B,T,E)
        selected_bte = _build_selected_from_probs(probs_bte, effk_bt, token_mask_bt)
        # Approximate kept (capacity)
        kept_bte = _approx_kept_from_capacity(selected_bte, probs_bte, token_mask_bt, capacity)

        valid_bt = _flatten_valid(token_mask_bt)
        if valid_bt[0].size == 0:
            continue
        cs, ck, use = _aggregate_cooc(selected_bte, kept_bte, valid_bt)
        co_sel += cs
        co_kep += ck
        usage += use

    # Keep top experts by usage for readability
    keep = np.argsort(usage)[::-1][:min(top_experts, E)]
    K = keep.size
    co_sel = co_sel[np.ix_(keep, keep)]
    co_kep = co_kep[np.ix_(keep, keep)]
    usage_sub = usage[keep]

    # Layout + aesthetics
    pos = _circle_layout(K, radius=1.0)
    sizes = usage_sub / (usage_sub.max() + 1e-9)
    sizes = np.sqrt(np.clip(sizes, 0.0, 1.0))

    # Colors per expert (respecting types)
    meta_for_colors = {
        "n_experts": K,
        "expert_type_ids": np.asarray(batch_stats["expert_type_ids"])[keep],
        "id_to_type": batch_stats["id_to_type"],
    }
    colors_all = _expert_color_table(meta_for_colors)

    # Build edges with threshold
    edges: List[Tuple[int, int, float, float]] = []
    max_edge = float(co_sel.max()) if co_sel.size else 1.0
    thr = max_edge * float(min_edge_frac)
    for a in range(K):
        for b in range(a + 1, K):
            w = float(co_sel[a, b])
            if w <= thr:
                continue
            k = float(co_kep[a, b])
            edges.append((a, b, w, k))

    # Draw
    fig, ax = plt.subplots(figsize=(8.8, 8.8), dpi=170)
    _draw_graph(
        ax,
        pos,
        sizes,
        colors_all,
        edges,
        title=f"Expert Co-Usage (selected vs kept α) • top {K}/{E} • step {step}",
    )

    kept_ratio = (co_kep.sum() / max(co_sel.sum(), 1.0)) if co_sel.size else 0.0
    fig.text(0.01, 0.01, f"global kept/selected ratio (approx): {kept_ratio:.2f}",
             fontsize=9, color="#444")

    wandb.log({"routing/expert_co_usage": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)
