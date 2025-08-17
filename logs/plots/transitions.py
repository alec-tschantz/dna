# logs/plots/transitions.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import wandb

from .utils import has_routing, _collect_top1_indices_all


def _top_used_experts(usage: np.ndarray, top_e: int) -> np.ndarray:
    """Return indices of top-used experts (descending)."""
    top_e = int(min(top_e, usage.size))
    return np.argsort(usage)[::-1][:top_e]


def _make_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=160)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel("count", rotation=90, va="center")
    fig.tight_layout()
    return fig


def log_expert_transition_heatmap(
    model,
    batch: Dict[str, Any],
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    top_experts: int = 24,
):
    """Logs E×E top-1 transition counts across consecutive hops (filtered to top used experts)."""
    if not has_routing(model):
        return

    top1, mask, meta = _collect_top1_indices_all(
        model,
        batch,
        key=key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if top1 is None or meta is None:
        return

    # top1: (H, B, T), mask: (B, T)
    H = int(top1.shape[0])
    B, T = mask.shape
    E = int(meta["n_experts"])
    valid = mask.astype(bool)

    # Usage per expert (how often it was top1 anywhere)
    usage = np.zeros((E,), dtype=np.int64)
    for h in range(H):
        idx = top1[h]  # (B, T)
        usage += np.bincount(idx[valid].ravel(), minlength=E)

    keep = _top_used_experts(usage, top_experts)
    K = keep.size

    # Transition counts across hops (h -> h+1), aggregated over batch+tokens
    trans = np.zeros((E, E), dtype=np.int64)
    for h in range(H - 1):
        a = top1[h][valid]      # (N,)
        b = top1[h + 1][valid]  # (N,)
        np.add.at(trans, (a, b), 1)

    # Filter to top experts
    sub = trans[np.ix_(keep, keep)]

    fig = _make_heatmap(
        sub,
        f"Top-1 expert transitions (top {K}/{E} experts) • step {step}",
        "next hop expert id",
        "current hop expert id",
    )
    logs = {
        "step": step,
        "router/eval/transition_heatmap": wandb.Image(fig),
        "router/eval/transition_self_frac": float(np.trace(trans) / max(trans.sum(), 1)),
        "router/eval/transition_density": float((trans > 0).mean()),
    }
    plt.close(fig)
    wandb.log(logs)


def log_type_transition_heatmap(
    model,
    batch: Dict[str, Any],
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
):
    """Logs type→type transitions (e.g., Attention→FeedForward)."""
    if not has_routing(model):
        return

    top1, mask, meta = _collect_top1_indices_all(
        model,
        batch,
        key=key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if top1 is None or meta is None:
        return

    type_ids = np.asarray(meta["expert_type_ids"])   # (E,)
    labels = list(meta["id_to_type"])                # list of unique type names
    Tn = len(labels)
    H = int(top1.shape[0])
    valid = mask.astype(bool)

    trans = np.zeros((Tn, Tn), dtype=np.int64)
    for h in range(H - 1):
        a = type_ids[top1[h][valid]]
        b = type_ids[top1[h + 1][valid]]
        np.add.at(trans, (a, b), 1)

    fig = _make_heatmap(
        trans,
        f"Type→Type transitions • step {step}",
        "next hop type id",
        "current hop type id",
    )
    ax = fig.axes[0]
    ax.set_xticks(range(Tn)); ax.set_yticks(range(Tn))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.tight_layout()

    logs = {
        "step": step,
        "routing/type_transition_heatmap": wandb.Image(fig),
        "routing/type_self_frac": float(np.trace(trans) / max(trans.sum(), 1)),
    }
    plt.close(fig)
    wandb.log(logs)
