# logs/plots/histograms.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import wandb


def _flatten_per_token(stats_host) -> Dict[str, np.ndarray]:
    """Collect per-token arrays across hops and batch, applying token_mask."""
    if not isinstance(stats_host, (tuple, list)) or len(stats_host) == 0:
        return {}

    rhos, ents, effks = [], [], []
    for hop in stats_host:
        if not hop:
            continue
        rho = np.asarray(hop["rho"])          # (B, T) or (T,)
        ent = np.asarray(hop["entropy"])      # (B, T) or (T,)
        effk = np.asarray(hop["eff_topk"])    # (B, T) or (T,)
        msk = np.asarray(hop["token_mask"])   # (B, T) or (T,)

        # Broadcast to 2D for uniform handling.
        if rho.ndim == 1:   rho = rho[None, :]
        if ent.ndim == 1:   ent = ent[None, :]
        if effk.ndim == 1:  effk = effk[None, :]
        if msk.ndim == 1:   msk = msk[None, :]

        valid = msk.astype(bool)
        rhos.append(rho[valid])
        ents.append(ent[valid])
        effks.append(effk[valid])

    if not rhos:
        return {}

    return {
        "rho": np.concatenate(rhos, axis=0) if rhos else np.array([]),
        "entropy": np.concatenate(ents, axis=0) if ents else np.array([]),
        "eff_topk": np.concatenate(effks, axis=0) if effks else np.array([]),
    }


def _aggregate_per_expert(stats_host) -> Dict[str, np.ndarray]:
    """Collect per-expert arrays (load, importance) across hops and batch."""
    loads, importances = [], []
    for hop in stats_host:
        if not hop:
            continue
        load = np.asarray(hop["load"])            # (B, E) or (E,)
        imp = np.asarray(hop["importance"])       # (B, E) or (E,)

        if load.ndim == 1: load = load[None, :]
        if imp.ndim == 1:  imp = imp[None, :]

        loads.append(load)
        importances.append(imp)

    if not loads:
        return {}

    loads = np.concatenate(loads, axis=0)          # (B*, E)
    importances = np.concatenate(importances, 0)   # (B*, E)
    return {"load": loads, "importance": importances}


def _matplot_hist(data: np.ndarray, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    ax.hist(data, bins=40)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def _matplot_bars(y: np.ndarray, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=150)
    ax.bar(np.arange(len(y)), y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def log_router_histograms(stats_host, *, step: int, prefix: str = "routing"):
    """
    Logs:
      - Histograms: rho, entropy, eff_topk (per-token).
      - Bar chart: per-expert load (summed across batch & hops).
      - Scalars: dropped-edge rate approx via rho==0, summary percentiles.
    """
    per_token = _flatten_per_token(stats_host)
    per_expert = _aggregate_per_expert(stats_host)

    logs: Dict[str, Any] = {"step": step}

    # --- Per-token: rho / entropy / eff_topk ---
    if per_token:
        rho = per_token["rho"]
        ent = per_token["entropy"]
        effk = per_token["eff_topk"]

        # Histograms
        fig_rho = _matplot_hist(rho, "ρ distribution (per token)", "ρ")
        fig_ent = _matplot_hist(ent, "Routing entropy (per token)", "Entropy")
        fig_eff = _matplot_hist(effk, "Effective top-k (per token)", "k_eff")

        logs[f"{prefix}/rho_hist"] = wandb.Image(fig_rho)
        logs[f"{prefix}/entropy_hist"] = wandb.Image(fig_ent)
        logs[f"{prefix}/efftopk_hist"] = wandb.Image(fig_eff)

        # Scalars
        if rho.size:
            logs.update({
                f"{prefix}/rho_p10": float(np.percentile(rho, 10)),
                f"{prefix}/rho_p50": float(np.percentile(rho, 50)),
                f"{prefix}/rho_p90": float(np.percentile(rho, 90)),
                f"{prefix}/frac_rho_zero": float((rho <= 1e-9).mean()),
            })
        if ent.size:
            logs.update({
                f"{prefix}/entropy_mean_token": float(ent.mean()),
                f"{prefix}/entropy_p90_token": float(np.percentile(ent, 90)),
            })
        if effk.size:
            logs.update({
                f"{prefix}/efftopk_mean_token": float(effk.mean()),
            })

        plt.close(fig_rho); plt.close(fig_ent); plt.close(fig_eff)

    # --- Per-expert: load bars & importance summary ---
    if per_expert:
        load = per_expert["load"].sum(axis=0)          # (E,)
        imp = per_expert["importance"].sum(axis=0)     # (E,)

        fig_load = _matplot_bars(load, "Expert load (summed)", "expert id", "tokens kept")
        logs[f"{prefix}/expert_load_bars"] = wandb.Image(fig_load)
        logs[f"{prefix}/expert_load_mean"] = float(load.mean())
        logs[f"{prefix}/expert_load_gini"] = _gini(load)
        if imp.size:
            logs[f"{prefix}/importance_gini"] = _gini(imp)
        plt.close(fig_load)

    wandb.log(logs)


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = x.flatten()
    if np.any(x < 0):
        x = x - x.min()
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # Gini via Lorenz curve: 1 + 1/n - 2 * sum((n+1-i)*x_i) / (n * sum x)
    g = 1.0 + 1.0 / n - 2.0 * np.sum((np.arange(1, n + 1)) * x_sorted) / (n * cum[-1])
    return float(max(0.0, min(1.0, g)))
