from typing import Any, Dict, List, Optional, Tuple
import textwrap
import plotly.graph_objects as go
from rich.pretty import pprint as rpprint
from collections import Counter
from pathlib import Path
from datetime import datetime
import json

import matplotlib as mpl

mpl.rcParams["text.usetex"] = False
from matplotlib.patches import Rectangle, PathPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dna import generate, Attention, FeedForward, Identity


def print_ascii_dna() -> None:
    D = [
        "███████  ",
        "██    ██ ",
        "██    ██ ",
        "██    ██ ",
        "██    ██ ",
        "███████  ",
    ]
    N = [
        "██    ██ ",
        "███   ██ ",
        "██ ██ ██ ",
        "██  ███  ",
        "██   ███ ",
        "██    ██ ",
    ]
    A = [
        "  █████  ",
        " ██   ██ ",
        "██     ██",
        "█████████",
        "██     ██",
        "██     ██",
    ]

    rows = max(len(D), len(N), len(A))
    lines = []
    for i in range(rows):
        d = D[i] if i < len(D) else " " * len(D[0])
        n = N[i] if i < len(N) else " " * len(N[0])
        a = A[i] if i < len(A) else " " * len(A[0])
        lines.append(d + "   " + n + "   " + a)

    h, w = len(lines), max(len(l) for l in lines)
    dx, dy = 2, 1
    canvas = [[" " for _ in range(w + dx)] for _ in range(h + dy)]
    for y in range(h):
        for x, ch in enumerate(lines[y]):
            if ch != " ":
                if canvas[y + dy][x + dx] == " ":
                    canvas[y + dy][x + dx] = "░"
                canvas[y][x] = ch

    shaded = ["".join(row).rstrip() for row in canvas]
    width = max(66, min(140, max(len(s) for s in shaded) + 4))  # nice box width
    print(ansi_box("DNA", shaded, width=width))

def count_params(tree) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(x.size for x in leaves))


def l2_tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq + 1e-12))


def router_l2_norm(model) -> float:
    if has_routing(model):
        leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
        if not leaves:
            return 0.0
        sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
        return float(jnp.sqrt(sq + 1e-12))
    return 0.0



def batch_seq_stats(mask: jnp.ndarray, seq_len: int) -> Tuple[float, int, int, float]:
    lens = jnp.sum(mask, axis=1)
    return (
        float(jnp.mean(lens)),
        int(jnp.min(lens)),
        int(jnp.max(lens)),
        float(jnp.mean(seq_len - lens)),
    )


def has_routing(model) -> bool:
    return hasattr(model, "routers") and getattr(model, "routers") is not None


def ansi_box(title: str, body_lines: list[str], width: int = 88) -> str:
    """Render a heavy-line ANSI box with a title and wrapped body lines."""
    h, v = "─", "│"
    tl, tr, bl, br = "┌", "┐", "└", "┘"
    sep_left, sep_right = "├", "┤"

    def pad(s: str) -> str:
        s = s.replace("\t", " ")[: width - 4]
        return s + " " * (width - 4 - len(s))

    top = f"{tl}{h*(width-2)}{tr}"
    title_line = f"{v} {pad(title)} {v}"
    mid = f"{sep_left}{h*(width-2)}{sep_right}"
    bot = f"{bl}{h*(width-2)}{br}"

    lines = [top, title_line, mid]
    for ln in body_lines:
        for w in textwrap.wrap(ln, width=width - 4, break_long_words=False):
            lines.append(f"{v} {pad(w)} {v}")
    lines.append(bot)
    return "\n".join(lines)



def extra_routing_metrics(stats_host, prefix: str) -> Dict[str, float]:
    """Extra routing metrics."""
    if not isinstance(stats_host, (tuple, list)) or len(stats_host) == 0:
        return {}

    sel_sum = 0
    kept_sum = 0
    rho_all, mask_all = [], []

    for hop in stats_host:
        if not hop:
            continue
        sel_sum += int(np.asarray(hop["selected_edges"]).sum())
        kept_sum += int(np.asarray(hop["kept_edges"]).sum())
        rho_all.append(np.asarray(hop["rho"]))
        mask_all.append(np.asarray(hop["token_mask"]))

    if not rho_all:
        return {}

    rho = np.concatenate(rho_all, axis=1)
    msk = np.concatenate(mask_all, axis=1)
    rho_valid = rho[msk.astype(bool)]

    dropped_edge_rate = float((sel_sum - kept_sum) / max(sel_sum, 1))
    frac_tokens_rho0 = float((rho_valid <= 1e-9).sum() / max(rho_valid.size, 1))
    p10 = float(np.percentile(rho_valid, 10.0)) if rho_valid.size else 0.0
    p90 = float(np.percentile(rho_valid, 90.0)) if rho_valid.size else 0.0
    mean = float(rho_valid.mean()) if rho_valid.size else 0.0

    return {
        f"{prefix}/dropped_edge_rate": dropped_edge_rate,
        f"{prefix}/frac_tokens_rho0": frac_tokens_rho0,
        f"{prefix}/rho_mean": mean,
        f"{prefix}/rho_p10": p10,
        f"{prefix}/rho_p90": p90,
    }



def routing_metrics_from_stats(
    stats_host, *, prefix: str, capacity: Optional[int] = None
) -> Dict[str, float]:
    """Aggregate per-hop stats into scalar logs."""
    if not isinstance(stats_host, (tuple, list)) or len(stats_host) == 0:
        return {}

    logs = []
    for hop in stats_host:
        if not hop:
            continue

        load = np.asarray(hop["load"])
        imp = np.asarray(hop["importance"])
        rho = np.asarray(hop["rho"])
        ent = np.asarray(hop["entropy"])
        effk = np.asarray(hop["eff_topk"])
        mask = np.asarray(hop["token_mask"])

        rho_m = np.nanmean(np.where(mask, rho, np.nan))
        ent_m = np.nanmean(np.where(mask, ent, np.nan))
        effk_m = np.nanmean(np.where(mask, effk, np.nan))

        util = (load > 0).mean() if load.size else 0.0
        denom = load.sum(keepdims=True) + 1e-9
        load_std = (load / denom).std() if load.size else 0.0

        if capacity and capacity > 0 and load.size:
            cap_util = (load / float(capacity)).mean()
            cap_sat = (load >= capacity).mean()
        else:
            cap_util, cap_sat = 0.0, 0.0

        logs.append(
            dict(
                rho_mean=rho_m,
                entropy_mean=ent_m,
                effk_mean=effk_m,
                util=float(util),
                load_std=float(load_std),
                cap_util=float(cap_util),
                cap_sat=float(cap_sat),
                importance_mean=float(imp.mean()) if imp.size else 0.0,
            )
        )

    if not logs:
        return {}

    def _mean(key):
        vals = [h[key] for h in logs]
        return float(sum(vals) / max(len(vals), 1))

    return {
        f"{prefix}/rho_mean": _mean("rho_mean"),
        f"{prefix}/entropy_mean": _mean("entropy_mean"),
        f"{prefix}/eff_topk_mean": _mean("effk_mean"),
        f"{prefix}/util": _mean("util"),
        f"{prefix}/load_std": _mean("load_std"),
        f"{prefix}/cap_util": _mean("cap_util"),
        f"{prefix}/cap_saturated_frac": _mean("cap_sat"),
        f"{prefix}/importance_mean": _mean("importance_mean"),
    }

