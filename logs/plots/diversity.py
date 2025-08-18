from typing import Any, Dict, List
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import wandb

from .utils import (
    _collect_top1_indices_all,
    _expert_color_table,
)


def analyze_path_diversity(
    model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    top_paths: int = 12,
):
    """Compute entropy, effective #paths, Gini, coverage + top-N barcodes."""
    top1, mask_bt, meta = _collect_top1_indices_all(
        model,
        batch,
        key=key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if top1 is None:
        return None

    H, B, T = top1.shape
    mask_np = np.asarray(mask_bt)
    ids_np = np.asarray(batch["input_ids"])

    paths = []
    token_ids = []
    for b in range(B):
        for t in range(T):
            if not mask_np[b, t]:
                continue
            paths.append(tuple(int(top1[h, b, t]) for h in range(H)))
            token_ids.append(int(ids_np[b, t]))
    if not paths:
        return None

    counts = Counter(paths)
    total = sum(counts.values())
    items = counts.most_common()
    probs = np.array([c / total for _, c in items], dtype=np.float64)

    # diversity metrics
    H_shannon = float(-(probs * np.log(probs + 1e-12)).sum())
    unique = len(items)
    H_norm = float(H_shannon / max(np.log(max(unique, 1)), 1e-12))  # 0..1
    eff_num = float(np.exp(H_shannon))  # effective #paths

    p_sorted = np.sort(probs)
    cum = np.cumsum(p_sorted)
    lorenz = np.concatenate([[0.0], cum])
    area = np.trapz(lorenz, dx=1.0 / len(p_sorted))
    gini = float(1.0 - 2.0 * area)

    def topk_cov(k: int) -> float:
        return float(sum(c for _, c in items[:k]) / total)

    cov5, cov10, cov20 = topk_cov(5), topk_cov(10), topk_cov(20)

    expert_rgba = _expert_color_table(meta)
    topN = items[:top_paths]

    return dict(
        meta=meta,
        H=H,
        total_tokens=total,
        unique_paths=unique,
        shannon=H_shannon,
        shannon_norm=H_norm,
        effective_paths=eff_num,
        gini=gini,
        coverage={5: cov5, 10: cov10, 20: cov20},
        probs=probs,
        items=items,
        topN=topN,
        expert_rgba=expert_rgba,
    )


def plot_path_diversity_dashboard(div: Dict[str, Any], step: int):
    """Multi-panel summary (metric tiles, Lorenz, top paths, barcodes)."""
    if div is None:
        return

    topN = div["topN"]
    H = div["H"]
    expert_rgba = div["expert_rgba"]

    # figure grid
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.4], wspace=0.25, hspace=0.35)

    axA = fig.add_subplot(gs[0, :])
    axA.axis("off")
    tiles = [
        ("Unique paths", f"{div['unique_paths']:,}"),
        ("Effective #paths", f"{div['effective_paths']:.1f}"),
        ("Entropy (norm.)", f"{div['shannon_norm']:.2f}"),
        ("Gini (↑ better spread)", f"{div['gini']:.2f}"),
        ("Top-5 cov.", f"{div['coverage'][5]*100:.1f}%"),
        ("Top-10 cov.", f"{div['coverage'][10]*100:.1f}%"),
    ]
    x = 0.02
    for title, value in tiles:
        box = FancyBboxPatch(
            (x, 0.15),
            0.15,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#ddd",
            facecolor="#f7f7f9",
        )
        axA.add_patch(box)
        axA.text(
            x + 0.075, 0.58, value, ha="center", va="center", fontsize=16, weight="bold"
        )
        axA.text(
            x + 0.075, 0.36, title, ha="center", va="center", fontsize=10, color="#555"
        )
        x += 0.165
    axA.set_title(
        f"Path Diversity • step {step}", loc="left", fontsize=13, weight="bold"
    )

    axB = fig.add_subplot(gs[1, 0])
    p_sorted = np.sort(div["probs"])
    L = np.concatenate([[0.0], np.cumsum(p_sorted)])
    X = np.linspace(0, 1, len(L))
    axB.plot(X, L, linewidth=2.2)
    axB.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    axB.set_xlabel("Cumulative fraction of paths")
    axB.set_ylabel("Cumulative fraction of tokens")
    axB.set_title(f"Lorenz Curve (Gini={div['gini']:.2f})")

    axC = fig.add_subplot(gs[1, 1])
    labels = [f"P{i+1}" for i in range(len(topN))]
    counts = [c for _, c in topN]
    bars = axC.bar(labels, counts)
    axC.set_ylabel("Token count")
    axC.set_title("Top paths")
    for b, c in zip(bars, counts):
        axC.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axD = fig.add_subplot(gs[1, 2])
    color_grid = np.zeros((len(topN), H, 4), dtype=np.float32)
    for i, (path, _) in enumerate(topN):
        e_ids = np.array(path, dtype=int)
        color_grid[i, :, :] = expert_rgba[e_ids]
    axD.imshow(color_grid.swapaxes(0, 1), aspect="auto", interpolation="nearest")
    axD.set_xlabel("Hop")
    axD.set_ylabel("Top paths")
    axD.set_yticks(range(H))
    axD.set_yticklabels([f"H{h}" for h in range(H)])
    axD.set_title("Path composition (expert color)")

    # plt.tight_layout()
    wandb.log({"routing/path_diversity": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)
