from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb

from .utils import (
    has_routing,
    evaluate_for_visuals,
    _mpl_safe,
    _expert_color_table,
    _type_base_cmap,
)


def plot_heatmap(batch_stats: Dict[str, Any], step: int) -> None:
    """Expert importance heatmap (hop × expert)."""
    if batch_stats is None:
        return

    imp = jax.device_get(batch_stats["importance_matrix"])
    labels = batch_stats.get("expert_labels", [f"E{i}" for i in range(imp.shape[1])])

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(imp.T, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Routing Hop")
    ax.set_ylabel("Expert Module")
    ax.set_title(f"Expert Importance Heatmap • Step {step}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Importance", rotation=270, labelpad=20)

    ax.set_xticks(range(imp.shape[0]))
    ax.set_xticklabels([f"Hop {i}" for i in range(imp.shape[0])])
    ax.set_yticks(range(imp.shape[1]))
    ax.set_yticklabels(labels)

    wandb.log({"routing/heatmap": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)


def plot_token_flow_rich(
    example_stats: Dict[str, Any],
    batch_stats: Dict[str, Any],
    tok,
    *,
    step: int,
    max_tokens: int = 96,
    title: str = "Token Routing Flow (top-1)",
):
    """Per-token top-1 expert per hop, colored by expert (with opacity by confidence)."""
    if example_stats is None or batch_stats is None:
        return

    H, S, T = np.asarray(example_stats["top1_expert"]).shape
    ids = np.asarray(example_stats["ids"])
    mask = np.asarray(example_stats["mask"]).astype(bool)
    if S == 0:
        return

    valid_pos = np.where(mask[0])[0]
    if valid_pos.size == 0:
        return
    Tvis = int(min(valid_pos.size, max_tokens))
    vis_idx = valid_pos[:Tvis]
    ids_vis = ids[0, vis_idx]

    try:
        preview = tok.decode(ids_vis.tolist(), skip_special_tokens=True)
    except Exception:
        preview = "[decode error]"
    preview = _mpl_safe(preview)
    if len(preview) > 120:
        preview = preview[:117] + "..."

    expert_rgba = _expert_color_table(batch_stats)
    top1_idx = np.asarray(example_stats["top1_expert"])
    top1_prob = np.asarray(example_stats["top1_prob"])
    top1_type = np.asarray(example_stats["top1_type_id"])
    type_names = list(batch_stats["id_to_type"])
    initials = {
        t: (
            "A"
            if "ttent" in t.lower()
            else "F" if ("feed" in t.lower() or "mlp" in t.lower()) else "I"
        )
        for t in type_names
    }

    grid = np.zeros((H, Tvis, 4), dtype=np.float32)
    labels = np.empty((H, Tvis), dtype=object)

    for h in range(H):
        e_ids = top1_idx[h, 0, vis_idx]
        probs = np.clip(top1_prob[h, 0, vis_idx], 0.0, 1.0)
        colors = expert_rgba[e_ids].copy()
        colors[:, 3] = 0.25 + 0.75 * probs
        grid[h, :, :] = colors

        t_ids = top1_type[h, 0, vis_idx]
        for c in range(Tvis):
            tname = type_names[int(t_ids[c])]
            labels[h, c] = f"{initials[tname]}{int(e_ids[c])}"

    fig_h = max(5.5, 0.5 * H + 2.5)
    fig_w = max(14.0, 0.12 * Tvis + 8.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(grid, aspect="auto", interpolation="nearest")

    if Tvis <= 120:
        for h in range(H):
            for c in range(Tvis):
                ax.text(
                    c,
                    h,
                    labels[h, c],
                    ha="center",
                    va="center",
                    fontsize=7 if Tvis > 80 else 8,
                    color="white",
                )

    ax.set_yticks(np.arange(H))
    ax.set_yticklabels([f"Hop {h}" for h in range(H)])
    ax.set_xticks([])
    ax.set_title(f"{title} • step {step}\npreview: {preview}", pad=10)

    wandb.log({"routing/token_flow_rich": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)


def log_routing_visuals(
    model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    tok,
    step: int,
    num_examples: int = 1,
    max_tokens_grid: int = 96,
):
    """Run + log heatmap and token-flow if routing is present."""
    if not has_routing(model):
        return None, None

    batch_stats, example_stats = evaluate_for_visuals(
        model,
        batch,
        key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
        num_examples=num_examples,
    )

    if batch_stats is not None:
        plot_heatmap(batch_stats, step)
        # plot_token_flow_rich(
        #     example_stats, batch_stats, tok, step=step, max_tokens=max_tokens_grid
        # )

    return batch_stats, example_stats