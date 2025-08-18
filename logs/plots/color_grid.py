from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import wandb

from .utils import (
    has_routing,
    evaluate_for_visuals,
    _mpl_safe,
    _expert_color_table,
    _type_base_cmap,
)


def log_token_expert_color_grid(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    max_tokens: int = 48,
    alpha_by_conf: bool = True,
):
    if not has_routing(model):
        return

    batch_stats, example_stats = evaluate_for_visuals(
        model,
        batch,
        key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
        num_examples=1,
    )
    if batch_stats is None or example_stats is None:
        return

    H = int(np.asarray(example_stats["top1_expert"]).shape[0])
    ids_all = np.asarray(example_stats["ids"])[0]
    mask_all = np.asarray(example_stats["mask"])[0].astype(bool)
    valid_pos = np.where(mask_all)[0]
    if valid_pos.size == 0:
        return
    Tvis = int(min(valid_pos.size, max_tokens))
    vis_idx = valid_pos[:Tvis]

    ids = ids_all[vis_idx]
    top1_idx = np.asarray(example_stats["top1_expert"])[:, 0, vis_idx]  # (H, Tvis)
    top1_prob = np.asarray(example_stats["top1_prob"])[
        :, 0, vis_idx
    ]  # (H, Tvis)  <-- FIX

    def _short_token(tid: int) -> str:
        try:
            s = tok.decode([int(tid)], skip_special_tokens=False)
        except Exception:
            s = "?"
        s = (s or "").replace("Ġ ", "·").replace("\n", "\\n")
        if s.strip() == "":
            s = "·"
        return s[:3] + ("…" if len(s) > 3 else "")

    token_labels = [_short_token(int(t)) for t in ids]

    try:
        preview = tok.decode(ids.tolist(), skip_special_tokens=False)
    except Exception:
        preview = ""
    preview = _mpl_safe(preview.replace("\n", "\\n"))
    if len(preview) > 140:
        preview = preview[:137] + "…"

    expert_rgba = _expert_color_table(batch_stats)
    type_names = list(batch_stats["id_to_type"])
    expert_type_ids = np.asarray(batch_stats["expert_type_ids"])

    def _etype_initial(eid: int) -> str:
        tname = type_names[int(expert_type_ids[eid])]
        return (
            "A"
            if "ttent" in tname.lower()
            else ("F" if ("feed" in tname.lower() or "mlp" in tname.lower()) else "I")
        )

    fig_w = max(12.0, 0.62 * Tvis)
    fig_h = max(5.0, 1.22 * H + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    ax.set_xlim(-0.5, Tvis - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.axis("off")

    for h in range(H):
        for j in range(Tvis):
            eid = int(top1_idx[h, j])
            prob = float(np.clip(top1_prob[h, j], 0.0, 1.0))
            col = expert_rgba[eid].copy()
            col[3] = 0.35 + 0.6 * prob if alpha_by_conf else 0.9

            rect = FancyBboxPatch(
                (j - 0.48, h - 0.48),
                0.96,
                0.96,
                boxstyle="round,pad=0.06,rounding_size=0.06",
                linewidth=0.7,
                edgecolor="white",
                facecolor=col,
                zorder=1,
            )
            ax.add_patch(rect)

            ax.text(  # module id
                j,
                h + 0.12,
                f"{_etype_initial(eid)}{eid}",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="white",
                zorder=2,
            )
            ax.text(  # token (tiny caption)
                j,
                h - 0.20,
                token_labels[j],
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                alpha=0.95,
                zorder=2,
            )

    ax.set_yticks(range(H))
    ax.set_yticklabels([f"Hop {h}" for h in range(H)], fontsize=9)

    fig.suptitle(
        f"Token → Module (Top-1, colored by module type) • step {step}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.94,
        f"preview: {preview}",
        ha="center",
        va="center",
        fontsize=9,
        color="#444",
    )

    try:
        handles = []
        for tname in type_names:
            cmap = plt.get_cmap(_type_base_cmap(tname))
            rgba = np.array(cmap(0.75))
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markerfacecolor=rgba,
                    markeredgecolor="none",
                    label=tname,
                )
            )
        if handles:
            ax.legend(
                handles=handles,
                title="Module types",
                loc="upper right",
                bbox_to_anchor=(1.0, 1.05),
                frameon=False,
                fontsize=8,
                title_fontsize=9,
            )
    except Exception:
        pass

    wandb.log(
        {"routing/token_expert_color_grid": wandb.Image(fig)}, step=step, commit=False
    )
    plt.close(fig)
