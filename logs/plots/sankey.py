from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb

from .utils import (
    has_routing,
    _collect_top1_indices_all,
    _expert_color_table,
    _rgba_to_plotly,
)


def log_routing_sankey_if_available(
    model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    top_paths_per_hop: int = 18,
    min_frac: float = 0.01,
    by_type: bool = False,
):

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
    E = int(meta["expert_type_ids"].shape[0])
    type_names = list(meta["id_to_type"])
    type_ids = np.asarray(meta["expert_type_ids"])
    labels_all = list(meta["expert_labels"])
    expert_rgba = _expert_color_table(meta)

    # transitions per adjacent hop
    flows = []
    mask_flat = np.asarray(mask_bt).reshape(-1)
    for h in range(H - 1):
        a = np.asarray(top1[h]).reshape(-1)[mask_flat]
        b = np.asarray(top1[h + 1]).reshape(-1)[mask_flat]
        pair = (a * E + b).astype(np.int64)
        counts = np.bincount(pair, minlength=E * E).reshape(E, E)
        flows.append(counts)

    initials = {
        t: (
            "A"
            if "ttent" in t.lower()
            else "F" if ("feed" in t.lower() or "mlp" in t.lower()) else "I"
        )
        for t in type_names
    }

    # nodes + links
    node_id = {}
    node_labels, node_colors = [], []

    def _ensure_node(h, e):
        key_ = (h, int(e))
        if key_ in node_id:
            return node_id[key_]
        base = (
            labels_all[int(e)]
            or f"{initials[type_names[int(type_ids[int(e)])]]}{int(e)}"
        )
        lbl = f"H{h} • {base}"
        node_id[key_] = len(node_labels)
        node_labels.append(lbl)
        node_colors.append(_rgba_to_plotly(expert_rgba[int(e)], alpha=0.85))
        return node_id[key_]

    link_src, link_tgt, link_val, link_col = [], [], [], []
    for h, mat in enumerate(flows):
        total = float(mat.sum())
        if total <= 0:
            continue
        flat = mat.flatten()
        k = min(top_paths_per_hop, flat.size)
        thr = max(int(min_frac * total), 1)

        idx_thr = np.where(flat >= thr)[0]
        idx_top = (
            np.argpartition(flat, -k)[-k:] if k < flat.size else np.arange(flat.size)
        )
        keep = np.unique(np.concatenate([idx_thr, idx_top]))
        keep = keep[np.argsort(-flat[keep])]

        for idx in keep:
            v = int(flat[idx])
            if v <= 0:
                continue
            e0 = idx // E
            e1 = idx % E
            s = _ensure_node(h, e0)
            t = _ensure_node(h + 1, e1)
            link_src.append(s)
            link_tgt.append(t)
            link_val.append(v)
            link_col.append(_rgba_to_plotly(expert_rgba[int(e0)], alpha=0.35))

    total_links = int(np.sum(link_val)) if link_val else 0
    total_tokens = int(np.sum([m.sum() for m in flows])) if flows else 0
    coverage = (total_links / max(total_tokens, 1)) if total_tokens else 0.0

    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".0f",
                node=dict(
                    pad=12,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=link_src, target=link_tgt, value=link_val, color=link_col
                ),
            )
        ]
    )
    fig_sankey.update_layout(
        title=(
            f"Routing Token Paths • step {step} • nodes={len(node_labels)} • "
            f"links={len(link_val)} • cov={coverage:.2f}"
        ),
        font=dict(size=12),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    wandb.log({"routing/sankey": fig_sankey}, step=step, commit=False)

    if flows:
        ncols = min(3, max(1, len(flows)))
        nrows = int(np.ceil(len(flows) / ncols))
        fig_hm, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for i, mat in enumerate(flows):
            ax = axes[i]
            im = ax.imshow(mat, aspect="auto", cmap="magma")
            ax.set_title(f"Hop {i} → {i+1} transitions")
            ax.set_xlabel("Expert @ hop+1")
            ax.set_ylabel("Expert @ hop")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for j in range(len(flows), len(axes)):
            axes[j].axis("off")
        wandb.log({"routing/transition_heatmap": wandb.Image(fig_hm)}, step=step, commit=False)
        plt.close(fig_hm)

    return fig_sankey