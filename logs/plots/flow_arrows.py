from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import wandb

from .utils import (
    has_routing,
    evaluate_for_visuals,
    _mpl_safe,
    _expert_color_table,
    _type_base_cmap,
)


def log_token_expert_flow_arrows(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    max_tokens: int = 64,
    show_topk: int = 2,
    max_experts: int = 32,
    show_other_bucket: bool = True,
    eps: float = 1e-8,
):
    if not has_routing(model):
        return

    # Collect routing/example stats
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

    # Shapes
    H = int(np.asarray(example_stats["top1_expert"]).shape[0])
    ids_all = np.asarray(example_stats["ids"])[0]
    mask_all = np.asarray(example_stats["mask"])[0].astype(bool)
    probs_all = np.asarray(example_stats["probs"])[:, 0, :, :]  # (H, T, E)
    E = int(probs_all.shape[-1])

    # Visible window (keep order)
    valid_pos = np.where(mask_all)[0]
    if valid_pos.size == 0:
        return
    Tvis = int(min(valid_pos.size, max_tokens))
    vis_idx = valid_pos[:Tvis]

    # Slice tensors to visible window
    ids = ids_all[vis_idx]
    P = probs_all[:, vis_idx, :]  # (H, Tvis, E)

    # Token text (ASCII-safe)
    VISIBLE_SPACE = "·"

    def _ids_to_tokens(ids_1d: np.ndarray) -> list[str]:
        out = []
        for tid in ids_1d.tolist():
            try:
                try:
                    s = tok.decode(
                        [int(tid)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    s = tok.decode([int(tid)], skip_special_tokens=False)
            except Exception:
                s = "?"
            s = (s or "").replace("Ġ ", VISIBLE_SPACE)
            s = s.replace("\n", "\\n")
            if s.strip() == "":
                s = VISIBLE_SPACE
            out.append(s if len(s) <= 4 else (s[:3] + "…"))
        return out

    tok_disp = _ids_to_tokens(ids)

    # Preview text (escaped newlines for a single-line title)
    try:
        try:
            preview = tok.decode(
                ids.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            preview = tok.decode(ids.tolist(), skip_special_tokens=False)
    except Exception:
        preview = ""
    preview = preview.replace("\n", "\\n")
    preview = _mpl_safe(preview)

    if len(preview) > 160:
        preview = preview[:157] + "…"

    # Normalize probs per (hop, token)
    denom = P.sum(axis=-1, keepdims=True)
    Pn = np.where(denom > eps, P / (denom + eps), 0.0)

    # Choose a fixed global subset of experts to display
    mass_per_e = Pn.sum(axis=(0, 1))
    order = np.argsort(-mass_per_e)
    keep = order[: min(max_experts, E)].tolist()
    keep_set = set(int(e) for e in keep)
    OTHER = -1
    include_other = show_other_bucket and (
        len(keep_set) < E and float(mass_per_e.sum()) > 0.0
    )

    n_nodes = len(keep) + (1 if include_other else 0)
    if n_nodes == 0:
        return
    ex_x = (
        np.array([max(0, (Tvis - 1) * 0.5)], dtype=float)
        if n_nodes == 1
        else np.linspace(0, max(Tvis - 1, 1), num=n_nodes, dtype=float)
    )
    # keep order is deterministic (descending mass), then optional OTHER
    e_list = keep + ([OTHER] if include_other else [])
    e_to_x = {int(e): float(x) for e, x in zip(e_list, ex_x)}

    # Colors / labels
    expert_rgba = _expert_color_table(batch_stats)  # (E,4)
    type_names = list(batch_stats["id_to_type"])
    expert_type_ids = np.asarray(batch_stats["expert_type_ids"])

    def _etype_initial(eid: int) -> str:
        if eid == OTHER:
            return "O"
        tname = type_names[int(expert_type_ids[eid])]
        return (
            "A"
            if "ttent" in tname.lower()
            else ("F" if ("feed" in tname.lower() or "mlp" in tname.lower()) else "I")
        )

    def _expert_label(eid: int) -> str:
        return "Other" if eid == OTHER else f"{_etype_initial(eid)}{int(eid)}"

    def _expert_color(eid: int) -> np.ndarray:
        if eid == OTHER:
            return np.array([0.62, 0.62, 0.68, 1.0], dtype=np.float32)
        return expert_rgba[int(eid)]

    def _line_style(eid: int, p: float) -> tuple[np.ndarray, float]:
        base = _expert_color(eid).copy()
        p = float(np.clip(p, 0.0, 1.0))
        base[:3] = base[:3] * (0.35 + 0.65 * p) + (1.0 - p) * 0.10
        base[3] = 0.20 + 0.75 * p
        lw = 0.8 + 3.0 * p
        return base, lw

    # Greedy predictions aligned to same positions
    try:
        logits, _ = model(
            jnp.asarray(ids_all[None, :]),
            key=key,
            inference=True,
            mask=jnp.asarray(mask_all[None, :], dtype=jnp.float32),
            gumbel_tau=float(gumbel_tau),
            router_temp=float(router_temp),
            select_temp=float(select_temp),
            # return_stats=False,
        )
        pred_full = np.asarray(jnp.argmax(logits[0], axis=-1))
    except Exception:
        pred_full = np.roll(ids_all, -1)
        if pred_full.size >= 2:
            pred_full[-1] = pred_full[-2]
    pred_ids = pred_full[vis_idx]
    pred_disp = _ids_to_tokens(pred_ids)

    # Layout
    row_gap = 1.55
    token_h = 0.74
    expert_h = 0.70

    def y_token_row(i: int) -> float:
        return (2 * i) * row_gap

    def y_expert_row(h: int) -> float:
        return (2 * h + 1) * row_gap

    y_pred = y_token_row(H) + 1.0
    fig_w = max(15.0, 0.34 * Tvis + 10.0)
    fig_h = max(7.5, (2 * H + 1) * row_gap + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    ax.set_xlim(-0.8, Tvis - 0.2)
    ax.set_ylim(-0.8, y_pred + 1.3)
    ax.axis("off")

    # Background bands per expert row
    for h in range(H):
        y0 = y_expert_row(h) - 0.55
        ax.add_patch(
            Rectangle(
                (-0.9, y0),
                Tvis + 1.8,
                1.9,
                facecolor="#fafafa",
                edgecolor="none",
                zorder=0,
            )
        )

    # Token row helper
    def _draw_token_row(
        y: float, labels: list[str], face="#e7eef6", edge="#c9d5e3", txt="#0f172a"
    ):
        for j in range(Tvis):
            ax.add_patch(
                Rectangle(
                    (j - 0.42, y - token_h / 2),
                    0.84,
                    token_h,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.8,
                    zorder=7,
                )
            )
            ax.text(
                j,
                y,
                labels[j],
                ha="center",
                va="center",
                fontsize=8,
                color=txt,
                zorder=9,
            )

    # Top token row
    _draw_token_row(y_token_row(0), tok_disp)

    # Expert nodes for each hop (fixed x)
    for h in range(H):
        yE = y_expert_row(h)
        for e in e_list:
            col = _expert_color(e)
            ax.add_patch(
                Rectangle(
                    (e_to_x[e] - 0.50, yE - expert_h / 2),
                    1.00,
                    expert_h,
                    facecolor=col,
                    edgecolor="#0f172a",
                    linewidth=0.45,
                    zorder=5,
                )
            )
            ax.text(
                e_to_x[e],
                yE,
                _expert_label(e),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold",
                zorder=6,
            )

    # Curved connectors token(h) -> expert(h)
    def _curve(x0, y0, x1, y1, col, lw):
        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0 + 0.30
        path = MplPath(
            [(x0, y0), (xm, ym), (x1, y1)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        )
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=col,
                linewidth=lw,
                capstyle="round",
                zorder=4,
            )
        )

    # Draw only top-k edges per token per hop
    k = max(1, int(show_topk))
    for h in range(H):
        y0 = y_token_row(h) + 0.24
        y1 = y_expert_row(h) - 0.24
        for j in range(Tvis):
            pe = Pn[h, j]
            if pe.sum() <= eps:
                continue
            kk = min(k, int(np.count_nonzero(pe > eps)))
            idx = np.argpartition(pe, -kk)[-kk:]
            idx = idx[np.argsort(-pe[idx])]
            for e in idx:
                eid = int(e)
                target = eid if eid in keep_set else OTHER
                col, lw = _line_style(target, float(pe[eid]))
                _curve(float(j), y0, float(e_to_x[target]), y1, col, lw)

        # Next token row (same labels) to make T(h) -> E(h) -> T(h+1)
        _draw_token_row(y_token_row(h + 1), tok_disp, face="#eef2f7", edge="#cdd7e1")

    # Bottom predictions
    for j in range(Tvis):
        ax.add_patch(
            Rectangle(
                (j - 0.42, y_pred - token_h / 2),
                0.84,
                token_h,
                facecolor="#d1fae5",
                edgecolor="#10b981",
                linewidth=0.9,
                zorder=8,
            )
        )
        ax.text(
            j,
            y_pred,
            pred_disp[j],
            ha="center",
            va="center",
            fontsize=8,
            color="#065f46",
            zorder=9,
        )

    # Titles / legend
    ax.text(
        0.0,
        y_pred + 0.78,
        f"Token → Expert Flow per Hop • step {step}",
        fontsize=13,
        weight="bold",
        ha="left",
        va="bottom",
        transform=ax.transData,
    )
    ax.text(
        0.0,
        y_pred + 0.41,
        f"input preview: {preview}",
        fontsize=9,
        color="#374151",
        ha="left",
        va="bottom",
        transform=ax.transData,
    )
    ax.text(
        -0.6,
        y_pred + 0.10,
        "Predicted next tokens (greedy):",
        fontsize=9,
        color="#065f46",
        ha="left",
        va="bottom",
        transform=ax.transData,
    )

    try:
        handles = []
        for tname in list(batch_stats["id_to_type"]):
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
        if include_other:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markerfacecolor=(0.62, 0.62, 0.68),
                    markeredgecolor="none",
                    label="Other",
                )
            )
        if handles:
            ax.legend(
                handles=handles,
                title="Expert types",
                loc="upper right",
                bbox_to_anchor=(1.0, 1.02),
                frameon=False,
                ncol=min(3, len(handles)),
                fontsize=8,
                title_fontsize=9,
            )
    except Exception:
        pass

    wandb.log({"routing/token_expert_flow_arrows": wandb.Image(fig), "step": step})
    plt.close(fig)