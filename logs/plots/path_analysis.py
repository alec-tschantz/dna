from typing import Any, Dict, List, Tuple
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb

from .utils import (
    _collect_top1_indices_all,
    _expert_color_table,
    _mpl_safe,
    _type_base_cmap,
)


def _tok_category(s: str) -> str:
    """Simple token category buckets for diagnostics."""
    if s is None or s == "":
        return "other"
    if s.startswith("Ċ") or s == "\n":
        return "newline"
    if s.startswith("Ġ"):
        s = s[1:] or " "
        if s.strip() == "":
            return "space"
        if s[:1].isalpha():
            return "space+word"
    if s.strip() == "":
        return "space"
    if any(ch.isdigit() for ch in s) and s.strip().isdigit():
        return "digit"
    if all(not ch.isalnum() and not ch.isspace() for ch in s):
        return "punct"
    if s[:1].isalpha() and s[:1].upper() == s[:1] and s[1:].lower() == s[1:]:
        return "Capital"
    if any(ch.isalpha() for ch in s):
        return "word"
    return "other"


def analyze_token_path_specialization(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    max_paths: int = 10,
    min_token_count: int = 5,
):
    """Which tokens/categories best characterize each path (by lift)."""
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
    ids_np = np.asarray(batch["input_ids"])
    mask_np = np.asarray(mask_bt)

    # build (token -> path) assignments
    paths: List[Tuple[int, ...]] = []
    token_ids: List[int] = []
    for b in range(B):
        for t in range(T):
            if not mask_np[b, t]:
                continue
            paths.append(tuple(int(top1[h, b, t]) for h in range(H)))
            token_ids.append(int(ids_np[b, t]))
    if not paths:
        return None

    path_counts = Counter(paths)
    top_paths = [p for p, _ in path_counts.most_common(max_paths)]
    path_index = {p: i for i, p in enumerate(top_paths)}

    token_global = Counter(token_ids)

    per_path_tokens = [Counter() for _ in top_paths]
    for tid, p in zip(token_ids, paths):
        if p in path_index:
            per_path_tokens[path_index[p]][tid] += 1

    categories = [
        "space",
        "newline",
        "digit",
        "punct",
        "Capital",
        "word",
        "space+word",
        "other",
    ]
    cat_index = {c: i for i, c in enumerate(categories)}
    cat_matrix = np.zeros((len(top_paths), len(categories)), dtype=np.float32)
    top_token_lists = []
    counts_out = []

    N_total = len(token_ids)

    for i, p in enumerate(top_paths):
        total_p = sum(per_path_tokens[i].values())
        counts_out.append(total_p)
        token_list = []
        for tid, cnt in per_path_tokens[i].most_common(200):
            if token_global[tid] < min_token_count:
                continue
            p_token_given_path = cnt / max(total_p, 1)
            p_token = token_global[tid] / max(N_total, 1)
            lift = p_token_given_path / max(p_token, 1e-12)
            try:
                s = tok.decode([tid], skip_special_tokens=True) or "·"
            except Exception:
                s = "?"
            token_list.append((s, cnt, float(lift), tid))
            cat = _tok_category(s)
            cat_matrix[i, cat_index[cat]] += cnt

        token_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
        top_token_lists.append(token_list[:10])

    # normalize rows to show mix proportions
    row_sum = cat_matrix.sum(axis=1, keepdims=True) + 1e-9
    cat_norm = cat_matrix / row_sum

    return dict(
        meta=meta,
        H=H,
        top_paths=top_paths,
        path_counts=counts_out,
        categories=categories,
        category_matrix=cat_norm,
        top_tokens=top_token_lists,
    )


def plot_token_path_specialization(spec: Dict[str, Any], step: int):
    """Token-category mix (top-left) + expert-path barcode + compact top-token cards.
    API unchanged. Logs a single matplotlib figure to W&B.
    """
    if spec is None:
        return

    paths = spec["top_paths"]  # list[tuple(expert_id per hop)]
    categories = spec["categories"]  # list[str]
    M, C = len(paths), len(categories)  # #paths, #categories
    H = spec["H"]  # #hops
    counts_per_path = spec["path_counts"]  # absolute token counts per path

    # --- Colors for experts (grouped by module type) ---
    expert_rgba = _expert_color_table(spec["meta"])  # (E,4) RGBA per expert

    # Build barcode grid: rows=hops, cols=paths (so read columns as paths)
    color_grid = np.zeros((H, M, 4), dtype=np.float32)
    for j, path in enumerate(paths):
        e_ids = np.array(path, dtype=int)
        color_grid[:, j, :] = expert_rgba[e_ids]

    # --- Figure/layout ---
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.35,
        hspace=0.38,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    mat = np.asarray(spec["category_matrix"])  # shape (M, C), rows sum to 1

    im = ax1.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    # Row labels include counts so you know the absolute scale for each path
    row_labels = [f"P{i+1}  (n={counts_per_path[i]})" for i in range(M)]
    ax1.set_yticks(range(M))
    ax1.set_yticklabels(row_labels)
    ax1.set_xticks(range(C))
    ax1.set_xticklabels(categories, rotation=28, ha="right")
    ax1.set_title("Token category mix by path (row-normalized)", pad=8)

    # Light vertical grid lines to help scan categories
    for x in range(C):
        ax1.axvline(x - 0.5, color="white", alpha=0.07, linewidth=1)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion within path", rotation=270, labelpad=16)

    # Optional annotations for strong cells
    try:
        if M * C <= 200:  # annotate only when reasonably small
            for i in range(M):
                for j in range(C):
                    v = mat[i, j]
                    if v >= 0.35:
                        ax1.text(
                            j,
                            i,
                            f"{v*100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                            weight="bold",
                        )
    except Exception:
        pass

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(color_grid, aspect="auto", interpolation="nearest")  # shape (H, M, 4)

    # Correct, informative axes: X=paths, Y=hops
    ax2.set_xlabel("Path")
    ax2.set_xticks(range(M))
    ax2.set_xticklabels([f"P{i+1}" for i in range(M)], rotation=0)
    ax2.set_ylabel("Hop")
    ax2.set_yticks(range(H))
    ax2.set_yticklabels([f"H{i}" for i in range(H)])
    ax2.set_title("Path composition (expert per hop)", pad=8)

    # Legend for module types (uses base colormap mid-shade)
    type_names = list(spec["meta"]["id_to_type"])
    legend_handles = []
    for t in type_names:
        cmap = plt.get_cmap(_type_base_cmap(t))
        rgba = np.array(cmap(0.7))
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markerfacecolor=rgba,
                markeredgecolor="none",
                label=t,
            )
        )
    if legend_handles:
        ax2.legend(
            handles=legend_handles,
            title="Module types",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=min(3, len(legend_handles)),
            frameon=False,
        )

    ax3 = fig.add_subplot(gs[:, 1])
    ax3.axis("off")

    # Build two columns of cards for readability
    cards = []
    for i, tokens in enumerate(spec["top_tokens"]):
        # Path signature (expert IDs) for quick reference
        path_sig = "→".join(str(e) for e in paths[i])
        card = f"P{i+1}  (n={counts_per_path[i]})\n" f"path: [{path_sig}]\n"
        for s, cnt, lift, _ in tokens[:6]:
            s_disp = _mpl_safe(s or "·")  # escape $ and newlines
            card += f"  • '{s_disp}'  ×{cnt}   lift {lift:.1f}\n"
        cards.append(card.rstrip())

    # Layout cards in up to 2 columns
    n = len(cards)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    y_top = 0.96
    y_step = 0.9 / max(nrows, 1)

    box_kw = dict(
        boxstyle="round,pad=0.45",
        facecolor="#f8fafc",
        edgecolor="#d9e2ec",
        linewidth=1.0,
        alpha=1.0,
    )

    for idx, text in enumerate(cards):
        col = idx // nrows
        row = idx % nrows
        x = 0.02 + col * 0.48
        y = y_top - row * y_step
        ax3.text(
            x,
            y,
            text,
            transform=ax3.transAxes,
            va="top",
            ha="left",
            # family="monospace",
            fontsize=9,
            bbox=box_kw,
        )

    plt.suptitle(
        f"Token ↔ Path Specialization • step {step}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    wandb.log(
        {"routing/token_path_specialization": wandb.Image(fig)}, step=step, commit=False
    )
    plt.close(fig)


def log_token_path_sankey(spec: Dict[str, Any], step: int):
    """Optional Plotly Sankey: token categories → paths."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    cat = spec["categories"]
    mat = (spec["category_matrix"] * np.array(spec["path_counts"])[:, None]).T
    M = len(spec["top_paths"])
    C = len(cat)

    labels = [f"cat:{c}" for c in cat] + [f"path:P{i+1}" for i in range(M)]
    colors = ["rgba(120,120,120,0.85)"] * C + ["rgba(52,152,219,0.85)"] * M

    src, dst, val = [], [], []
    for i in range(C):
        for j in range(M):
            w = float(mat[i, j])
            if w <= 0:
                continue
            src.append(i)
            dst.append(C + j)
            val.append(w)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(label=labels, color=colors, pad=12, thickness=16),
                link=dict(source=src, target=dst, value=val),
            )
        ]
    )
    fig.update_layout(
        title=f"Token Categories → Paths • step {step}",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    wandb.log({"routing/token_path_sankey": fig}, step=step, commit=False)
