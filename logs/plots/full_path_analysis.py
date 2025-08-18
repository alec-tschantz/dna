# logs/plots/full_path_analysis.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from .utils import _collect_top1_indices_all, _mpl_safe


# ---------- analysis ----------

def analyze_full_path_distribution(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    max_paths: int = 20,          # keep this many most-common paths
    min_token_count: int = 4,     # ignore globally-ultra-rare tokens
    top_by_lift: int = 40,        # compute more than we show (UI trims)
    top_by_freq: int = 40,
) -> Optional[Dict[str, Any]]:
    """
    Build a concentrated view of the path distribution and per-path token lists.

    Returns a dict with:
      - H: int (#hops)
      - paths: List[Tuple[int,...]] top paths (hop-wise expert ids)
      - counts: List[int], share: List[float]
      - N_total: int (valid tokens)
      - entropy: float; eff_paths: float (exp(entropy))
      - coverage_topM: float (share of kept paths)
      - tokens: List[Dict[str, List[Tuple[str,int,float,float,int]]]]
        each contains "top_lift" and "top_freq" lists of (text, count, lift, p|path, tid)
    """
    top1, mask_bt, _meta = _collect_top1_indices_all(
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
    valid_np = np.asarray(mask_bt).astype(bool)

    # (token -> path) assignments
    paths: List[Tuple[int, ...]] = []
    token_ids: List[int] = []
    for b in range(B):
        for t in range(T):
            if not valid_np[b, t]:
                continue
            paths.append(tuple(int(top1[h, b, t]) for h in range(H)))
            token_ids.append(int(ids_np[b, t]))
    if not paths:
        return None

    path_counts = Counter(paths)
    total_valid = len(paths)

    # entropy & effective #paths
    probs_all = np.array([c / total_valid for c in path_counts.values()], dtype=np.float64)
    entropy = float(-(probs_all * np.log(probs_all + 1e-12)).sum())
    eff_paths = float(np.exp(entropy))

    # keep top M
    kept_paths = [p for p, _ in path_counts.most_common(max_paths)]
    kept_index = {p: i for i, p in enumerate(kept_paths)}
    kept_counts = [path_counts[p] for p in kept_paths]
    kept_share = [c / total_valid for c in kept_counts]
    coverage_topM = float(sum(kept_share))

    # token stats (global & per-path)
    global_token_counts = Counter(token_ids)
    per_path_token_counts = [Counter() for _ in kept_paths]
    for tid, p in zip(token_ids, paths):
        if p in kept_index:
            per_path_token_counts[kept_index[p]][tid] += 1

    def _decode(tid: int) -> str:
        try:
            s = tok.decode([tid], skip_special_tokens=True) or "·"
        except Exception:
            s = "?"
        return s

    # per-path token lists
    per_path_token_lists: List[Dict[str, List[Tuple[str, int, float, float, int]]]] = []
    for i in range(len(kept_paths)):
        cnts = per_path_token_counts[i]
        total_p = max(sum(cnts.values()), 1)
        items: List[Tuple[int, int, float, float]] = []  # (tid, c_in_path, lift, p_in_path)
        for tid, c_in_path in cnts.items():
            if global_token_counts[tid] < min_token_count:
                continue
            p_in_path = c_in_path / total_p
            p_global = global_token_counts[tid] / total_valid
            lift = p_in_path / max(p_global, 1e-12)
            items.append((tid, c_in_path, float(lift), float(p_in_path)))

        # sort, then trim later when rendering
        top_lift = sorted(items, key=lambda x: (x[2], x[1]), reverse=True)[:top_by_lift]
        top_freq = sorted(items, key=lambda x: (x[1], x[2]), reverse=True)[:top_by_freq]

        def _pack(raw):
            out = []
            for tid, c, lift, pin in raw:
                out.append((_decode(tid), int(c), float(lift), float(pin), int(tid)))
            return out

        per_path_token_lists.append({"top_lift": _pack(top_lift), "top_freq": _pack(top_freq)})

    return dict(
        H=H,
        paths=kept_paths,
        counts=kept_counts,
        share=kept_share,
        N_total=total_valid,
        entropy=entropy,
        eff_paths=eff_paths,
        coverage_topM=coverage_topM,
        tokens=per_path_token_lists,
    )


# ---------- card-only plotting (paged) ----------

def _short(s: str, maxlen: int = 18) -> str:
    s = _mpl_safe((s or "·").replace("\n", "\\n"))
    return s if len(s) <= maxlen else (s[: maxlen - 1] + "…")

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def _suggest_layout(n_paths: int) -> Tuple[int, int]:
    """
    Return (ncols, max_paths_per_page). We bias toward readability.
    """
    if n_paths <= 8:
        return 2, 8
    if n_paths <= 18:
        return 3, 18
    if n_paths <= 24:
        return 4, 24
    # many paths → 4 columns, paginate every 24
    return 4, 24

def _lines_per_card(ncols: int) -> Tuple[int, int]:
    """
    Choose how many lines to show for (top_lift, top_freq) per card, based on columns.
    """
    if ncols == 2:
        return 16, 12
    if ncols == 3:
        return 12, 10
    return 9, 7  # ncols >= 4

def _render_cards_page(
    *,
    paths: List[Tuple[int, ...]],
    counts: List[int],
    shares: List[float],
    lists: List[Dict[str, List[Tuple[str, int, float, float, int]]]],
    H: int,
    N_total: int,
    entropy: float,
    eff_paths: float,
    coverage: float,
    step: int,
    page_num: int,
    ncols: int,
    k_lift_show: int,
    k_freq_show: int,
    wandb_key: str,
):
    n = len(paths)
    rows = int(np.ceil(n / ncols))
    fig_h = max(8.0, 2.0 + rows * 2.4)    # taller with more rows
    fig_w = 18.0 if ncols <= 3 else 20.0  # a bit wider for 4 cols

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.86])  # single canvas
    ax.axis("off")

    # Header
    header = (
        f"Shown {n} paths • coverage {coverage*100:.1f}% • hops {H} • tokens {N_total:,} • "
        f"entropy {entropy:.3f} • eff#paths {eff_paths:.1f} • page {page_num}"
    )
    ax.text(0.0, 1.02, header, transform=ax.transAxes, fontsize=13, weight="bold", va="bottom")

    # Grid placement
    col_w = 1.0 / ncols
    row_h = 1.0 / max(rows, 1)
    box = dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#d9e2ec", linewidth=1.0)

    def fmt_block(title: str, items, k: int) -> str:
        lines = [title + ":"]
        for s, c, lift, pin, _tid in items[:k]:
            lines.append(f"  • '{_short(s)}' ×{c}   lift {lift:.1f}   p|path {pin:.0%}")
        return "\n".join(lines)

    for i in range(n):
        col = i % ncols
        row = i // ncols
        x = col * col_w + 0.01
        y = 1.0 - (row + 1) * row_h + 0.01

        sig = "→".join(map(str, paths[i]))
        head = f"P{i+1} • {counts[i]} ({shares[i]*100:.1f}%)\npath: [{sig}]"
        body = (
            fmt_block("Top by lift", lists[i]["top_lift"], k_lift_show) + "\n" +
            fmt_block("Most frequent", lists[i]["top_freq"], k_freq_show)
        )
        ax.text(
            x, y + row_h - 0.02,
            head + "\n" + body,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            family=None,  # default font, more compact than monospace
            bbox=box,
        )

    wandb.log({f"{wandb_key}/page_{page_num:02d}": wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)

def plot_full_path_cards_paged(
    spec: Dict[str, Any],
    step: int,
    *,
    wandb_key: str = "routing/full_path_cards",
    max_paths_show: Optional[int] = None,
):
    """
    Card-only visualization. Paginates automatically to keep cards readable.
    """
    paths = list(spec["paths"])
    counts = list(spec["counts"])
    shares = list(spec["share"])
    lists = list(spec["tokens"])

    H = int(spec["H"])
    N_total = int(spec["N_total"])
    entropy = float(spec["entropy"])
    eff_paths = float(spec["eff_paths"])

    # Optionally cap total paths to show
    if max_paths_show is not None:
        paths, counts, shares, lists = paths[:max_paths_show], counts[:max_paths_show], shares[:max_paths_show], lists[:max_paths_show]

    # Layout & paging
    ncols, max_per_page = _suggest_layout(len(paths))
    k_lift_show, k_freq_show = _lines_per_card(ncols)

    pages = list(_chunk(list(zip(paths, counts, shares, lists)), max_per_page))
    for pnum, page in enumerate(pages, start=1):
        p_paths, p_counts, p_shares, p_lists = list(zip(*page))
        coverage = float(sum(p_shares))
        _render_cards_page(
            paths=list(p_paths),
            counts=list(p_counts),
            shares=list(p_shares),
            lists=list(p_lists),
            H=H,
            N_total=N_total,
            entropy=entropy,
            eff_paths=eff_paths,
            coverage=coverage,
            step=step,
            page_num=pnum,
            ncols=ncols,
            k_lift_show=k_lift_show,
            k_freq_show=k_freq_show,
            wandb_key=wandb_key,
        )


# ---------- public entry ----------

def log_full_path_analysis(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    step: int,
    # analysis knobs
    max_paths: int = 20,
    min_token_count: int = 4,
    top_by_lift: int = 40,
    top_by_freq: int = 40,
    # plotting knobs
    max_paths_show: Optional[int] = None,
    wandb_key: str = "routing/full_path_cards",
):
    """
    Analyze then log a card-only, paged path report (no heatmaps/bars).
    """
    spec = analyze_full_path_distribution(
        model, batch, tok,
        key=key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
        max_paths=max_paths,
        min_token_count=min_token_count,
        top_by_lift=top_by_lift,
        top_by_freq=top_by_freq,
    )
    if spec is None:
        return
    plot_full_path_cards_paged(
        spec, step,
        wandb_key=wandb_key,
        max_paths_show=max_paths_show,
    )
