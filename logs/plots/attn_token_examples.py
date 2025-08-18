# logs/plots/attn_examples_cards.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from .utils import (
    has_routing,
    evaluate_for_visuals,        # batch meta (types, labels, counts)
    _forward_batched_for_stats,  # per-hop (B,T,E) stats
    _mpl_safe,
)

# ----------------------------
# Selection reconstruction
# ----------------------------

def _build_selected_from_probs(
    probs_bte: np.ndarray,          # (B,T,E)
    effk_bt: np.ndarray,            # (B,T)
    token_mask_bt: np.ndarray,      # (B,T) bool
) -> np.ndarray:
    B, T, E = probs_bte.shape
    sel = np.zeros((B, T, E), dtype=bool)
    effk_bt = np.clip(effk_bt.astype(int), 0, E)
    for b in range(B):
        valid_t = token_mask_bt[b]
        for t in np.where(valid_t)[0]:
            k = int(effk_bt[b, t])
            if k <= 0:
                continue
            row = probs_bte[b, t]
            idx = np.argpartition(row, -k)[-k:]
            sel[b, t, idx] = True
    return sel

def _approx_kept_from_capacity(
    selected_bte: np.ndarray,       # (B,T,E) bool
    scores_bte: np.ndarray,         # (B,T,E) float (use probs as proxy)
    token_mask_bt: np.ndarray,      # (B,T) bool
    capacity: Optional[int],
) -> np.ndarray:
    B, T, E = selected_bte.shape
    if not capacity or capacity <= 0:
        return selected_bte.copy()
    kept = np.zeros_like(selected_bte, dtype=bool)
    for b in range(B):
        valid_t = token_mask_bt[b]
        for e in range(E):
            pos = np.where(selected_bte[b, :, e] & valid_t)[0]
            if pos.size == 0:
                continue
            if pos.size <= capacity:
                kept[b, pos, e] = True
            else:
                vals = scores_bte[b, pos, e]
                top = np.argpartition(vals, -capacity)[-capacity:]
                kept[b, pos[top], e] = True
    return kept

def _decode_token(tok, tid: int) -> str:
    try:
        s = tok.decode([int(tid)], skip_special_tokens=True) or "·"
    except Exception:
        s = "?"
    s = s.replace("\n", "\\n")
    return s if s.strip() != "" else "·"

def _short(s: str, maxlen: int = 20) -> str:
    s = _mpl_safe(s)
    return s if len(s) <= maxlen else (s[: maxlen - 1] + "…")


# ----------------------------
# Analysis
# ----------------------------

def analyze_attention_examples_single(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    min_token_global: int = 1,   # ignore globally-ultra-rare tokens if >1
    per_module_limit: int = 64,  # store up to N tokens per module
) -> Optional[Dict[str, Any]]:
    """
    Returns flattened cards (across all hops) for Attention experts only:
      each card: {hop, expert_id, n_tokens, n_unique, top_freq:[(text,count,tid), ...]}
    """
    if not has_routing(model):
        return None

    # Batch meta
    batch_stats, _ = evaluate_for_visuals(
        model,
        batch,
        key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
        num_examples=1,
    )
    if batch_stats is None:
        return None

    expert_type_ids = np.asarray(batch_stats["expert_type_ids"])  # (E,)
    id_to_type = list(batch_stats["id_to_type"])
    def _is_attention(eid: int) -> bool:
        return "ttent" in id_to_type[int(expert_type_ids[int(eid)])].lower()

    # Full per-hop stats
    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B = int(ids.shape[0])
    fkeys = jax.random.split(jax.random.fold_in(key, 20211), B)
    hops = _forward_batched_for_stats(
        model, ids, mask, fkeys,
        gumbel_tau=gumbel_tau, router_temp=router_temp, select_temp=select_temp
    )
    if not isinstance(hops, (tuple, list)) or len(hops) == 0:
        return None

    capacity = int(getattr(model, "capacity", 0) or 0)
    token_ids_all = np.asarray(ids)
    global_counts = Counter(int(t) for t in token_ids_all.flatten())

    cards: List[Dict[str, Any]] = []
    H = len(hops)

    for h, hop in enumerate(hops):
        token_mask_bt = np.asarray(hop["token_mask"]).astype(bool)  # (B,T)
        probs_bte = np.asarray(hop["routing_probs"], dtype=np.float32)  # (B,T,E)

        eff = hop.get("eff_topk", None)
        if eff is None:
            try:
                k_guess = int(getattr(model.routers[0], "k", 1))
            except Exception:
                k_guess = 1
            effk_bt = np.full(token_mask_bt.shape, k_guess, dtype=int)
        else:
            effk_bt = np.asarray(eff).astype(int)
            if effk_bt.ndim == 1:
                effk_bt = effk_bt[None, :]

        selected_bte = _build_selected_from_probs(probs_bte, effk_bt, token_mask_bt)
        kept_bte = _approx_kept_from_capacity(selected_bte, probs_bte, token_mask_bt, capacity)

        _, _, E = kept_bte.shape
        for e in range(E):
            if not _is_attention(e):
                continue
            kept = kept_bte[:, :, e] & token_mask_bt  # (B,T)
            b_idx, t_idx = np.where(kept)
            if b_idx.size == 0:
                cards.append({
                    "hop": int(h),
                    "expert_id": int(e),
                    "n_tokens": 0,
                    "n_unique": 0,
                    "top_freq": [],
                })
                continue

            tids = token_ids_all[b_idx, t_idx].astype(int)
            counts = Counter(int(t) for t in tids)
            if min_token_global > 1:
                counts = Counter({
                    tid: c for tid, c in counts.items()
                    if global_counts[tid] >= min_token_global
                })

            top_pairs = counts.most_common(per_module_limit)
            top_freq = [(_decode_token(tok, tid), int(c), int(tid)) for tid, c in top_pairs]

            cards.append({
                "hop": int(h),
                "expert_id": int(e),
                "n_tokens": int(sum(counts.values())),
                "n_unique": int(len(counts)),
                "top_freq": top_freq,
            })

    # Sort: by hop, then by n_tokens desc
    cards.sort(key=lambda d: (d["hop"], -d["n_tokens"], d["expert_id"]))
    return {"H": H, "cards": cards}


# ----------------------------
# Single-page card renderer
# ----------------------------

def _suggest_layout(n_cards: int) -> Tuple[int, int]:
    # (ncols, max_lines_per_card)
    # More cards → more columns and fewer lines per card.
    if n_cards <= 8:
        return 2, 24
    if n_cards <= 15:
        return 3, 20
    if n_cards <= 24:
        return 4, 16
    if n_cards <= 32:
        return 5, 14
    return 6, 12  # very dense

def log_attention_token_examples(
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
    min_token_global: int = 1,
    per_module_limit: int = 96,    # collect plenty; UI trims by lines
    # plotting knobs
    wandb_key: str = "routing/attn_cards_single",
):
    """
    Single-page, cards-only plot across ALL hops.
    Each card = one Attention expert (shows hop, expert id, total/unique counts, and many 'top by freq' tokens).
    """
    spec = analyze_attention_examples_single(
        model, batch, tok,
        key=key,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
        min_token_global=min_token_global,
        per_module_limit=per_module_limit,
    )
    if spec is None:
        return

    cards = spec["cards"]
    if len(cards) == 0:
        return

    ncols, max_lines = _suggest_layout(len(cards))
    rows = int(np.ceil(len(cards) / ncols))

    # Figure size scales with rows/cols
    fig_h = max(7.5, 2.0 + rows * 2.4)
    fig_w = 18.0 if ncols <= 4 else 20.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.86]); ax.axis("off")

    header = f"Attention Modules • total cards {len(cards)} • hops={spec['H']}"
    ax.text(0.0, 1.02, header, transform=ax.transAxes,
            fontsize=13, weight="bold", va="bottom")

    col_w = 1.0 / ncols
    row_h = 1.0 / max(rows, 1)
    box = dict(boxstyle="round,pad=0.45", facecolor="#f8fafc",
               edgecolor="#d9e2ec", linewidth=1.0)

    def fmt_top_freq(pairs, k: int) -> str:
        lines = ["Top by freq:"]
        for s, c, _tid in pairs[:k]:
            lines.append(f"  • '{_short(s)}' ×{c}")
        return "\n".join(lines)

    for i, info in enumerate(cards):
        col = i % ncols
        row = i // ncols
        x = col * col_w + 0.01
        y = 1.0 - (row + 1) * row_h + 0.01

        head = f"H{info['hop']} • A{info['expert_id']} • tokens {info['n_tokens']:,} (unique {info['n_unique']:,})"
        body = fmt_top_freq(info["top_freq"], max_lines)

        ax.text(
            x, y + row_h - 0.02, head + "\n" + body,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=box,
        )

    fig.suptitle("Tokens Routed to Attention Experts (single page)", fontsize=14, weight="bold", y=0.995)
    wandb.log({wandb_key: wandb.Image(fig)}, step=step, commit=False)
    plt.close(fig)
