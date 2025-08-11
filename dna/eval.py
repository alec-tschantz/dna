"""Evaluation utilities for DNA model.

Key improvements
----------------
- Robust expert metadata from `model.groups` (no hardcoded Attn/FFN/Id).
- Heatmap shows average *importance* per hop/expert with real expert labels.
- Token-type grid per example (rows) × per hop (cols) with:
  • Color = top-1 expert *type* per token
  • Brightness overlay = top-1 probability
  • Left-side caption shows decoded preview text (the "prompt") for each example
- Cleaner spacing & typography so plots are readable in W&B.
- Random, non-repeating example selection per eval call.
- Extra routing metrics (KL-to-uniform, top1 share, capacity saturation, per-type util).

Notes
-----
- `eval_step` remains JIT-compiled. Visualization path is not JITted.
- `Model._stats` must return `routing_probs` (B,T,E) for routing visuals.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import wandb

from dna import Model, sample


# ============================== Eval loss & accuracy ============================== #


def compute_loss_for_eval(
    model: Model,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    inference: bool,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]]:
    """Compute masked cross-entropy loss for evaluation.

    Returns `(loss, (logits_shift, labels_shift, mask_shift, stats))`.
    `stats` is a tuple over hops; each hop is a dict with arrays batched over B.
    """
    ids = batch["input_ids"]  # (B, T)
    mask = batch["attention_mask"]  # (B, T)
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def fwd(x, m, k):
        logits, stats = model(
            x,
            key=k,
            inference=inference,
            attention_mask=m,
            gumbel_tau=gumbel_tau,
            router_temperature=router_temp,
            select_temperature=select_temp,
        )
        return logits, stats

    logits, stats = jax.vmap(fwd, in_axes=(0, 0, 0))(ids, mask, keys)

    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = mask[:, 1:]

    raw = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift)
    loss = (raw * mask_shift).sum() / jnp.maximum(mask_shift.sum(), 1)
    return loss, (logits_shift, labels_shift, mask_shift, stats)


@eqx.filter_jit

def eval_step(
    model: Model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
) -> Tuple[float, float]:
    """Single evaluation step returning `(loss, accuracy)`.

    JITted; avoid Python side-effects inside.
    """
    loss, (logits, labels, mask, _stats) = compute_loss_for_eval(
        model,
        batch,
        key,
        inference=True,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    preds = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = ((preds == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    return loss, acc


# ============================== Expert metadata helpers =========================== #


def get_expert_metadata(model: Model) -> Dict[str, Any]:
    """Build mapping from expert index → human label and type."""
    assert hasattr(model, "groups") and len(model.groups) > 0, "Model must have groups"

    # Total number of experts
    E = 0
    for g in model.groups:
        E = max(E, int(jnp.max(g["idx"]) + 1))
    E = max(E, sum(int(len(g["idx"])) for g in model.groups))

    expert_labels: List[Optional[str]] = [None] * E
    expert_types: List[Optional[str]] = [None] * E

    type_counters: Dict[str, int] = {}
    for g in model.groups:
        tname = type(g["static"]).__name__
        cnt = type_counters.get(tname, 0)
        idxs = list(map(int, jax.device_get(g["idx"]).tolist()))
        for i, e_idx in enumerate(idxs):
            expert_labels[e_idx] = f"{tname}_{cnt + i}"
            expert_types[e_idx] = tname
        type_counters[tname] = cnt + len(idxs)

    for i in range(E):
        if expert_labels[i] is None:
            expert_labels[i] = f"Expert_{i}"
        if expert_types[i] is None:
            expert_types[i] = "Unknown"

    unique_types = sorted(set(expert_types))
    type_to_id = {t: i for i, t in enumerate(unique_types)}
    expert_type_ids = jnp.array([type_to_id[t] for t in expert_types], dtype=jnp.int32)

    return dict(
        n_experts=E,
        expert_labels=expert_labels,
        expert_types=expert_types,
        expert_type_ids=expert_type_ids,
        type_to_id=type_to_id,
        id_to_type=unique_types,
    )


# ============================== Heatmap / comparison eval ========================= #


def evaluate_heatmap(
    model: Model,
    batch: Dict[str, Any],
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    *,
    num_examples: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate routing patterns for visualization and comparison.

    Returns
    -------
    batch_stats: Dict[str, Any]
        - importance_matrix: (H, E) average importance across batch (per hop, expert)
        - n_hops: int
        - n_experts: int
        - expert_labels / expert_types / expert_type_ids / type_to_id / id_to_type
    example_stats: Dict[str, Any]
        - indices: (S,) chosen batch indices
        - ids: (S, T), mask: (S, T)
        - top1_expert: (H, S, T), top1_prob: (H, S, T), top1_type_id: (H, S, T)
        - probs: (H, S, T, E) routing probs for the chosen examples
        - prompts: List[str] decoded preview text for each example
    """
    ids = batch["input_ids"]      # (B, T)
    mask = batch["attention_mask"]  # (B, T)
    B, T = ids.shape
    keys = jax.random.split(key, B)

    def fwd(x, m, k):
        logits, stats = model(
            x,
            key=k,
            inference=True,
            attention_mask=m,
            gumbel_tau=gumbel_tau,
            router_temperature=router_temp,
            select_temperature=select_temp,
        )
        return logits, stats

    # Forward pass over batch
    _, stats = jax.vmap(fwd, in_axes=(0, 0, 0))(ids, mask, keys)
    n_hops = len(stats)

    # Expert metadata
    meta = get_expert_metadata(model)
    E = meta["n_experts"]

    # Average importance across batch per hop
    batch_importance: List[jnp.ndarray] = []
    for hop_stats in stats:
        avg_importance = jnp.mean(hop_stats["importance_mean"], axis=0)  # (E,)
        if avg_importance.shape[0] != E:
            pad = jnp.zeros((E,), dtype=avg_importance.dtype)
            pad = pad.at[: avg_importance.shape[0]].set(avg_importance)
            avg_importance = pad
        batch_importance.append(avg_importance)
    importance_matrix = jnp.stack(batch_importance, axis=0)  # (H, E)

    # Choose *different* examples for detailed comparison
    S = int(min(num_examples, B))
    key_sel = jax.random.fold_in(key, 17)
    sel_idx = jax.random.choice(key_sel, B, shape=(S,), replace=False)

    # Collect per-example routing probabilities per hop
    probs_by_hop: List[jnp.ndarray] = []   # (H, S, T, E)
    top1_expert: List[jnp.ndarray] = []    # (H, S, T)
    top1_prob: List[jnp.ndarray] = []      # (H, S, T)
    top1_type_id: List[jnp.ndarray] = []   # (H, S, T)

    for hop_stats in stats:
        if "routing_probs" not in hop_stats:
            raise ValueError("Model stats missing 'routing_probs'. Ensure Model._stats returns it.")
        probs_bte = hop_stats["routing_probs"]            # (B, T, E)
        probs_ste = jnp.take(probs_bte, sel_idx, axis=0)  # (S, T, E)

        sums = probs_ste.sum(axis=-1)  # (S, T)
        valid = sums > 1e-9
        p_norm = jnp.where(valid[..., None], probs_ste / (sums[..., None] + 1e-9), 0.0)

        t1_idx = jnp.argmax(p_norm, axis=-1)  # (S, T)
        t1_val = jnp.take_along_axis(p_norm, t1_idx[..., None], axis=-1)[..., 0]  # (S, T)
        type_ids = meta["expert_type_ids"]  # (E,)
        t1_type = jnp.take(type_ids, t1_idx)  # (S, T)

        probs_by_hop.append(probs_ste)
        top1_expert.append(t1_idx)
        top1_prob.append(t1_val)
        top1_type_id.append(t1_type)

    probs = jnp.stack(probs_by_hop, axis=0)        # (H, S, T, E)
    top1_expert = jnp.stack(top1_expert, axis=0)   # (H, S, T)
    top1_prob = jnp.stack(top1_prob, axis=0)       # (H, S, T)
    top1_type_id = jnp.stack(top1_type_id, axis=0) # (H, S, T)

    ids_sel = jnp.take(ids, sel_idx, axis=0)     # (S, T)
    mask_sel = jnp.take(mask, sel_idx, axis=0)   # (S, T)

    batch_stats: Dict[str, Any] = {
        "importance_matrix": importance_matrix,
        "n_hops": n_hops,
        "n_experts": E,
        "expert_labels": meta["expert_labels"],
        "expert_types": meta["expert_types"],
        "expert_type_ids": meta["expert_type_ids"],
        "type_to_id": meta["type_to_id"],
        "id_to_type": meta["id_to_type"],
    }

    example_stats: Dict[str, Any] = {
        "indices": sel_idx,
        "ids": ids_sel,
        "mask": mask_sel,
        "probs": probs,
        "top1_expert": top1_expert,
        "top1_prob": top1_prob,
        "top1_type_id": top1_type_id,
        # prompts will be filled in plotting function via tokenizer.decode
    }

    return batch_stats, example_stats


# ============================== Plotting utilities =============================== #


def _build_type_cmap(id_to_type: List[str]) -> ListedColormap:
    """Color map with one distinct color per expert type."""
    n = max(10, len(id_to_type))
    base = plt.get_cmap("tab10")
    colors = [base(i % 10) for i in range(n)]
    return ListedColormap(colors)


def plot_routing_heatmap(batch_stats: Dict[str, Any], step: int) -> None:
    """Create and log the global routing heatmap to Weights & Biases."""
    importance = jax.device_get(batch_stats["importance_matrix"])  # (H, E)
    expert_labels = batch_stats.get("expert_labels", [f"E{i}" for i in range(importance.shape[1])])

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(importance.T, aspect="auto", cmap="hot", interpolation="nearest")

    ax.set_xlabel("Hop (Layer)", fontsize=12)
    ax.set_ylabel("Expert Module", fontsize=12)
    ax.set_title(f"Routing Heatmap - Step {step}", fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Importance", rotation=270, labelpad=20)

    ax.set_xticks(range(importance.shape[0]))
    ax.set_xticklabels([f"Hop {i}" for i in range(importance.shape[0])])

    ax.set_yticks(range(importance.shape[1]))
    ax.set_yticklabels(expert_labels, fontsize=8)

    plt.tight_layout()
    wandb.log({"routing/heatmap": wandb.Image(fig), "step": step})
    plt.close(fig)


def plot_token_type_grid(
    example_stats: Dict[str, Any],
    batch_stats: Dict[str, Any],
    tok,
    *,
    step: int,
    max_tokens: int = 256,
) -> None:
    """Visualize, for several examples and hops, which *type* each token routed to.

    Rows = examples, Cols = hops. Each cell is a 1×T strip:
      color = top-1 type, brightness overlay = confidence.
    Left caption shows the decoded preview (the example's prompt/text).
    """
    top1_type = jax.device_get(example_stats["top1_type_id"])  # (H, S, T)
    top1_prob = jax.device_get(example_stats["top1_prob"])    # (H, S, T)
    ids = jax.device_get(example_stats["ids"])                # (S, T)
    mask = jax.device_get(example_stats["mask"])              # (S, T)

    # Build decoded previews (clean text, no Ġ artifacts)
    prompts: List[str] = []
    for s in range(ids.shape[0]):
        valid_ids = ids[s][mask[s] > 0][:max_tokens]
        try:
            preview = tok.decode(valid_ids.tolist(), skip_special_tokens=True)
        except Exception:
            preview = ""
        prompts.append(preview)

    id_to_type: List[str] = batch_stats.get("id_to_type", [])
    type_cmap = _build_type_cmap(id_to_type)

    H, S, T = top1_type.shape
    Tvis = min(T, max_tokens)

    # Roomy layout
    fig_width = max(2.8 * H, 8.0)
    fig_height = max(2.2 * S, 4.0)
    fig, axes = plt.subplots(S, H, figsize=(fig_width, fig_height), squeeze=False)

    for s in range(S):
        for h in range(H):
            ax = axes[s, h]
            strip_types = top1_type[h, s, :Tvis][None, :]
            strip_probs = top1_prob[h, s, :Tvis][None, :]

            ax.imshow(strip_types, aspect="auto", cmap=type_cmap, vmin=0, vmax=max(1, len(id_to_type) - 1))
            ax.imshow(strip_probs, aspect="auto", cmap="gray", alpha=0.28, vmin=0.0, vmax=1.0)

            ax.set_xticks([])
            ax.set_yticks([])
            if s == 0:
                ax.set_title(f"Hop {h}", fontsize=12)

        # Left margin: example tag + prompt preview
        preview = prompts[s]
        label = f"Ex {s}\n{preview[:80]}{'…' if len(preview) > 80 else ''}"
        axes[s, 0].set_ylabel(label, rotation=0, labelpad=26, va="center", fontsize=10)

    # Legend for types
    handles = [plt.Rectangle((0, 0), 1, 1, color=type_cmap(i)) for i in range(len(id_to_type))]
    labels = [t for t in id_to_type]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, wspace=0.35, bottom=0.15)
    wandb.log({"routing/token_type_grid": wandb.Image(fig), "step": step})
    plt.close(fig)


# ============================== Text inspection / printing ======================= #


def print_example_routing(
    example_stats: Dict[str, Any],
    batch_stats: Dict[str, Any],
    tok,
    *,
    num_tokens: int = 10,
) -> None:
    """Pretty-print routing for chosen examples with real expert labels."""
    probs = jax.device_get(example_stats["probs"])  # (H, S, T, E)
    ids = jax.device_get(example_stats["ids"])      # (S, T)
    mask = jax.device_get(example_stats["mask"])    # (S, T)

    expert_labels: List[str] = batch_stats.get("expert_labels", [])

    print("\n" + "=" * 60)
    print("Example Routing Patterns (multi-sample)")
    print("=" * 60)

    H, S, T, E = probs.shape

    for s in range(S):
        valid_ids = ids[s][mask[s] > 0][:128]
        try:
            preview = tok.decode(valid_ids.tolist(), skip_special_tokens=True)
        except Exception:
            preview = ""

        print(f"\n--- Example {s} ---")
        print(f"Prompt: {preview[:120]}{'…' if len(preview) > 120 else ''}")
        for h in range(H):
            print(f"\nHop {h}:")
            hop_probs = probs[h, s]  # (T, E)
            shown = min(num_tokens, valid_ids.shape[0])
            for t_idx in range(shown):
                p = hop_probs[t_idx]
                top3_idx = jnp.argsort(p)[-3:][::-1]
                top3_p = p[top3_idx]
                names = [expert_labels[int(e)] if int(e) < len(expert_labels) else f"E{int(e)}" for e in top3_idx]
                print(
                    f"  t{t_idx:02d}: {names[0]}={float(top3_p[0]):.3f}, "
                    f"{names[1]}={float(top3_p[1]):.3f}, {names[2]}={float(top3_p[2]):.3f}"
                )


# ============================== Text generation ================================= #


def generate_examples(
    model: Model,
    tok,
    *,
    key: jax.Array,
    gen_len: int = 100,
    per_prompt: int = 1,
    router_temp: float = 1.5,
    select_temp: float = 1.75,
    gumbel_tau: float = 1.2,
    prompts: Optional[List[str]] = None,
    n_examples: int = 5,
) -> None:
    """Generate example text completions for quick sanity checks."""
    if prompts is None:
        prompts = [
            "One day, ",
            "Once upon a time, ",
            "In a small town, ",
            "Long ago, ",
            "On a sunny morning, ",
        ]

    print("\n" + "=" * 40)
    print("Generated Examples")
    print("=" * 40)

    for p in prompts[:n_examples]:
        prompt_ids = jnp.array(tok.encode(p), dtype=jnp.int32)
        key, *subs = jax.random.split(key, per_prompt + 1)
        subs = jnp.stack(subs)

        @jax.vmap
        def _sample(k):
            return sample(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=gen_len,
                temperature=0.8,
                key=k,
                router_temperature=router_temp,
                select_temperature=select_temp,
                gumbel_tau=gumbel_tau,
                greedy=False,
                pad_id=tok.pad_token_id,
                eos_id=tok.pad_token_id,
            )

        toks = _sample(subs)
        for seq in jax.device_get(toks):
            seq = list(seq)
            if tok.eos_token_id in seq:
                seq = seq[: seq.index(tok.eos_token_id) + 1]
            text = tok.decode(seq, skip_special_tokens=True)
            print(f"[{p}] {text}")
            print("-" * 40)


# ============================== Routing metrics ================================= #


def routing_metrics_from_stats(
    stats_tuple_batched: Tuple[Dict[str, Any], ...],
    *,
    prefix: str,
    expert_type_ids: Optional[jnp.ndarray] = None,
    capacity: Optional[int] = None,
) -> Dict[str, float]:
    """Extract aggregated routing metrics across hops & batch."""
    hop_logs: List[Dict[str, float]] = []

    for hop in stats_tuple_batched:
        load = jnp.asarray(hop["load"])  # (B, E)
        rho_mean = jnp.asarray(hop["rho_mean"])  # (B,)
        entropy_mean = jnp.asarray(hop["entropy_mean"])  # (B,)
        cap_drop = jnp.asarray(hop["cap_drop_frac_edges"])  # (B,)
        eff_topk_mean = jnp.asarray(hop["eff_topk_mean"])  # (B,)
        eff_topk_min = jnp.asarray(hop["eff_topk_min"])  # (B,)
        eff_topk_max = jnp.asarray(hop["eff_topk_max"])  # (B,)
        cap_util_mean = jnp.asarray(hop["cap_util_mean"])  # (B,)
        cap_util_min = jnp.asarray(hop["cap_util_min"])  # (B,)
        cap_util_max = jnp.asarray(hop["cap_util_max"])  # (B,)

        util_b = jnp.mean((load > 0).astype(jnp.float32), axis=1)  # (B,)

        denom = jnp.sum(load, axis=1, keepdims=True) + 1e-9
        load_norm = load / denom  # (B, E)
        load_std_b = jnp.std(load_norm, axis=1)  # (B,)

        top1_share = jnp.array(0.0)
        kl_uniform = jnp.array(0.0)
        if "routing_probs" in hop:
            p = jnp.asarray(hop["routing_probs"])  # (B, T, E)
            sums = p.sum(axis=-1)
            valid = sums > 1e-9
            p_norm = jnp.where(valid[..., None], p / (sums[..., None] + 1e-9), 0.0)
            t1 = jnp.max(p_norm, axis=-1)
            top1_share = jnp.where(valid, t1, 0.0).sum() / jnp.maximum(valid.sum(), 1)

            E = p.shape[-1]
            kl = (p_norm * jnp.log(p_norm * E + 1e-9)).sum(axis=-1)
            kl_uniform = jnp.where(valid, kl, 0.0).sum() / jnp.maximum(valid.sum(), 1)

        cap_sat = jnp.array(0.0)
        if capacity is not None and capacity > 0:
            cap_sat = jnp.mean((load >= capacity).astype(jnp.float32))

        hop_log = dict(
            rho_mean=float(jnp.mean(rho_mean)),
            entropy_mean=float(jnp.mean(entropy_mean)),
            util=float(jnp.mean(util_b)),
            load_std=float(jnp.mean(load_std_b)),
            cap_drop_frac=float(jnp.mean(cap_drop)),
            eff_topk_mean=float(jnp.mean(eff_topk_mean)),
            eff_topk_min=float(jnp.mean(eff_topk_min)),
            eff_topk_max=float(jnp.mean(eff_topk_max)),
            cap_util_mean=float(jnp.mean(cap_util_mean)),
            cap_util_min=float(jnp.mean(cap_util_min)),
            cap_util_max=float(jnp.mean(cap_util_max)),
            top1_share_mean=float(top1_share),
            kl_uniform_mean=float(kl_uniform),
            cap_saturated_frac=float(cap_sat),
        )

        if expert_type_ids is not None:
            type_ids = jnp.asarray(expert_type_ids)
            total_load = jnp.sum(load)
            if total_load > 0:
                load_e = jnp.sum(load, axis=0)
                for t_id in jnp.unique(type_ids):
                    t_id_int = int(t_id)
                    frac = jnp.sum(jnp.where(type_ids == t_id, load_e, 0)) / total_load
                    hop_log[f"type_util_{t_id_int}"] = float(frac)

        hop_logs.append(hop_log)

    def _mean_key(k: str) -> float:
        vals = [h[k] for h in hop_logs if k in h]
        return float(sum(vals) / max(len(vals), 1))

    out = {
        f"routing/{prefix}/rho_mean": _mean_key("rho_mean"),
        f"routing/{prefix}/entropy_mean": _mean_key("entropy_mean"),
        f"routing/{prefix}/util": _mean_key("util"),
        f"routing/{prefix}/load_std": _mean_key("load_std"),
        f"routing/{prefix}/cap_drop_frac": _mean_key("cap_drop_frac"),
        f"routing/{prefix}/eff_topk_mean": _mean_key("eff_topk_mean"),
        f"routing/{prefix}/eff_topk_min": _mean_key("eff_topk_min"),
        f"routing/{prefix}/eff_topk_max": _mean_key("eff_topk_max"),
        f"routing/{prefix}/cap_util_mean": _mean_key("cap_util_mean"),
        f"routing/{prefix}/cap_util_min": _mean_key("cap_util_min"),
        f"routing/{prefix}/cap_util_max": _mean_key("cap_util_max"),
        f"routing/{prefix}/top1_share_mean": _mean_key("top1_share_mean"),
        f"routing/{prefix}/kl_uniform_mean": _mean_key("kl_uniform_mean"),
        f"routing/{prefix}/cap_saturated_frac": _mean_key("cap_saturated_frac"),
    }

    type_keys = sorted({k for h in hop_logs for k in h.keys() if k.startswith("type_util_")})
    for tk in type_keys:
        out[f"routing/{prefix}/{tk}"] = _mean_key(tk)

    return out


# ============================== W&B helper ======================================= #


def log_routing_visuals(
    model: Model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    tok,
    step: int,
    num_examples: int = 2,
    max_tokens_grid: int = 256,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run evaluation + log heatmaps/strips to W&B, and return the stats dicts."""
    batch_stats, example_stats = evaluate_heatmap(
        model,
        batch,
        key,
        gumbel_tau,
        router_temp,
        select_temp,
        num_examples=num_examples,
    )

    plot_routing_heatmap(batch_stats, step)
    plot_token_type_grid(example_stats, batch_stats, tok, step=step, max_tokens=max_tokens_grid)

    return batch_stats, example_stats
