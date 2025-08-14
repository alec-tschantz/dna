# logs.py
"""Generic training/eval utilities for DNA + Dense models."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import textwrap
from collections import Counter
from pathlib import Path
from datetime import datetime
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import wandb

from dna import generate


# =============================================================================
# Public logging API
# =============================================================================


def log_checkpoint(
    *,
    run_name: str,
    cfg: Config,
    step: int,
    model: eqx.Module,
    opt_state,
    lr_value: float,
):
    """Save model/optimizer + minimal JSON meta."""
    ckpt_dir = Path(cfg.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_dir / f"modelstep{step}.eqx"
    opt_path = ckpt_dir / f"optstep{step}.eqx"
    meta_path = ckpt_dir / f"metastep{step}.json"

    eqx.tree_serialise_leaves(model_path, eqx.filter(model, eqx.is_array))
    eqx.tree_serialise_leaves(opt_path, opt_state)

    meta = {
        "step": int(step),
        "run_name": run_name,
        "time": datetime.utcnow().isoformat() + "Z",
        "lr": float(lr_value),
        "seed": int(cfg.seed),
        "model_type": cfg.model_type,
        "router_type": cfg.router_type,
        "seq_len": int(cfg.seq_len),
        "batch_size": int(cfg.batch_size),
        "n_hops": int(cfg.n_hops),
        "topk": int(cfg.topk),
        "capacity": int(cfg.capacity),
        "d_model": int(cfg.d_model),
        "n_heads": int(cfg.n_heads),
        "n_modules": int(cfg.n_modules),
        "wandb_project": cfg.wandb_project,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ckpt] saved: {model_path}  |  {opt_path}")


def log_initial_stats(
    model,
    first_batch: Dict[str, Any],
    *,
    cfg: Any = None,
    seq_len: int = 0,
    capacity: Optional[int] = None,
    topk: Optional[int] = None,
    n_hops: Optional[int] = None,
    model_type: str = "dna",
) -> None:
    """Log initial model + data stats. (API unchanged)"""
    n_params = count_params(model)
    lmean, lmin, lmax, pmean = batch_seq_stats(
        jnp.asarray(first_batch["attention_mask"]), seq_len
    )

    print(cfg)
    print("=" * 60)
    print(f"Total parameters: {n_params:,}")
    if model_type.lower() == "dna":
        print(f"Architecture: Capacity={capacity}, TopK={topk}, Hops={n_hops}")
    else:
        print(f"Architecture: Dense | Layers(n_hops)={n_hops}")
    print(f"Sequence stats: mean={lmean:.1f}, min={lmin}, max={lmax}")
    print(f"Average padding: {pmean:.1f}")
    print("=" * 60)

    base = {
        "model/type": model_type,
        "n_params": n_params,
        "hops": n_hops,
        "seq/len_mean": lmean,
        "seq/len_min": lmin,
        "seq/len_max": lmax,
        "seq/pad_mean": pmean,
        "step": 0,
    }
    if model_type.lower() == "dna":
        base.update({"capacity": capacity, "topk": topk})
    wandb.log(base)


def log_train_step(
    *,
    step: int,
    cfg,
    schedule_fn,
    model,
    loss,
    acc,
    gnorm,
    step_time_ms: float,
    stats,
    model_kwargs: Dict[str, jax.Array],
):
    """Log training metrics + router metrics. (API unchanged)"""
    train_logs = {
        "train/loss": float(loss),
        "train/acc": float(acc),
        "train/grad_norm": float(gnorm),
        "train/lr": float(schedule_fn(step)),
        "train/tok_per_sec": cfg.batch_size
        * cfg.seq_len
        / max(step_time_ms / 1000.0, 1e-9),
        "train/weights_global_norm": l2_tree_norm(model),
        "step": step,
    }

    if has_routing(model) and isinstance(stats, (tuple, list)) and len(stats) > 0:
        stats_host = jax.tree_util.tree_map(jax.device_get, stats)
        router_logs = {
            "router/norm": router_l2_norm(model),
            "router/temps/router": float(
                model_kwargs.get("router_temp", jnp.array([0.0]))[0]
            ),
            "router/temps/select": float(
                model_kwargs.get("select_temp", jnp.array([0.0]))[0]
            ),
            "router/temps/gumbel": float(
                model_kwargs.get("gumbel_tau", jnp.array([0.0]))[0]
            ),
            "step": step,
        }
        router_logs.update(
            routing_metrics_from_stats(
                stats_host,
                prefix="router/train",
                capacity=getattr(cfg, "capacity", None),
            )
        )
        router_logs.update(_extra_routing_metrics(stats_host, prefix="router/train"))
        wandb.log({**train_logs, **router_logs})
    else:
        wandb.log(train_logs)


def run_eval_suite(
    *,
    step: int,
    cfg,
    model,
    eval_step_fn,
    val_stream,
    key: jax.Array,
    tok,
    model_kwargs_train: Dict[str, jax.Array],
    sample_batch_fn,
):
    # ---- core eval ----
    key, eval_key = jax.random.split(key)
    eval_batch = sample_batch_fn(val_stream, cfg.eval_samples // cfg.seq_len)
    eval_kwargs = {
        "gumbel_tau": jnp.array([0.0], dtype=jnp.float32),
        "router_temp": jnp.array(
            [model_kwargs_train.get("router_temp", jnp.array([1.0]))[0]],
            dtype=jnp.float32,
        ),
        "select_temp": jnp.array(
            [model_kwargs_train.get("select_temp", jnp.array([1.0]))[0]],
            dtype=jnp.float32,
        ),
    }
    val_loss, val_acc = eval_step_fn(
        model, eval_batch, key=eval_key, model_kwargs=eval_kwargs
    )
    wandb.log({"eval/loss": float(val_loss), "eval/acc": float(val_acc), "step": step})
    print(f"  [Eval] Loss: {float(val_loss):.4f} | Acc: {float(val_acc):.4f}")

    # ---- routing visuals ----
    key, vis_key = jax.random.split(key)
    vis_batch = sample_batch_fn(val_stream, min(16, cfg.batch_size))

    log_routing_visuals_if_available(
        model,
        vis_batch,
        key=vis_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        tok=tok,
        step=step,
        num_examples=1,
        max_tokens_grid=96,
    )

    # ---- token→expert arrow graphic (new) ----
    key, arrows_key = jax.random.split(key)
    try:
        log_token_expert_flow_arrows(
            model,
            vis_batch,
            tok,
            key=arrows_key,
            gumbel_tau=0.0,
            router_temp=float(eval_kwargs["router_temp"][0]),
            select_temp=float(eval_kwargs["select_temp"][0]),
            step=step,
            max_tokens=64,  # limit for readability (32–64 recommended)
        )
    except Exception as e:
        print(f"[warn] token→expert arrows failed: {e}")

    _ = log_routing_sankey_if_available(
        model,
        vis_batch,
        key=vis_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        top_paths_per_hop=18,
        min_frac=0.01,
        by_type=False,
    )

    # ---- Path Diversity ----
    key, div_key = jax.random.split(key)
    div = analyze_path_diversity(
        model,
        vis_batch,
        key=div_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        top_paths=14,
    )
    if div is not None:
        plot_path_diversity_dashboard(div, step)

    # ---- Token↔Path Specialization ----
    key, spec_key = jax.random.split(key)
    spec = analyze_token_path_specialization(
        model,
        vis_batch,
        tok,
        key=spec_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        max_paths=10,
        min_token_count=6,
    )
    if spec is not None:
        plot_token_path_specialization(spec, step)
        log_token_path_sankey(spec, step)  # optional, Plotly

    # ---- generation ----
    key, gen_key = jax.random.split(key)
    prompts = [
        "Once upon a time, ",
        "The little robot ",
        "In the magical forest, ",
        "One sunny morning, ",
        "The brave knight ",
    ]
    results = generate(
        model,
        tok,
        key=gen_key,
        gen_len=cfg.gen_len,
        per_prompt=1,
        router_temp=float(model_kwargs_train.get("router_temp", jnp.array([1.0]))[0]),
        select_temp=float(model_kwargs_train.get("select_temp", jnp.array([1.0]))[0]),
        gumbel_tau=float(model_kwargs_train.get("gumbel_tau", jnp.array([0.0]))[0]),
        prompts=prompts,
        n_examples=cfg.n_examples,
    )

    print("\n" + "=" * 60)
    print("Generated Examples")
    print("=" * 60)
    for r in results:
        p = r["prompt"]
        for comp in r["completions"]:
            text = comp["text"].replace("\n", "\\n")
            print(f"\nPrompt: {p}")
            print(
                f"Completion [{comp['length']} tokens]{' (eos)' if comp['stopped_eos'] else ''}:"
            )
            for line in textwrap.wrap(text, width=100, break_long_words=False):
                print(line)
            print("-" * 40)

    return key


# =============================================================================
# Core utility functions
# =============================================================================


def count_params(tree) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(x.size for x in leaves))


def l2_tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq + 1e-12))


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


def router_l2_norm(model) -> float:
    if has_routing(model):
        leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
        if not leaves:
            return 0.0
        sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
        return float(jnp.sqrt(sq + 1e-12))
    return 0.0


# =============================================================================
# Routing metrics computation
# =============================================================================


def _extra_routing_metrics(stats_host, prefix: str) -> Dict[str, float]:
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


# =============================================================================
# Expert metadata extraction
# =============================================================================


def _expert_meta(model) -> Optional[Dict[str, Any]]:
    """Collect labels/types for each expert from model.groups."""
    if not has_routing(model) or not hasattr(model, "groups") or len(model.groups) == 0:
        return None

    E = 0
    for g in model.groups:
        E = max(E, int(jnp.max(g["idx"]) + 1))

    labels: List[Optional[str]] = [None] * E
    types: List[Optional[str]] = [None] * E
    counters: Dict[str, int] = {}

    for g in model.groups:
        tname = type(g["static"]).__name__
        cnt = counters.get(tname, 0)
        idxs = list(map(int, jax.device_get(g["idx"]).tolist()))
        for i, e_idx in enumerate(idxs):
            labels[e_idx] = f"{tname}_{cnt + i}"
            types[e_idx] = tname
        counters[tname] = cnt + len(idxs)

    for i in range(E):
        if labels[i] is None:
            labels[i] = f"Expert_{i}"
        if types[i] is None:
            types[i] = "Unknown"

    uniq = sorted(set(types))
    type_to_id = {t: i for i, t in enumerate(uniq)}
    type_ids = jnp.array([type_to_id[t] for t in types], dtype=jnp.int32)

    return dict(
        n_experts=E,
        expert_labels=labels,
        expert_types=types,
        expert_type_ids=type_ids,
        type_to_id=type_to_id,
        id_to_type=uniq,
    )


def _forward_batched_for_stats(
    model,
    ids_bt: jnp.ndarray,
    mask_bt: jnp.ndarray,
    keys_b: jax.Array,
    *,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
):
    """Run model across batch to collect per-hop routing stats."""

    def fwd(x, m, k):
        _, stats = model(
            x,
            key=k,
            inference=True,
            mask=m,
            gumbel_tau=gumbel_tau,
            router_temp=router_temp,
            select_temp=select_temp,
        )
        return stats

    return jax.vmap(fwd, in_axes=(0, 0, 0))(ids_bt, mask_bt, keys_b)


def _type_base_cmap(type_name: str) -> str:
    t = type_name.lower()
    if "ttent" in t:
        return "Blues"
    if "feed" in t or "mlp" in t:
        return "Oranges"
    if "ident" in t:
        return "Greens"
    return "Purples"


def _expert_color_table(meta) -> np.ndarray:
    """RGBA per expert, grouped by type with graded shades."""
    type_ids = np.asarray(meta["expert_type_ids"])
    id_to_type = list(meta["id_to_type"])
    E = type_ids.shape[0]
    colors = np.zeros((E, 4), dtype=np.float32)

    for t_id, t_name in enumerate(id_to_type):
        idxs = np.where(type_ids == t_id)[0]
        if idxs.size == 0:
            continue
        cmap = plt.get_cmap(_type_base_cmap(t_name))
        shades = [0.75] if idxs.size == 1 else np.linspace(0.35, 0.95, idxs.size)
        for j, e in enumerate(sorted(idxs.tolist())):
            rgba = np.array(cmap(float(shades[j])))
            rgba[3] = 1.0
            colors[e] = rgba
    return colors


# =============================================================================
# Basic routing visualizations
# =============================================================================


def evaluate_for_visuals(
    model,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    num_examples: int = 2,
):
    """Collect batch + per-example routing info for visuals."""
    if not has_routing(model):
        return None, None

    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B, _ = ids.shape
    keys = jax.random.split(key, B)

    stats = _forward_batched_for_stats(
        model,
        ids,
        mask,
        keys,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if not isinstance(stats, (tuple, list)) or len(stats) == 0:
        return None, None

    meta = _expert_meta(model)
    if meta is None:
        return None, None

    n_hops = len(stats)
    E = meta["n_experts"]

    # Average importance per hop over batch
    importance_matrix = []
    for hop_stats in stats:
        imp_be = hop_stats["importance"]
        avg = jnp.mean(imp_be, axis=0)
        if avg.shape[0] != E:
            pad = jnp.zeros((E,), dtype=avg.dtype).at[: avg.shape[0]].set(avg)
            avg = pad
        importance_matrix.append(avg)
    importance_matrix = jnp.stack(importance_matrix, axis=0)

    # sample a few sequences for token-flow visual
    S = int(min(num_examples, B))
    sel_idx = jax.random.choice(
        jax.random.fold_in(key, 123), B, shape=(S,), replace=False
    )

    probs_by_hop, top1_idx, top1_prob, top1_type = [], [], [], []
    for hop_stats in stats:
        p_bte = hop_stats["routing_probs"]
        p_ste = jnp.take(p_bte, sel_idx, axis=0)

        sums = p_ste.sum(axis=-1)
        valid = sums > 1e-9
        p_norm = jnp.where(valid[..., None], p_ste / (sums[..., None] + 1e-9), 0.0)

        idx = jnp.argmax(p_norm, axis=-1)
        val = jnp.take_along_axis(p_norm, idx[..., None], axis=-1)[..., 0]

        type_ids = meta["expert_type_ids"]
        typ = jnp.take(type_ids, idx)

        probs_by_hop.append(p_ste)
        top1_idx.append(idx)
        top1_prob.append(val)
        top1_type.append(typ)

    batch_stats = dict(
        importance_matrix=importance_matrix,
        n_hops=n_hops,
        n_experts=E,
        expert_labels=meta["expert_labels"],
        expert_types=meta["expert_types"],
        expert_type_ids=meta["expert_type_ids"],
        type_to_id=meta["type_to_id"],
        id_to_type=meta["id_to_type"],
    )

    example_stats = dict(
        indices=sel_idx,
        ids=jnp.take(ids, sel_idx, axis=0),
        mask=jnp.take(mask, sel_idx, axis=0),
        probs=jnp.stack(probs_by_hop, axis=0),
        top1_expert=jnp.stack(top1_idx, axis=0),
        top1_prob=jnp.stack(top1_prob, axis=0),
        top1_type_id=jnp.stack(top1_type, axis=0),
    )

    return batch_stats, example_stats


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

    # plt.tight_layout()
    wandb.log({"routing/heatmap": wandb.Image(fig), "step": step})
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

    # plt.tight_layout()
    wandb.log({"routing/token_flow_rich": wandb.Image(fig), "step": step})
    plt.close(fig)


def log_routing_visuals_if_available(
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
        plot_token_flow_rich(
            example_stats, batch_stats, tok, step=step, max_tokens=max_tokens_grid
        )

    return batch_stats, example_stats


# =============================================================================
# Advanced routing analysis: Sankey + transitions
# =============================================================================


def _collect_top1_indices_all(
    model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
):
    """Get top-1 expert per token per hop for an entire batch."""
    if not has_routing(model):
        return None, None, None

    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"]).astype(bool)
    B, T = ids.shape
    keys = jax.random.split(key, B)

    stats = _forward_batched_for_stats(
        model,
        ids,
        mask.astype(jnp.float32),
        keys,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if not isinstance(stats, (tuple, list)) or len(stats) == 0:
        return None, None, None

    top1_idx = []
    for hop in stats:
        p = jnp.asarray(hop["routing_probs"])
        idx = jnp.argmax(p, axis=-1)
        top1_idx.append(idx)
    top1 = jnp.stack(top1_idx, axis=0)

    meta = _expert_meta(model)
    if meta is None:
        return None, None, None

    return jax.device_get(top1), jax.device_get(mask), meta


def _rgba_to_plotly(rgba: np.ndarray, alpha: float = 0.8) -> str:
    r, g, b, _ = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(int)
    return f"rgba({r},{g},{b},{alpha})"


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
    """Interactive Sankey of token routing transitions + transition heatmaps."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

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
    wandb.log({"routing/sankey": fig_sankey, "step": step})

    # transition heatmaps (matplotlib)
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
        # plt.tight_layout()
        wandb.log({"routing/transition_heatmap": wandb.Image(fig_hm), "step": step})
        plt.close(fig_hm)

    return fig_sankey


# =============================================================================
# Path Diversity
# =============================================================================


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

    # (A) metric tiles
    axA = fig.add_subplot(gs[0, :])
    axA.axis("off")
    tiles = [
        ("Unique paths", f"{div['unique_paths']:,}"),
        ("Effective #paths", f"{div['effective_paths']:.1f}"),
        ("Entropy (norm.)", f"{div['shannon_norm']:.2f}"),
        ("Gini (↓ better spread)", f"{div['gini']:.2f}"),
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

    # (B) Lorenz curve
    axB = fig.add_subplot(gs[1, 0])
    p_sorted = np.sort(div["probs"])
    L = np.concatenate([[0.0], np.cumsum(p_sorted)])
    X = np.linspace(0, 1, len(L))
    axB.plot(X, L, linewidth=2.2)
    axB.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    axB.set_xlabel("Cumulative fraction of paths")
    axB.set_ylabel("Cumulative fraction of tokens")
    axB.set_title(f"Lorenz Curve (Gini={div['gini']:.2f})")

    # (C) Top paths bar chart
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

    # (D) Path barcodes (rows = paths, cols = hops)
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
    wandb.log({"routing/path_diversity": wandb.Image(fig), "step": step})
    plt.close(fig)


# =============================================================================
#  Token ↔ Path Specialization
# =============================================================================


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

    # ============== (1) TOP-LEFT: Category mix per path (row-normalized) ==========
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

    # ============== (2) BOTTOM-LEFT: Path composition (experts per hop) ===========
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

    # ============== (3) RIGHT COLUMN: Compact top-token cards ======================
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.axis("off")

    # Build two columns of cards for readability
    cards = []
    for i, tokens in enumerate(spec["top_tokens"]):
        # Path signature (expert IDs) for quick reference
        path_sig = "→".join(str(e) for e in paths[i])
        card = f"P{i+1}  (n={counts_per_path[i]})\n" f"path: [{path_sig}]\n"
        for s, cnt, lift, _ in tokens[:6]:
            s_disp = (s or "·").replace("\n", "\\n")
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
            family="monospace",
            fontsize=9,
            bbox=box_kw,
        )

    plt.suptitle(
        f"Token ↔ Path Specialization • step {step}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    wandb.log({"routing/token_path_specialization": wandb.Image(fig), "step": step})
    plt.close(fig)


def log_token_path_sankey(spec: Dict[str, Any], step: int):
    """Optional Plotly Sankey: token categories → paths."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    cat = spec["categories"]
    # approximate counts by mixing proportions × path counts
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
    wandb.log({"routing/token_path_sankey": fig, "step": step})

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
    show_topk: int = 2,         # << only draw the actual routed top-k edges
    max_experts: int = 16,      # global cap to keep the canvas readable
    show_other_bucket: bool = True,
    eps: float = 1e-8,
):
    """
    Token → (top-k) lines → Experts → Tokens, repeated for each hop.
    - Tokens are repeated per hop (T(h) -> E(h) -> T(h+1)), so per-step routing is explicit.
    - Only the top-k edges (default 2) per token & hop are drawn.
    - Experts are a fixed global subset ranked by mass on the visible slice, with optional 'Other'.
    - Line width/opacity/color reflect the routing probability.
    - Bottom row shows greedy predicted next tokens.
    Logs a matplotlib figure to W&B at 'routing/token_expert_flow_arrows'.
    """
    import numpy as np
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, PathPatch
    from matplotlib.path import Path as MplPath

    if not has_routing(model):
        return

    # ---- Collect routing/example stats (reuses your helper) ----
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
    H = int(np.asarray(example_stats["top1_expert"]).shape[0])   # hops
    ids_all = np.asarray(example_stats["ids"])[0]                # (T,)
    mask_all = np.asarray(example_stats["mask"])[0].astype(bool) # (T,)
    probs_all = np.asarray(example_stats["probs"])[:, 0, :, :]   # (H, T, E)
    E = int(probs_all.shape[-1])

    # ---- Visible window (keep order) ----
    valid_pos = np.where(mask_all)[0]
    if valid_pos.size == 0:
        return
    Tvis = int(min(valid_pos.size, max(8, max_tokens)))
    vis_idx = valid_pos[:Tvis]

    # Slice tensors to visible window
    ids = ids_all[vis_idx]                # (Tvis,)
    mask = np.ones_like(ids, dtype=bool)
    P = probs_all[:, vis_idx, :]          # (H, Tvis, E)

    # ---- Token text (robust, matching preview) ----
    def _ids_to_tokens(ids_1d: np.ndarray) -> list[str]:
        # 1) HF: convert_ids_to_tokens if available (preserves exact split)
        if hasattr(tok, "convert_ids_to_tokens"):
            try:
                toks = tok.convert_ids_to_tokens(ids_1d.tolist(), skip_special_tokens=False)
                # make them compact but faithful
                out = []
                for s in toks:
                    s = str(s)
                    # Show space markers from BPEs in a human way but keep alignment
                    s = s.replace("Ġ", "␠")
                    out.append(s if len(s) <= 4 else (s[:3] + "…"))
                return out
            except Exception:
                pass
        # 2) Fallback: decode each id individually without skipping/cleanup
        out = []
        for tid in ids_1d.tolist():
            s = ""
            try:
                # Not all tokenizers accept clean_up_tokenization_spaces;
                # try best-effort and fall back.
                try:
                    s = tok.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except TypeError:
                    s = tok.decode([int(tid)], skip_special_tokens=False)
            except Exception:
                s = "?"
            s = (s or "·").replace("\n", "⏎")
            if s.strip() == "":
                s = "␠"
            out.append(s if len(s) <= 4 else (s[:3] + "…"))
        return out

    tok_disp = _ids_to_tokens(ids)

    # Preview line uses the same ids slice to avoid any mismatch
    try:
        try:
            preview = tok.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            preview = tok.decode(ids.tolist(), skip_special_tokens=False)
    except Exception:
        preview = ""
    if len(preview) > 160:
        preview = preview[:157] + "…"

    # ---- Normalize probs per (hop, token) ----
    denom = P.sum(axis=-1, keepdims=True)
    Pn = np.where(denom > eps, P / (denom + eps), 0.0)  # (H, Tvis, E)

    # ---- Choose a fixed global subset of experts to display ----
    mass_per_e = Pn.sum(axis=(0, 1))  # (E,)
    order = np.argsort(-mass_per_e)
    keep = order[: min(max_experts, E)].tolist()
    keep_set = set(int(e) for e in keep)
    OTHER = -1
    include_other = show_other_bucket and (len(keep_set) < E and float(mass_per_e.sum()) > 0.0)

    # Fixed X positions for experts (even across the token span)
    n_nodes = len(keep_set) + (1 if include_other else 0)
    if n_nodes == 0:
        return
    if n_nodes == 1:
        ex_x = np.array([max(0, (Tvis - 1) * 0.5)], dtype=float)
    else:
        ex_x = np.linspace(0, max(Tvis - 1, 1), num=n_nodes, dtype=float)
    e_list = list(keep_set) + ([OTHER] if include_other else [])
    e_to_x = {int(e): float(x) for e, x in zip(e_list, ex_x)}

    # ---- Colors ----
    expert_rgba = _expert_color_table(batch_stats)  # (E,4)
    type_names = list(batch_stats["id_to_type"])
    expert_type_ids = np.asarray(batch_stats["expert_type_ids"])

    def _etype_initial(eid: int) -> str:
        if eid == OTHER:
            return "O"
        tname = type_names[int(expert_type_ids[eid])]
        return ("A" if "ttent" in tname.lower()
                else "F" if ("feed" in tname.lower() or "mlp" in tname.lower())
                else "I")

    def _expert_label(eid: int) -> str:
        return "Other" if eid == OTHER else f"{_etype_initial(eid)}{int(eid)}"

    def _expert_color(eid: int) -> np.ndarray:
        if eid == OTHER:
            return np.array([0.62, 0.62, 0.68, 1.0], dtype=np.float32)
        return expert_rgba[int(eid)]

    # color/opacity/width as a function of prob
    def _line_style(eid: int, p: float) -> tuple[np.ndarray, float]:
        base = _expert_color(eid).copy()
        p = float(np.clip(p, 0.0, 1.0))
        base[:3] = base[:3] * (0.35 + 0.65 * p) + (1.0 - p) * 0.10
        base[3] = 0.20 + 0.75 * p
        lw = 0.8 + 3.0 * p
        return base, lw

    # ---- Greedy predictions aligned to same positions ----
    try:
        logits, _ = model(
            jnp.asarray(ids_all[None, :]),  # full sequence for logits alignment
            key=key,
            inference=True,
            mask=jnp.asarray(mask_all[None, :], dtype=jnp.float32),
            gumbel_tau=float(gumbel_tau),
            router_temp=float(router_temp),
            select_temp=float(select_temp),
        )
        pred_full = np.asarray(jnp.argmax(logits[0], axis=-1))
    except Exception:
        pred_full = np.roll(ids_all, -1)
        if pred_full.size >= 2:
            pred_full[-1] = pred_full[-2]
    pred_ids = pred_full[vis_idx]
    pred_disp = _ids_to_tokens(pred_ids)

    # ---- Layout with EXTRA spacing ----
    row_gap = 1.55   # << more vertical spacing
    token_h = 0.74
    expert_h = 0.70

    def y_token_row(i: int) -> float:
        return (2 * i) * row_gap

    def y_expert_row(h: int) -> float:
        return (2 * h + 1) * row_gap

    y_pred = y_token_row(H) + 0.40 + 0.60

    fig_w = max(15.0, 0.34 * Tvis + 10.0)
    fig_h = max(7.5, (2 * H + 1) * row_gap + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    ax.set_xlim(-0.8, Tvis - 0.2)
    ax.set_ylim(-0.8, y_pred + 1.3)
    ax.axis("off")

    # Background bands per expert row
    for h in range(H):
        y0 = y_expert_row(h) - 0.55
        ax.add_patch(Rectangle((-0.9, y0), Tvis + 1.8, 1.9, facecolor="#fafafa", edgecolor="none", zorder=0))

    # Draw a token row
    def _draw_token_row(y: float, labels: list[str], face="#e7eef6", edge="#c9d5e3", txt="#0f172a"):
        for j in range(Tvis):
            ax.add_patch(Rectangle((j - 0.42, y - token_h/2), 0.84, token_h,
                                   facecolor=face, edgecolor=edge, linewidth=0.8, zorder=7))
            ax.text(j, y, labels[j], ha="center", va="center",
                    fontsize=8, family="monospace", color=txt, zorder=9)

    # Top token row
    _draw_token_row(y_token_row(0), tok_disp)

    # Expert nodes for each hop (fixed x)
    for h in range(H):
        yE = y_expert_row(h)
        for e in e_list:
            col = _expert_color(e)
            ax.add_patch(Rectangle((e_to_x[e] - 0.50, yE - expert_h/2), 1.00, expert_h,
                                   facecolor=col, edgecolor="#0f172a", linewidth=0.45, zorder=5))
            ax.text(e_to_x[e], yE, _expert_label(e),
                    ha="center", va="center", fontsize=7, color="white",
                    weight="bold", family="monospace", zorder=6)

    # Curved connectors token(h) -> expert(h)
    def _curve(x0, y0, x1, y1, col, lw):
        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0 + 0.30
        path = MplPath([(x0, y0), (xm, ym), (x1, y1)],
                       [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=col, linewidth=lw, capstyle="round", zorder=4))

    # Draw only top-k edges per token per hop
    k = max(1, int(show_topk))
    for h in range(H):
        y0 = y_token_row(h) + 0.24
        y1 = y_expert_row(h) - 0.24
        for j in range(Tvis):
            pe = Pn[h, j]  # (E,)
            if pe.sum() <= eps:
                continue
            # top-k for this token
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
        ax.add_patch(Rectangle((j - 0.42, y_pred - token_h/2), 0.84, token_h,
                               facecolor="#d1fae5", edgecolor="#10b981", linewidth=0.9, zorder=8))
        ax.text(j, y_pred, pred_disp[j], ha="center", va="center",
                fontsize=8, family="monospace", color="#065f46", zorder=9)

    # Titles / legend
    ax.text(0.0, y_pred + 0.78, f"Token → Expert Flow per Hop • step {step}",
            fontsize=13, weight="bold", ha="left", va="bottom", transform=ax.transData)
    ax.text(0.0, y_pred + 0.41, f"input preview: {preview}",
            fontsize=9, color="#374151", ha="left", va="bottom", transform=ax.transData)
    ax.text(-0.6, y_pred + 0.10, "Predicted next tokens (greedy):",
            fontsize=9, color="#065f46", ha="left", va="bottom", transform=ax.transData)

    # Legend by module type (+ Other)
    try:
        handles = []
        for tname in list(batch_stats["id_to_type"]):
            cmap = plt.get_cmap(_type_base_cmap(tname))
            rgba = np.array(cmap(0.75))
            handles.append(
                plt.Line2D([0], [0], marker="s", linestyle="",
                           markerfacecolor=rgba, markeredgecolor="none", label=tname)
            )
        if include_other:
            handles.append(
                plt.Line2D([0], [0], marker="s", linestyle="",
                           markerfacecolor=(0.62, 0.62, 0.68), markeredgecolor="none", label="Other")
            )
        if handles:
            ax.legend(handles=handles, title="Expert types",
                      loc="upper right", bbox_to_anchor=(1.0, 1.02),
                      frameon=False, ncol=min(3, len(handles)),
                      fontsize=8, title_fontsize=9)
    except Exception:
        pass

    wandb.log({"routing/token_expert_flow_arrows": wandb.Image(fig), "step": step})
    plt.close(fig)
