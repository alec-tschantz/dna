# logs.py
"""Generic training/eval utilities for DNA + Dense models."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Patch
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
    ckpt_root = Path(cfg.ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    tag = f"{run_name}_step{step:06d}"
    model_path = ckpt_root / f"{tag}.model.eqx"
    opt_path = ckpt_root / f"{tag}.opt.eqx"
    meta_path = ckpt_root / f"{tag}.meta.json"

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
    print(f"[ckpt] saved: {model_path.name}, {opt_path.name}")


def log_initial_stats(
    model,
    first_batch: Dict[str, Any],
    *,
    cfg: Any = None,
    seq_len: int,
    capacity: Optional[int] = None,
    topk: Optional[int] = None,
    n_hops: Optional[int] = None,
    model_type: str = "dna",
) -> None:
    """Log initial model and data statistics."""
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
    """Log training step metrics."""
    # Core training metrics
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

    # Routing metrics (only if model has routers)
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
    """Run comprehensive evaluation suite including routing analysis."""
    import numpy as np

    # Evaluation loss/accuracy
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

    # Routing visualizations
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

    # Path analysis
    key, path_key = jax.random.split(key)
    path_analysis = analyze_routing_paths(
        model,
        vis_batch,
        tok,
        key=path_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        top_k_paths=15,
        samples_per_path=8,
    )

    if path_analysis is not None:
        plot_path_analysis(path_analysis, step)
        if path_analysis["top_paths"]:
            print(
                f"  [Path Analysis] Unique paths: {path_analysis['unique_paths']}, "
                f"Top path frequency: {path_analysis['top_paths'][0][1]/path_analysis['total_tokens']*100:.1f}%"
            )

    # Module specialization analysis
    key, spec_key = jax.random.split(key)
    specialization_data = analyze_module_specialization(
        model,
        vis_batch,
        tok,
        key=spec_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        n_batches=1,
    )

    if specialization_data is not None:
        plot_module_specialization(specialization_data, step)

        # Print summary statistics
        module_stats = specialization_data["module_stats"]
        diversities = [m.get("token_diversity", 0.0) for m in module_stats.values()]
        if diversities:
            avg_diversity = np.mean(diversities)
            most_used = max(module_stats.items(), key=lambda x: x[1]["total_usage"])
            print(
                f"  [Module Spec] Avg token diversity: {avg_diversity:.3f}, "
                f"Most used: Module {most_used[0]} ({most_used[1]['type']})"
            )

    # Text generation
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
            text = comp["text"]
            preview = text.replace("\n", "\\n")

            print(f"\nPrompt: {p}")
            print(
                f"Completion [{comp['length']} tokens]"
                f"{' (eos)' if comp['stopped_eos'] else ''}:"
            )

            for line in textwrap.wrap(preview, width=100, break_long_words=False):
                print(line)

            print("-" * 40)

    return key


# =============================================================================
# Core utility functions
# =============================================================================


def count_params(tree) -> int:
    """Count total parameters in a model."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(x.size for x in leaves))


def l2_tree_norm(tree) -> float:
    """Compute L2 norm of all parameters."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq + 1e-12))


def batch_seq_stats(mask: jnp.ndarray, seq_len: int) -> Tuple[float, int, int, float]:
    """Compute sequence length statistics from attention mask."""
    lens = jnp.sum(mask, axis=1)
    return (
        float(jnp.mean(lens)),
        int(jnp.min(lens)),
        int(jnp.max(lens)),
        float(jnp.mean(seq_len - lens)),
    )


def has_routing(model) -> bool:
    """Check if model has routing capabilities."""
    return hasattr(model, "routers") and getattr(model, "routers") is not None


def router_l2_norm(model) -> float:
    """Compute L2 norm of router parameters."""
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
    """Compute additional routing metrics from stats."""
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
    """Compute routing metrics from hop statistics."""
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
    """Extract expert metadata from model."""
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
    """Forward pass to collect routing statistics."""

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
    """Get colormap for module type."""
    t = type_name.lower()
    if "ttent" in t:
        return "Blues"
    if "feed" in t or "mlp" in t:
        return "Oranges"
    if "ident" in t:
        return "Greens"
    return "Purples"


def _expert_color_table(meta) -> np.ndarray:
    """Create color table for experts based on their types."""
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
    """Evaluate model and collect visualization data."""
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

    # Collect importance matrix
    importance_matrix = []
    for hop_stats in stats:
        imp_be = hop_stats["importance"]
        avg = jnp.mean(imp_be, axis=0)
        if avg.shape[0] != E:
            pad = jnp.zeros((E,), dtype=avg.dtype).at[: avg.shape[0]].set(avg)
            avg = pad
        importance_matrix.append(avg)
    importance_matrix = jnp.stack(importance_matrix, axis=0)

    # Sample examples for detailed visualization
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
    """Plot expert importance heatmap."""
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

    plt.tight_layout()
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
    """Plot token routing flow visualization."""
    if example_stats is None or batch_stats is None:
        return

    H, S, T = np.asarray(example_stats["top1_expert"]).shape
    ids = np.asarray(example_stats["ids"])
    mask = np.asarray(example_stats["mask"]).astype(bool)

    valid_pos = np.where(mask[0])[0] if S > 0 else np.array([], int)
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

    plt.tight_layout()
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
    """Log routing visualizations if model has routing."""
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
# Advanced routing analysis functions
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
    """Collect top-1 routing decisions for all tokens."""
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
    """Convert RGBA array to plotly color string."""
    r, g, b, a = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(int)
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
    """Create Sankey diagram of token routing paths."""
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

    # Compute transition flows
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

    # Build Sankey nodes and links
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

    # Create Sankey figure
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
            f"Routing Token Paths • step {step} • "
            f"nodes={len(node_labels)} • links={len(link_val)} • "
            f"cov={coverage:.2f}"
        ),
        font=dict(size=12),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    wandb.log({"routing/sankey": fig_sankey, "step": step})

    # Create transition heatmaps
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

        plt.tight_layout()
        wandb.log({"routing/transition_heatmap": wandb.Image(fig_hm), "step": step})
        plt.close(fig_hm)

    return fig_sankey


def analyze_routing_paths(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    top_k_paths: int = 10,
    samples_per_path: int = 5,
):
    """Analyze most common routing paths."""
    if not has_routing(model):
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

    # Create path signatures
    paths = []
    token_ids = []
    positions = []

    ids_np = np.asarray(batch["input_ids"])
    mask_np = np.asarray(mask_bt)

    for b in range(B):
        for t in range(T):
            if not mask_np[b, t]:
                continue

            path = tuple(int(top1[h, b, t]) for h in range(H))
            paths.append(path)
            token_ids.append(int(ids_np[b, t]))
            positions.append((b, t))

    # Count path frequencies
    path_counts = Counter(paths)
    top_paths = path_counts.most_common(top_k_paths)

    # Collect examples for each top path
    path_examples = {}
    for path, count in top_paths:
        path_tokens = [token_ids[i] for i, p in enumerate(paths) if p == path]
        path_positions = [positions[i] for i, p in enumerate(paths) if p == path]

        sample_size = min(samples_per_path, len(path_tokens))
        if sample_size > 0:
            sample_indices = np.random.choice(
                len(path_tokens), sample_size, replace=False
            )
            sampled_tokens = [path_tokens[i] for i in sample_indices]
            sampled_positions = [path_positions[i] for i in sample_indices]

            # Decode tokens
            decoded = []
            for tid in sampled_tokens:
                try:
                    text = tok.decode([tid], skip_special_tokens=True)
                    text = text.replace(" ", "␣") if text else "·"
                    decoded.append(text)
                except:
                    decoded.append("?")

            # Get context for examples
            contexts = []
            for b, t in sampled_positions[:3]:
                start = max(0, t - 2)
                end = min(T, t + 3)
                context_ids = ids_np[b, start:end].tolist()
                try:
                    context_text = tok.decode(context_ids, skip_special_tokens=True)
                    if t - start == 2:
                        tokens = context_text.split()
                        if len(tokens) > 2:
                            tokens[2] = f"[{tokens[2]}]"
                        context_text = " ".join(tokens)
                except:
                    context_text = "?"
                contexts.append(context_text)

            path_examples[path] = {
                "count": count,
                "frequency": count / len(paths) if paths else 0,
                "tokens": sampled_tokens,
                "decoded": decoded,
                "contexts": contexts,
                "expert_types": [meta["expert_types"][e] for e in path],
            }

    return {
        "top_paths": top_paths,
        "path_examples": path_examples,
        "total_tokens": len(paths),
        "unique_paths": len(path_counts),
        "meta": meta,
    }


def plot_path_analysis(path_analysis: Dict[str, Any], step: int):
    """Create path analysis visualizations."""
    if path_analysis is None:
        return

    fig = plt.figure(figsize=(20, 12))

    # Path frequency bar chart
    ax1 = plt.subplot(2, 3, 1)
    paths = []
    counts = []
    for i, (path, count) in enumerate(path_analysis["top_paths"][:10]):
        paths.append(f"Path {i+1}")
        counts.append(count)

    bars = ax1.barh(paths[::-1], counts[::-1], color="steelblue")
    ax1.set_xlabel("Token Count")
    ax1.set_title(f"Top 10 Routing Paths (Step {step})")

    for bar, count in zip(bars, counts[::-1]):
        width = bar.get_width()
        ax1.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Path diversity pie chart
    ax2 = plt.subplot(2, 3, 2)
    top10_count = sum(c for _, c in path_analysis["top_paths"][:10])
    other_count = path_analysis["total_tokens"] - top10_count

    sizes = [top10_count, other_count]
    labels = [
        f"Top 10 paths\n({top10_count:,} tokens)",
        f"Other paths\n({other_count:,} tokens)",
    ]
    colors = ["coral", "lightgray"]

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    ax2.set_title(f'Path Diversity\n({path_analysis["unique_paths"]} unique paths)')

    # Path details table
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis("tight")
    ax3.axis("off")

    table_data = [["Path", "Frequency", "Expert Types", "Example Tokens"]]
    for i, (path, examples) in enumerate(
        list(path_analysis["path_examples"].items())[:5]
    ):
        path_str = "→".join(str(e) for e in path)
        freq = f"{examples['frequency']*100:.1f}%"
        types = "→".join(set(examples["expert_types"]))
        tokens = ", ".join(examples["decoded"][:3])
        table_data.append([path_str, freq, types, tokens])

    table = ax3.table(cellText=table_data, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax3.set_title("Top 5 Path Details", pad=20)

    # Path type composition
    ax4 = plt.subplot(2, 3, 4)
    type_patterns = {}
    for path, examples in list(path_analysis["path_examples"].items())[:10]:
        pattern = tuple(examples["expert_types"])
        type_patterns[pattern] = type_patterns.get(pattern, 0) + examples["count"]

    patterns = list(type_patterns.keys())[:5]
    pattern_counts = [type_patterns[p] for p in patterns]
    pattern_labels = ["→".join(p) for p in patterns]

    bars = ax4.bar(range(len(pattern_counts)), pattern_counts, color="teal")
    ax4.set_xticks(range(len(pattern_labels)))
    ax4.set_xticklabels(pattern_labels, rotation=45, ha="right")
    ax4.set_ylabel("Token Count")
    ax4.set_title("Top Module Type Sequences")

    # Example contexts
    ax5 = plt.subplot(2, 1, 2)
    ax5.axis("off")

    text_content = "Example Token Contexts for Top Paths:\n\n"
    for i, (path, examples) in enumerate(
        list(path_analysis["path_examples"].items())[:3]
    ):
        path_str = "→".join(str(e) for e in path)
        text_content += f"Path {i+1} ({path_str}):\n"
        for j, context in enumerate(examples["contexts"][:2]):
            text_content += f"  • {context}\n"
        text_content += "\n"

    ax5.text(
        0.05,
        0.95,
        text_content,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(f"Routing Path Analysis - Step {step}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    wandb.log({"routing/path_analysis": wandb.Image(fig), "step": step})
    plt.close(fig)


def analyze_module_specialization(
    model,
    batch: Dict[str, Any],
    tok,
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    n_batches: int = 1,
):
    """Analyze module specialization patterns."""
    if not has_routing(model):
        return None

    # Initialize statistics collectors
    module_stats = {}
    all_token_associations = {}

    for batch_idx in range(n_batches):
        top1, mask_bt, meta = _collect_top1_indices_all(
            model,
            batch,
            key=key,
            gumbel_tau=gumbel_tau,
            router_temp=router_temp,
            select_temp=select_temp,
        )

        if top1 is None:
            continue

        H, B, T = top1.shape
        E = meta["n_experts"]

        ids_np = np.asarray(batch["input_ids"])
        mask_np = np.asarray(mask_bt)

        # Initialize module stats if needed
        if not module_stats:
            for e in range(E):
                module_stats[e] = {
                    "hop_usage": np.zeros(H),
                    "total_usage": 0,
                    "type": meta["expert_types"][e],
                    "label": meta["expert_labels"][e],
                    "position_histogram": np.zeros(T),
                    "token_ids": [],
                }
                all_token_associations[e] = []

        # Collect statistics
        for h in range(H):
            for b in range(B):
                for t in range(T):
                    if not mask_np[b, t]:
                        continue

                    expert = int(top1[h, b, t])
                    module_stats[expert]["hop_usage"][h] += 1
                    module_stats[expert]["total_usage"] += 1
                    module_stats[expert]["position_histogram"][t] += 1
                    module_stats[expert]["token_ids"].append(int(ids_np[b, t]))

                    all_token_associations[expert].append(
                        {
                            "token_id": int(ids_np[b, t]),
                            "hop": h,
                            "position": t,
                            "batch": batch_idx,
                        }
                    )

    # Analyze token patterns for each module
    for e in module_stats:
        token_ids = module_stats[e]["token_ids"]
        if token_ids:
            token_counts = Counter(token_ids)
            top_tokens = token_counts.most_common(10)

            decoded_tokens = []
            for tid, count in top_tokens:
                try:
                    text = tok.decode([tid], skip_special_tokens=True)
                    text = text.replace(" ", "␣") if text else "·"
                except:
                    text = "?"
                decoded_tokens.append((text, count))

            module_stats[e]["top_tokens"] = decoded_tokens
            module_stats[e]["token_diversity"] = len(set(token_ids)) / max(
                len(token_ids), 1
            )
        else:
            # Handle modules with no tokens
            module_stats[e]["top_tokens"] = []
            module_stats[e]["token_diversity"] = 0.0

    return {
        "module_stats": module_stats,
        "token_associations": all_token_associations,
        "meta": meta,
        "n_hops": H,
        "n_experts": E,
    }


def plot_module_specialization(specialization_data: Dict[str, Any], step: int):
    """Create module specialization visualizations."""
    if specialization_data is None:
        return

    module_stats = specialization_data["module_stats"]
    n_experts = specialization_data["n_experts"]
    n_hops = specialization_data["n_hops"]

    fig = plt.figure(figsize=(24, 16))

    # Hop specialization heatmap
    ax1 = plt.subplot(3, 4, 1)
    hop_matrix = np.zeros((n_experts, n_hops))
    for e in range(n_experts):
        if module_stats[e]["total_usage"] > 0:
            hop_matrix[e] = (
                module_stats[e]["hop_usage"] / module_stats[e]["total_usage"]
            )

    im1 = ax1.imshow(hop_matrix, aspect="auto", cmap="YlOrRd")
    ax1.set_xlabel("Hop")
    ax1.set_ylabel("Module")
    ax1.set_title("Module Hop Specialization")
    ax1.set_xticks(range(n_hops))
    ax1.set_xticklabels([f"H{i}" for i in range(n_hops)])
    plt.colorbar(im1, ax=ax1, label="Usage Fraction")

    # Module usage distribution
    ax2 = plt.subplot(3, 4, 2)
    usage_counts = [module_stats[e]["total_usage"] for e in range(n_experts)]
    colors_by_type = []
    for e in range(n_experts):
        t = module_stats[e]["type"].lower()
        if "attention" in t:
            colors_by_type.append("skyblue")
        elif "feed" in t or "mlp" in t:
            colors_by_type.append("coral")
        else:
            colors_by_type.append("lightgreen")

    bars = ax2.bar(range(n_experts), usage_counts, color=colors_by_type)
    ax2.set_xlabel("Module ID")
    ax2.set_ylabel("Total Usage Count")
    ax2.set_title("Module Usage Distribution")
    ax2.set_xticks(range(0, n_experts, max(1, n_experts // 8)))

    legend_elements = [
        Patch(facecolor="skyblue", label="Attention"),
        Patch(facecolor="coral", label="FeedForward"),
        Patch(facecolor="lightgreen", label="Other"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")

    # Token diversity per module
    ax3 = plt.subplot(3, 4, 3)
    diversities = [
        module_stats[e].get("token_diversity", 0.0) for e in range(n_experts)
    ]
    scatter = ax3.scatter(
        range(n_experts), diversities, c=usage_counts, cmap="viridis", s=50, alpha=0.6
    )
    ax3.set_xlabel("Module ID")
    ax3.set_ylabel("Token Diversity (unique/total)")
    ax3.set_title("Module Token Specialization")
    plt.colorbar(scatter, ax=ax3, label="Usage Count")

    # Position preference heatmap
    ax4 = plt.subplot(3, 4, 4)
    position_prefs = np.zeros((min(8, n_experts), 32))
    for e in range(min(8, n_experts)):
        hist = module_stats[e]["position_histogram"][:32]
        if hist.sum() > 0:
            position_prefs[e] = hist / hist.sum()

    im4 = ax4.imshow(position_prefs, aspect="auto", cmap="Blues")
    ax4.set_xlabel("Token Position")
    ax4.set_ylabel("Module")
    ax4.set_title("Position Preferences (first 8 modules)")
    plt.colorbar(im4, ax=ax4, label="Preference")

    # Top modules by hop
    for hop_idx in range(min(4, n_hops)):
        ax = plt.subplot(3, 4, 5 + hop_idx)

        hop_usage = [
            (e, module_stats[e]["hop_usage"][hop_idx]) for e in range(n_experts)
        ]
        hop_usage.sort(key=lambda x: x[1], reverse=True)
        top_modules = hop_usage[:10]

        modules = [f"M{e}" for e, _ in top_modules]
        counts = [c for _, c in top_modules]
        colors = []
        for e, _ in top_modules:
            t = module_stats[e]["type"].lower()
            if "attention" in t:
                colors.append("skyblue")
            elif "feed" in t:
                colors.append("coral")
            else:
                colors.append("lightgreen")

        bars = ax.barh(modules[::-1], counts[::-1], color=colors[::-1])
        ax.set_xlabel("Usage Count")
        ax.set_title(f"Top Modules - Hop {hop_idx}")

    # Module type distribution by hop
    ax9 = plt.subplot(3, 4, 9)
    type_by_hop = np.zeros((3, n_hops))

    for e in range(n_experts):
        t = module_stats[e]["type"].lower()
        type_idx = 0 if "attention" in t else (1 if "feed" in t else 2)
        for h in range(n_hops):
            type_by_hop[type_idx, h] += module_stats[e]["hop_usage"][h]

    hop_totals = type_by_hop.sum(axis=0, keepdims=True) + 1e-9
    type_by_hop = type_by_hop / hop_totals

    x = np.arange(n_hops)
    width = 0.6

    ax9.bar(x, type_by_hop[0], width, label="Attention", color="skyblue")
    ax9.bar(
        x,
        type_by_hop[1],
        width,
        bottom=type_by_hop[0],
        label="FeedForward",
        color="coral",
    )
    ax9.bar(
        x,
        type_by_hop[2],
        width,
        bottom=type_by_hop[0] + type_by_hop[1],
        label="Other",
        color="lightgreen",
    )

    ax9.set_xlabel("Hop")
    ax9.set_ylabel("Fraction")
    ax9.set_title("Module Type Distribution by Hop")
    ax9.set_xticks(x)
    ax9.set_xticklabels([f"H{i}" for i in range(n_hops)])
    ax9.legend()

    # Token examples for selected modules
    for i in range(3):
        ax = plt.subplot(3, 4, 10 + i)
        ax.axis("off")

        if i == 0:
            candidates = [
                e
                for e in range(n_experts)
                if "attention" in module_stats[e]["type"].lower()
            ]
        elif i == 1:
            candidates = [
                e for e in range(n_experts) if "feed" in module_stats[e]["type"].lower()
            ]
        else:
            candidates = list(range(n_experts))

        if candidates:
            if i < 2:
                e = max(candidates, key=lambda x: module_stats[x]["total_usage"])
            else:
                e = max(
                    candidates, key=lambda x: module_stats[x].get("token_diversity", 0)
                )

            info = module_stats[e]
            text = f"Module {e} ({info['type']})\n"
            text += f"Total usage: {info['total_usage']}\n"
            text += f"Token diversity: {info.get('token_diversity', 0):.3f}\n"
            text += f"Primary hop: {np.argmax(info['hop_usage']) if info['total_usage'] > 0 else 'N/A'}\n\n"
            text += "Top tokens:\n"

            if "top_tokens" in info and info["top_tokens"]:
                for token, count in info["top_tokens"][:8]:
                    text += f"  '{token}': {count}\n"
            else:
                text += "  No tokens routed to this module\n"

            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.suptitle(
        f"Module Specialization Analysis - Step {step}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    wandb.log({"routing/module_specialization": wandb.Image(fig), "step": step})
    plt.close(fig)
