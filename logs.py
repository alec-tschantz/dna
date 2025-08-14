# metrics.py
"""Generic training/eval utilities for DNA + Dense models."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dna import generate


# =============================================================================
# Public logging API
# =============================================================================


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
    # ---- core training metrics under train/* ----
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

    # ---- routing metrics under router/* (only if model has routers) ----
    if has_routing(model) and isinstance(stats, (tuple, list)) and len(stats) > 0:
        stats_host = jax.tree_util.tree_map(jax.device_get, stats)

        # router norms + temps
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

        # aggregated routing stats
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
    # ---- eval loss/acc (agnostic) ----
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

    # ---- routing visuals if available ----
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
    rows = []

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

            rows.append(
                {
                    "step": step,
                    "prompt": p,
                    "completion": text,
                    "completion_with_special": comp["text_with_special"],
                    "length": comp["length"],
                    "stopped_eos": comp["stopped_eos"],
                }
            )

    return key


# =============================================================================
# Routing metrics/visuals
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
    lens = jnp.sum(mask, axis=1)  # (B,)
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


def _extra_routing_metrics(stats_host, prefix: str) -> Dict[str, float]:
    import numpy as _np

    if not isinstance(stats_host, (tuple, list)) or len(stats_host) == 0:
        return {}
    sel_sum = 0
    kept_sum = 0
    rho_all, mask_all = [], []
    for hop in stats_host:
        if not hop:
            continue
        sel_sum += int(_np.asarray(hop["selected_edges"]).sum())
        kept_sum += int(_np.asarray(hop["kept_edges"]).sum())
        rho_all.append(_np.asarray(hop["rho"]))
        mask_all.append(_np.asarray(hop["token_mask"]))
    if not rho_all:
        return {}
    rho = _np.concatenate(rho_all, axis=1)
    msk = _np.concatenate(mask_all, axis=1)
    rho_valid = rho[msk.astype(bool)]
    dropped_edge_rate = float((sel_sum - kept_sum) / max(sel_sum, 1))
    frac_tokens_rho0 = float((rho_valid <= 1e-9).sum() / max(rho_valid.size, 1))
    p10 = float(_np.percentile(rho_valid, 10.0)) if rho_valid.size else 0.0
    p90 = float(_np.percentile(rho_valid, 90.0)) if rho_valid.size else 0.0
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
    import numpy as _np

    if not isinstance(stats_host, (tuple, list)) or len(stats_host) == 0:
        return {}
    logs = []
    for hop in stats_host:
        if not hop:
            continue
        load = _np.asarray(hop["load"])
        imp = _np.asarray(hop["importance"])
        rho = _np.asarray(hop["rho"])
        ent = _np.asarray(hop["entropy"])
        effk = _np.asarray(hop["eff_topk"])
        mask = _np.asarray(hop["token_mask"])
        rho_m = _np.nanmean(_np.where(mask, rho, _np.nan))
        ent_m = _np.nanmean(_np.where(mask, ent, _np.nan))
        effk_m = _np.nanmean(_np.where(mask, effk, _np.nan))
        util = (load > 0).mean(axis=1).mean() if load.size else 0.0
        denom = load.sum(axis=1, keepdims=True) + 1e-9
        load_std = (load / denom).std(axis=1).mean() if load.size else 0.0
        if capacity and capacity > 0 and load.size:
            cap_util = (load / float(capacity)).mean(axis=1).mean()
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


# ======================= Visuals (only if routers exist) =====================


def _expert_meta(model) -> Optional[Dict[str, Any]]:
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
    import matplotlib.pyplot as plt

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


def _token_strings(tok, ids_1d: np.ndarray) -> list[str]:
    out = []
    for tid in ids_1d.tolist():
        try:
            s = tok.decode([int(tid)], skip_special_tokens=True)
        except Exception:
            s = ""
        s = s.replace(" ", "␣")
        out.append(s if s else "·")
    return out


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
    """Returns (batch_stats, example_stats) or (None, None) if no routing."""
    if not has_routing(model):
        return None, None
    ids = jnp.asarray(batch["input_ids"])  # (B,T)
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

    importance_matrix = []
    for hop_stats in stats:
        imp_be = hop_stats["importance"]  # (B,E)
        avg = jnp.mean(imp_be, axis=0)  # (E,)
        if avg.shape[0] != E:
            pad = jnp.zeros((E,), dtype=avg.dtype).at[: avg.shape[0]].set(avg)
            avg = pad
        importance_matrix.append(avg)
    importance_matrix = jnp.stack(importance_matrix, axis=0)  # (H,E)

    S = int(min(num_examples, B))
    sel_idx = jax.random.choice(
        jax.random.fold_in(key, 123), B, shape=(S,), replace=False
    )

    probs_by_hop, top1_idx, top1_prob, top1_type = [], [], [], []
    for hop_stats in stats:
        p_bte = hop_stats["routing_probs"]  # (B,T,E)
        p_ste = jnp.take(p_bte, sel_idx, axis=0)  # (S,T,E)
        sums = p_ste.sum(axis=-1)
        valid = sums > 1e-9
        p_norm = jnp.where(valid[..., None], p_ste / (sums[..., None] + 1e-9), 0.0)
        idx = jnp.argmax(p_norm, axis=-1)  # (S,T)
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
        probs=jnp.stack(probs_by_hop, axis=0),  # (H,S,T,E)
        top1_expert=jnp.stack(top1_idx, axis=0),  # (H,S,T)
        top1_prob=jnp.stack(top1_prob, axis=0),  # (H,S,T)
        top1_type_id=jnp.stack(top1_type, axis=0),  # (H,S,T)
    )
    return batch_stats, example_stats


def plot_heatmap(batch_stats: Dict[str, Any], step: int) -> None:
    if batch_stats is None:
        return
    imp = jax.device_get(batch_stats["importance_matrix"])  # (H,E)
    labels = batch_stats.get("expert_labels", [f"E{i}" for i in range(imp.shape[1])])
    import matplotlib.pyplot as plt

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
    if example_stats is None or batch_stats is None:
        return
    import numpy as np

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

    import matplotlib.pyplot as plt

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


# --- Sankey: token paths across hops -----------------------------------------


def _rgba_to_plotly(rgba: np.ndarray, alpha: float = 0.8) -> str:
    r, g, b, a = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(int)
    return f"rgba({r},{g},{b},{alpha})"


def _collect_top1_indices_all(
    model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
):
    """Return (top1[H,B,T], mask[B,T], meta). If no routing, returns (None, None, None)."""
    if not has_routing(model):
        return None, None, None

    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"]).astype(bool)  # (B,T)
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
        p = jnp.asarray(hop["routing_probs"])  # (B,T,E)
        idx = jnp.argmax(p, axis=-1)  # (B,T)
        top1_idx.append(idx)
    top1 = jnp.stack(top1_idx, axis=0)  # (H,B,T)

    meta = _expert_meta(model)
    if meta is None:
        return None, None, None

    return jax.device_get(top1), jax.device_get(mask), meta


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
    """Build Sankey and heatmap of token flows across hops; logs both to W&B."""
    import plotly.graph_objects as go

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
        idx_top = np.argpartition(flat, -k)[-k:]
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

    total_links = int(np.sum(link_val))
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
        title=f"Routing Token Paths • step {step} • nodes={len(node_labels)} • links={len(link_val)} • cov={coverage:.2f}",
        font=dict(size=12),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    wandb.log({"routing/sankey": fig_sankey, "step": step})

    ncols = min(3, max(1, len(flows)))
    nrows = int(np.ceil(len(flows) / ncols)) if len(flows) else 1
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
