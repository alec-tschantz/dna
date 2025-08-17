from typing import Any, Dict, List, Optional, Tuple
import time
from pathlib import Path
from datetime import datetime
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from dna import generate

from .utils import (
    print_ascii_dna,
    ansi_box,
    l2_tree_norm,
    router_l2_norm,
    has_routing,
    routing_metrics_from_stats,
    extra_routing_metrics,
)
from .plots.routing_visuals import log_routing_visuals_if_available
from .plots.flow_arrows import log_token_expert_flow_arrows
from .plots.color_grid import log_token_expert_color_grid
from .plots.sankey import log_routing_sankey_if_available
from .plots.diversity import analyze_path_diversity, plot_path_diversity_dashboard
from .plots.modular import analyze_token_path_specialization, plot_token_path_specialization, log_token_path_sankey
from .plots.histograms import log_router_histograms
from .plots.transitions import log_expert_transition_heatmap, log_type_transition_heatmap
from .plots.temp_sweep import log_temperature_sweep_grid
from .plots.co_usage import log_expert_co_usage_graph
from .plots.capacity_dashboard import log_capacity_saturation_dashboard
from .plots.phase_portrait import log_expert_phase_portrait


def log_checkpoint(
    *,
    run_name: str,
    cfg: Any,
    step: int,
    model: eqx.Module,
    opt_state,
    lr_value: float,
):
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
        "router_type": getattr(cfg, "router_type", None),
        "seq_len": int(cfg.seq_len),
        "batch_size": int(cfg.batch_size),
        "n_hops": int(cfg.n_hops),
        "topk": int(getattr(cfg, "topk", 0)),
        "capacity": int(getattr(cfg, "capacity", 0)),
        "d_model": int(cfg.d_model),
        "n_heads": int(cfg.n_heads),
        "wandb_project": cfg.wandb_project,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [Checkpoint] saved: {model_path}  |  {opt_path}")


def log_initial_stats(
    model, first_batch: Dict[str, Any], cfg: Any, *, stream: Optional[Any] = None
) -> None:

    n_params = sum(
        p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    mask = jnp.asarray(first_batch["attention_mask"])
    seq_len = int(mask.shape[1])
    lengths = mask.sum(axis=1)
    lmean, lmin, lmax = float(lengths.mean()), int(lengths.min()), int(lengths.max())
    pmean = float((seq_len - lengths).mean())
    pad_frac = pmean / max(seq_len, 1)

    lines = [
        f"dataset: {cfg.dataset_name}"
        + (f" [{cfg.dataset_config}]" if getattr(cfg, "dataset_config", None) else ""),
        f"batch_size: {cfg.batch_size} | seq_len: {seq_len}",
        f"avg_tokens: {lmean:.1f} | min: {lmin} | max: {lmax}",
        f"avg_pad: {pmean:.1f} ({pad_frac:.1%})",
        f"params: {n_params:,}",
        f"model_type: {cfg.model_type} | hops: {cfg.n_hops}"
        + (
            f" | topk: {cfg.topk} | capacity: {cfg.capacity}"
            if cfg.model_type.lower() == "dna"
            else ""
        ),
    ]

    from rich.pretty import pprint as rpprint
    rpprint(vars(cfg))
    rpprint(eqx.filter(model, eqx.is_array))

    print_ascii_dna()
    print(ansi_box("Dataset & Model Summary", lines, width=66))

    wandb.log(
        {
            "init/n_params": n_params,
            "init/seq_len": seq_len,
            "init/seq_len_mean": lmean,
            "init/seq_len_min": lmin,
            "init/seq_len_max": lmax,
            "init/pad_mean": pmean,
            "step": 0,
        }
    )


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
    """Log core training metrics, with optional router metrics if available."""
    tok_per_sec = (cfg.batch_size * cfg.seq_len) / max(step_time_ms / 1000.0, 1e-9)
    logs = {
        "step": int(step),
        "train/loss": float(loss),
        "train/acc": float(acc),
        "train/grad_norm": float(gnorm),
        "train/lr": float(schedule_fn(step)),
        "train/tok_per_sec": float(tok_per_sec),
        "train/weights_global_norm": float(l2_tree_norm(model)),
    }

    if has_routing(model) and isinstance(stats, (tuple, list)) and len(stats) > 0:
        stats_host = jax.tree_util.tree_map(jax.device_get, stats)
        logs.update(
            {
                "router/norm": float(router_l2_norm(model)),
                "router/temp/router": float(
                    model_kwargs.get("router_temp", jnp.array([0.0]))[0]
                ),
                "router/temp/select": float(
                    model_kwargs.get("select_temp", jnp.array([0.0]))[0]
                ),
                "router/temp/gumbel": float(
                    model_kwargs.get("gumbel_tau", jnp.array([0.0]))[0]
                ),
            }
        )
        logs.update(
            routing_metrics_from_stats(
                stats_host,
                prefix="router/train",
                capacity=getattr(cfg, "capacity", None),
            )
        )
        logs.update(extra_routing_metrics(stats_host, prefix="router/train"))

    # --- Console ---
    print(
        f"  [Train] Step: {step} | "
        f"Loss: {float(loss):.4f} | "
        f"Acc: {float(acc):.4f} | "
        f"LR: {logs['train/lr']:.5f} | "
        f"T/s: {tok_per_sec:,.0f}"
    )

    wandb.log(logs)



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
    val_loss, val_acc, eval_stats = eval_step_fn(
        model, eval_batch, key=eval_key, model_kwargs=eval_kwargs
    )
    
    eval_logs = {"eval/loss": float(val_loss), "eval/acc": float(val_acc), "step": step}
    
    # Log eval routing stats if available
    if has_routing(model) and isinstance(eval_stats, (tuple, list)) and len(eval_stats) > 0:
        stats_host = jax.tree_util.tree_map(jax.device_get, eval_stats)
        eval_logs.update(
            routing_metrics_from_stats(
                stats_host,
                prefix="router/eval",
                capacity=getattr(cfg, "capacity", None),
            )
        )
        eval_logs.update(extra_routing_metrics(stats_host, prefix="router/eval"))

        # NEW: histograms/distributions
        log_router_histograms(stats_host, step=step, prefix="router/eval")

    wandb.log(eval_logs)
    print(f"  [Eval] Loss: {float(val_loss):.4f} | Acc: {float(val_acc):.4f}")

    # Existing visuals
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
    key, arrows_key = jax.random.split(key)
    log_token_expert_flow_arrows(
        model,
        vis_batch,
        tok,
        key=arrows_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        max_tokens=64,
    )

    key, color_key = jax.random.split(key)
    log_token_expert_color_grid(
        model,
        vis_batch,
        tok,
        key=color_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        max_tokens=96,
        alpha_by_conf=True,
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
        log_token_path_sankey(spec, step)

    # NEW: expert transition heatmaps (expert→expert, type→type)
    key, tkey1 = jax.random.split(key)
    log_expert_transition_heatmap(
        model,
        vis_batch,
        key=tkey1,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        top_experts=24,
    )
    key, tkey2 = jax.random.split(key)
    log_type_transition_heatmap(
        model,
        vis_batch,
        key=tkey2,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
    )

    # NEW: tiny temperature sweep grid (quick)
    key = log_temperature_sweep_grid(
        model=model,
        eval_step_fn=eval_step_fn,
        val_stream=val_stream,
        key=key,
        tok=tok,
        cfg=cfg,
        base_router_temp=float(eval_kwargs["router_temp"][0]),
        base_select_temp=float(eval_kwargs["select_temp"][0]),
        sample_batch_fn=sample_batch_fn,
        step=step,
        r_scales=(0.7, 1.0, 1.3),
        s_scales=(0.7, 1.0, 1.3),
        batch_tokens=8192,
    )

     # --- NEW: Expert co-usage curved network (same-token, same-hop) ---
    key, cousage_key = jax.random.split(key)
    log_expert_co_usage_graph(
        model,
        vis_batch,
        key=cousage_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        top_experts=24,        # tweak as desired
        min_edge_frac=0.04,    # drop very thin edges
    )

    # --- NEW: Capacity pressure dashboard (drop fractions, totals, utilization) ---
    key, cap_key = jax.random.split(key)
    log_capacity_saturation_dashboard(
        model,
        vis_batch,
        key=cap_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        capacity=int(getattr(cfg, "capacity", 1)),  # uses your cfg.capacity
        top_experts=36,
    )

    # --- NEW: Phase portrait (kept-rate vs top-1 confidence, size by usage) ---
    key, portrait_key = jax.random.split(key)
    log_expert_phase_portrait(
        model,
        vis_batch,
        key=portrait_key,
        gumbel_tau=0.0,
        router_temp=float(eval_kwargs["router_temp"][0]),
        select_temp=float(eval_kwargs["select_temp"][0]),
        step=step,
        top_experts=42,
    )

    # Text generation (unchanged)
    key, gen_key = jax.random.split(key)
    prompts = [
        "Once upon a time, ",
        "The little robot ",
        "In the magical forest, ",
        "One sunny morning, ",
        "The brave knight ",
    ]
    temps = {
        "router_temp": float(eval_kwargs["router_temp"][0]),
        "select_temp": float(eval_kwargs["select_temp"][0]),
        "gumbel_tau": 0.0,
    }
    results = generate(
        model,
        tok,
        key=gen_key,
        gen_len=cfg.gen_len,
        per_prompt=1,
        router_temp=temps["router_temp"],
        select_temp=temps["select_temp"],
        gumbel_tau=temps["gumbel_tau"],
        prompts=prompts,
        n_examples=cfg.n_examples,
    )

    for r in results:
        prompt = r["prompt"]
        comp = r["completions"][0]
        text = comp["text"].replace("\n", " ")
        lines = [
            f"len={comp['length']} | eos={bool(comp.get('stopped_eos', False))}",
            "",
            text,
        ]
        print(ansi_box(f"Prompt: {prompt}", lines, width=88))
        print()

    return key