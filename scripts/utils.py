"""Utilities for DNA model training and evaluation."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import wandb

import numpy as np

from dna import Model, sample


# ============================================================================
# Helper Functions
# ============================================================================


def lr_schedule(
    step: jnp.ndarray, warmup: int, steps: int, lr_peak: float
) -> jnp.ndarray:
    """Cosine learning rate schedule with linear warmup.

    Parameters
    ----------
    step : jnp.ndarray
        Current training step.
    warmup : int
        Number of warmup steps.
    steps : int
        Total training steps.
    lr_peak : float
        Peak learning rate after warmup.

    Returns
    -------
    lr : jnp.ndarray
        Learning rate for current step.
    """
    # Linear warmup
    warm = jnp.minimum(step / warmup, 1.0)
    lr = lr_peak * warm

    # Cosine decay after warmup
    decay_steps = jnp.maximum(steps - warmup, 1)
    progress = jnp.clip((step - warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))

    return jnp.where(step >= warmup, lr_peak * cos, lr).astype(jnp.float32)


def count_params(tree) -> int:
    """Count total parameters in a pytree.

    Parameters
    ----------
    tree : PyTree
        Model or parameter tree.

    Returns
    -------
    n_params : int
        Total number of parameters.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(x.size for x in leaves))


def l2_tree_norm(tree) -> float:
    """Compute L2 norm of all parameters in a pytree.

    Parameters
    ----------
    tree : PyTree
        Model or parameter tree.

    Returns
    -------
    norm : float
        L2 norm of all parameters.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq + 1e-12))


def router_l2_norm(model: Model) -> float:
    """Compute L2 norm of router parameters only.

    Parameters
    ----------
    model : Model
        DNA model instance.

    Returns
    -------
    norm : float
        L2 norm of router parameters.
    """
    if hasattr(model, "routers"):
        leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
        if not leaves:
            return 0.0
        sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
        return float(jnp.sqrt(sq + 1e-12))
    return 0.0


def batch_seq_stats(mask: jnp.ndarray, seq_len: int) -> Tuple[float, int, int, float]:
    """Compute sequence length statistics for a batch.

    Parameters
    ----------
    mask : jnp.ndarray
        Attention mask of shape (B, T).
    seq_len : int
        Maximum sequence length.

    Returns
    -------
    stats : Tuple[float, int, int, float]
        Mean, min, max sequence lengths and mean padding.
    """
    # m = mask[:, 1:]
    lens = jnp.sum(mask, axis=1)  # (B,)

    return (
        float(jnp.mean(lens)),  # mean length
        int(jnp.min(lens)),  # min length
        int(jnp.max(lens)),  # max length
        float(jnp.mean((seq_len - 1) - lens)),  # mean padding
    )


def print_initial_stats(
    model: Model,
    first_batch: Dict[str, Any],
    seq_len: int,
    capacity: int,
    topk: int,
    n_hops: int,
) -> None:
    """Print and log initial model and data statistics.

    Parameters
    ----------
    model : Model
        Initialized DNA model.
    first_batch : Dict[str, Any]
        First training batch for statistics.
    seq_len : int
        Maximum sequence length.
    capacity : int
        Expert capacity.
    topk : int
        Top-k experts per token.
    n_hops : int
        Number of routing hops.
    log_to_wandb : bool
        Whether to log to Weights & Biases.
    """
    n_params = count_params(model)
    lmean, lmin, lmax, pmean = batch_seq_stats(first_batch["attention_mask"], seq_len)

    print("\n" + "=" * 60)
    print("Initial Statistics")
    print("=" * 60)
    print(f"Total parameters: {n_params:,}")
    print(f"Architecture: Capacity={capacity}, TopK={topk}, Hops={n_hops}")
    print(f"Sequence stats (T-1): mean={lmean:.1f}, min={lmin}, max={lmax}")
    print(f"Average padding (T-1): {pmean:.1f}")
    print("=" * 60)

    wandb.log(
        {
            "n_params": n_params,
            "capacity": capacity,
            "topk": topk,
            "hops": n_hops,
            "seq/len_mean": lmean,
            "seq/len_min": lmin,
            "seq/len_max": lmax,
            "seq/pad_mean": pmean,
            "step": 0,
        }
    )


# ============================================================================
# Text Generation
# ============================================================================


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
    """Generate and display example text completions.

    Parameters
    ----------
    model : Model
        DNA model for generation.
    tok : Tokenizer
        Tokenizer instance.
    key : jax.Array
        Random key.
    gen_len : int
        Number of tokens to generate.
    per_prompt : int
        Completions per prompt.
    router_temp : float
        Router temperature.
    select_temp : float
        Selection temperature.
    gumbel_tau : float
        Gumbel tau (unused in inference).
    prompts : Optional[List[str]]
        Custom prompts (uses defaults if None).
    n_examples : int
        Number of prompts to use.
    """
    if prompts is None:
        prompts = [
            "Once upon a time, ",
            "The little robot ",
            "In the magical forest, ",
            "One sunny morning, ",
            "The brave knight ",
        ]

    print("\n" + "=" * 60)
    print("Generated Examples")
    print("=" * 60)

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
            print(f"\nPrompt: {p}")
            print(f"Completion: {text}")
            print("-" * 40)


# ============================================================================
# Expert Metadata Extraction
# ============================================================================


def get_expert_metadata(model: Model) -> Dict[str, Any]:
    """Extract expert metadata from model groups.

    Parameters
    ----------
    model : Model
        DNA model instance.

    Returns
    -------
    metadata : Dict[str, Any]
        Expert labels, types, and mappings.
    """
    assert hasattr(model, "groups") and len(model.groups) > 0, "Model must have groups"

    # Determine total number of experts
    E = 0
    for g in model.groups:
        E = max(E, int(jnp.max(g["idx"]) + 1))

    # Initialize metadata lists
    expert_labels: List[Optional[str]] = [None] * E
    expert_types: List[Optional[str]] = [None] * E

    # Build labels from group information
    type_counters: Dict[str, int] = {}
    for g in model.groups:
        tname = type(g["static"]).__name__
        cnt = type_counters.get(tname, 0)
        idxs = list(map(int, jax.device_get(g["idx"]).tolist()))

        for i, e_idx in enumerate(idxs):
            expert_labels[e_idx] = f"{tname}_{cnt + i}"
            expert_types[e_idx] = tname

        type_counters[tname] = cnt + len(idxs)

    # Fill any missing entries
    for i in range(E):
        if expert_labels[i] is None:
            expert_labels[i] = f"Expert_{i}"
        if expert_types[i] is None:
            expert_types[i] = "Unknown"

    # Create type mappings
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


# ============================================================================
# Metrics Computation
# ============================================================================


def routing_metrics_from_stats(
    stats_tuple_batched: Tuple[Dict[str, Any], ...],
    *,
    prefix: str,
    capacity: Optional[int] = None,
) -> Dict[str, float]:
    """Compute routing metrics from raw statistics.

    This function computes all the aggregated statistics that were
    previously computed in model._stats, plus additional metrics.

    Parameters
    ----------
    stats_tuple_batched : Tuple[Dict[str, Any], ...]
        Raw statistics from each hop (batched).
    prefix : str
        Metric prefix for logging.
    capacity : Optional[int]
        Expert capacity for saturation metrics.

    Returns
    -------
    metrics : Dict[str, float]
        Computed routing metrics.
    """
    hop_logs: List[Dict[str, float]] = []

    for hop in stats_tuple_batched:
        # Extract raw arrays
        load = jnp.asarray(hop["load"])  # (B, E)
        importance = jnp.asarray(hop["importance"])  # (B, E)
        rho = jnp.asarray(hop["rho"])  # (B, T)
        entropy = jnp.asarray(hop["entropy"])  # (B, T)
        selected_edges = jnp.asarray(hop["selected_edges"])  # (B,)
        kept_edges = jnp.asarray(hop["kept_edges"])  # (B,)
        eff_topk = jnp.asarray(hop["eff_topk"])  # (B, T)
        token_mask = jnp.asarray(hop["token_mask"])  # (B, T)

        # Compute per-batch statistics
        B = load.shape[0]

        # Average over valid tokens only
        valid_counts = token_mask.sum(axis=1)  # (B,)

        # Importance statistics
        importance_sum = importance.sum(axis=0)  # (E,)
        importance_mean = jnp.zeros_like(importance_sum)
        total_valid = valid_counts.sum()
        if total_valid > 0:
            # Average importance per expert across all valid tokens
            for b in range(B):
                if valid_counts[b] > 0:
                    importance_mean += importance[b] / total_valid * valid_counts[b]

        # Rho statistics (per batch, then average)
        rho_masked = jnp.where(token_mask, rho, jnp.nan)
        rho_mean = jnp.nanmean(rho_masked, axis=1)  # (B,)
        rho_min = jnp.nanmin(rho_masked, axis=1)  # (B,)
        rho_max = jnp.nanmax(rho_masked, axis=1)  # (B,)

        # Entropy statistics
        entropy_masked = jnp.where(token_mask, entropy, jnp.nan)
        entropy_mean = jnp.nanmean(entropy_masked, axis=1)  # (B,)
        entropy_min = jnp.nanmin(entropy_masked, axis=1)  # (B,)
        entropy_max = jnp.nanmax(entropy_masked, axis=1)  # (B,)

        # Effective top-k statistics
        eff_topk_masked = jnp.where(token_mask, eff_topk.astype(jnp.float32), jnp.nan)
        eff_topk_mean = jnp.nanmean(eff_topk_masked, axis=1)  # (B,)
        eff_topk_min = jnp.nanmin(eff_topk_masked, axis=1)  # (B,)
        eff_topk_max = jnp.nanmax(eff_topk_masked, axis=1)  # (B,)

        # Capacity drop fraction
        cap_drop_frac = jnp.where(
            selected_edges > 0, (selected_edges - kept_edges) / selected_edges, 0.0
        )  # (B,)

        # Expert utilization (fraction with load > 0)
        util = jnp.mean((load > 0).astype(jnp.float32), axis=1)  # (B,)

        # Load standard deviation (normalized)
        denom = jnp.sum(load, axis=1, keepdims=True) + 1e-9
        load_norm = load / denom  # (B, E)
        load_std = jnp.std(load_norm, axis=1)  # (B,)

        # Capacity utilization
        if capacity is not None and capacity > 0:
            cap_util = load.astype(jnp.float32) / jnp.maximum(capacity, 1.0)  # (B, E)
            cap_util_mean = jnp.mean(cap_util, axis=1)  # (B,)
            cap_util_min = jnp.min(cap_util, axis=1)  # (B,)
            cap_util_max = jnp.max(cap_util, axis=1)  # (B,)
            cap_saturated_frac = jnp.mean((load >= capacity).astype(jnp.float32))
        else:
            cap_util_mean = jnp.zeros((B,))
            cap_util_min = jnp.zeros((B,))
            cap_util_max = jnp.zeros((B,))
            cap_saturated_frac = 0.0

        # Additional metrics from routing probabilities
        top1_share = jnp.array(0.0)
        kl_uniform = jnp.array(0.0)

        if "routing_probs" in hop:
            p = jnp.asarray(hop["routing_probs"])  # (B, T, E)

            # Normalize probabilities
            sums = p.sum(axis=-1)
            valid = sums > 1e-9
            p_norm = jnp.where(valid[..., None], p / (sums[..., None] + 1e-9), 0.0)

            # Top-1 share (average max probability)
            t1 = jnp.max(p_norm, axis=-1)
            top1_share = jnp.where(valid, t1, 0.0).sum() / jnp.maximum(valid.sum(), 1)

            # KL divergence from uniform
            E = p.shape[-1]
            kl = (p_norm * jnp.log(p_norm * E + 1e-9)).sum(axis=-1)
            kl_uniform = jnp.where(valid, kl, 0.0).sum() / jnp.maximum(valid.sum(), 1)

        # Store hop-level aggregated metrics
        hop_log = dict(
            importance_mean=float(jnp.mean(importance_mean)),
            rho_mean=float(jnp.mean(rho_mean)),
            rho_min=float(jnp.mean(rho_min)),
            rho_max=float(jnp.mean(rho_max)),
            entropy_mean=float(jnp.mean(entropy_mean)),
            entropy_min=float(jnp.mean(entropy_min)),
            entropy_max=float(jnp.mean(entropy_max)),
            util=float(jnp.mean(util)),
            load_std=float(jnp.mean(load_std)),
            cap_drop_frac=float(jnp.mean(cap_drop_frac)),
            eff_topk_mean=float(jnp.mean(eff_topk_mean)),
            eff_topk_min=float(jnp.mean(eff_topk_min)),
            eff_topk_max=float(jnp.mean(eff_topk_max)),
            cap_util_mean=float(jnp.mean(cap_util_mean)),
            cap_util_min=float(jnp.mean(cap_util_min)),
            cap_util_max=float(jnp.mean(cap_util_max)),
            top1_share_mean=float(top1_share),
            kl_uniform_mean=float(kl_uniform),
            cap_saturated_frac=float(cap_saturated_frac),
        )

        hop_logs.append(hop_log)

    # Average metrics across hops
    def _mean_key(k: str) -> float:
        vals = [h[k] for h in hop_logs if k in h]
        return float(sum(vals) / max(len(vals), 1))

    # Build output metrics dictionary
    metrics = {
        f"routing/{prefix}/importance_mean": _mean_key("importance_mean"),
        f"routing/{prefix}/rho_mean": _mean_key("rho_mean"),
        f"routing/{prefix}/rho_min": _mean_key("rho_min"),
        f"routing/{prefix}/rho_max": _mean_key("rho_max"),
        f"routing/{prefix}/entropy_mean": _mean_key("entropy_mean"),
        f"routing/{prefix}/entropy_min": _mean_key("entropy_min"),
        f"routing/{prefix}/entropy_max": _mean_key("entropy_max"),
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

    return metrics


# ============================================================================
# Visualization and Logging
# ============================================================================


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
    """Evaluate routing patterns for visualization.

    Parameters
    ----------
    model : Model
        DNA model.
    batch : Dict[str, Any]
        Input batch.
    key : jax.Array
        Random key.
    gumbel_tau : float
        Gumbel tau (unused in inference).
    router_temp : float
        Router temperature.
    select_temp : float
        Selection temperature.
    num_examples : int
        Number of examples for detailed analysis.

    Returns
    -------
    batch_stats : Dict[str, Any]
        Batch-level routing statistics.
    example_stats : Dict[str, Any]
        Per-example routing details.
    """
    ids = batch["input_ids"]  # (B, T)
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

    # Forward pass
    _, stats = jax.vmap(fwd, in_axes=(0, 0, 0))(ids, mask, keys)
    n_hops = len(stats)

    # Get expert metadata
    meta = get_expert_metadata(model)
    E = meta["n_experts"]

    # Compute average importance matrix across batch
    batch_importance: List[jnp.ndarray] = []
    for hop_stats in stats:
        # Get importance per expert averaged across batch
        imp = hop_stats["importance"]  # (B, E)
        avg_importance = jnp.mean(imp, axis=0)  # (E,)

        # Pad if needed
        if avg_importance.shape[0] != E:
            pad = jnp.zeros((E,), dtype=avg_importance.dtype)
            pad = pad.at[: avg_importance.shape[0]].set(avg_importance)
            avg_importance = pad

        batch_importance.append(avg_importance)

    importance_matrix = jnp.stack(batch_importance, axis=0)  # (H, E)

    # Select examples for detailed visualization
    S = int(min(num_examples, B))
    key_sel = jax.random.fold_in(key, 42)
    sel_idx = jax.random.choice(key_sel, B, shape=(S,), replace=False)

    # Collect per-example routing details
    probs_by_hop: List[jnp.ndarray] = []
    top1_expert: List[jnp.ndarray] = []
    top1_prob: List[jnp.ndarray] = []
    top1_type_id: List[jnp.ndarray] = []

    for hop_stats in stats:
        if "routing_probs" not in hop_stats:
            raise ValueError("Model stats must include 'routing_probs'")

        probs_bte = hop_stats["routing_probs"]  # (B, T, E)
        probs_ste = jnp.take(probs_bte, sel_idx, axis=0)  # (S, T, E)

        # Normalize probabilities
        sums = probs_ste.sum(axis=-1)  # (S, T)
        valid = sums > 1e-9
        p_norm = jnp.where(valid[..., None], probs_ste / (sums[..., None] + 1e-9), 0.0)

        # Get top-1 expert info
        t1_idx = jnp.argmax(p_norm, axis=-1)  # (S, T)
        t1_val = jnp.take_along_axis(p_norm, t1_idx[..., None], axis=-1)[
            ..., 0
        ]  # (S, T)

        # Map to expert types
        type_ids = meta["expert_type_ids"]  # (E,)
        t1_type = jnp.take(type_ids, t1_idx)  # (S, T)

        probs_by_hop.append(probs_ste)
        top1_expert.append(t1_idx)
        top1_prob.append(t1_val)
        top1_type_id.append(t1_type)

    # Stack across hops
    probs = jnp.stack(probs_by_hop, axis=0)  # (H, S, T, E)
    top1_expert = jnp.stack(top1_expert, axis=0)  # (H, S, T)
    top1_prob = jnp.stack(top1_prob, axis=0)  # (H, S, T)
    top1_type_id = jnp.stack(top1_type_id, axis=0)  # (H, S, T)

    # Get selected examples
    ids_sel = jnp.take(ids, sel_idx, axis=0)  # (S, T)
    mask_sel = jnp.take(mask, sel_idx, axis=0)  # (S, T)

    batch_stats = {
        "importance_matrix": importance_matrix,
        "n_hops": n_hops,
        "n_experts": E,
        "expert_labels": meta["expert_labels"],
        "expert_types": meta["expert_types"],
        "expert_type_ids": meta["expert_type_ids"],
        "type_to_id": meta["type_to_id"],
        "id_to_type": meta["id_to_type"],
    }

    example_stats = {
        "indices": sel_idx,
        "ids": ids_sel,
        "mask": mask_sel,
        "probs": probs,
        "top1_expert": top1_expert,
        "top1_prob": top1_prob,
        "top1_type_id": top1_type_id,
    }

    return batch_stats, example_stats


def plot_routing_heatmap(batch_stats: Dict[str, Any], step: int) -> None:
    """Create and log routing heatmap to Weights & Biases.

    Parameters
    ----------
    batch_stats : Dict[str, Any]
        Batch-level routing statistics.
    step : int
        Current training step.
    """
    importance = jax.device_get(batch_stats["importance_matrix"])  # (H, E)
    expert_labels = batch_stats.get(
        "expert_labels", [f"E{i}" for i in range(importance.shape[1])]
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(importance.T, aspect="auto", cmap="hot", interpolation="nearest")

    ax.set_xlabel("Routing Hop", fontsize=14, fontweight="bold")
    ax.set_ylabel("Expert Module", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Expert Importance Heatmap - Step {step}", fontsize=16, fontweight="bold"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Importance", rotation=270, labelpad=25, fontsize=12)

    # Set ticks
    ax.set_xticks(range(importance.shape[0]))
    ax.set_xticklabels([f"Hop {i}" for i in range(importance.shape[0])], fontsize=11)

    ax.set_yticks(range(importance.shape[1]))
    ax.set_yticklabels(expert_labels, fontsize=10)

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
    """Visualize token routing patterns by expert type.

    Creates a grid showing which expert type each token routes to
    across different hops and examples.

    Parameters
    ----------
    example_stats : Dict[str, Any]
        Per-example routing details.
    batch_stats : Dict[str, Any]
        Batch-level statistics.
    tok : Tokenizer
        Tokenizer for decoding.
    step : int
        Current training step.
    max_tokens : int
        Maximum tokens to visualize.
    """
    top1_type = jax.device_get(example_stats["top1_type_id"])  # (H, S, T)
    top1_prob = jax.device_get(example_stats["top1_prob"])  # (H, S, T)
    ids = jax.device_get(example_stats["ids"])  # (S, T)
    mask = jax.device_get(example_stats["mask"])  # (S, T)

    # Decode text previews
    prompts: List[str] = []
    for s in range(ids.shape[0]):
        valid_ids = ids[s][mask[s] > 0][:max_tokens]
        try:
            preview = tok.decode(valid_ids.tolist(), skip_special_tokens=True)
        except Exception:
            preview = "[Decode Error]"
        prompts.append(preview)

    # Create colormap for expert types
    id_to_type: List[str] = batch_stats.get("id_to_type", [])
    n_types = max(10, len(id_to_type))
    base = plt.get_cmap("tab10")
    colors = [base(i % 10) for i in range(n_types)]
    type_cmap = ListedColormap(colors)

    H, S, T = top1_type.shape
    Tvis = min(T, max_tokens)

    # Create figure with good spacing
    fig_width = max(3.0 * H, 12.0)
    fig_height = max(2.5 * S, 6.0)
    fig, axes = plt.subplots(S, H, figsize=(fig_width, fig_height), squeeze=False)

    for s in range(S):
        for h in range(H):
            ax = axes[s, h]

            # Visualize token types as colored strip
            strip_types = top1_type[h, s, :Tvis][None, :]
            strip_probs = top1_prob[h, s, :Tvis][None, :]

            # Show type colors
            ax.imshow(
                strip_types,
                aspect="auto",
                cmap=type_cmap,
                vmin=0,
                vmax=max(1, len(id_to_type) - 1),
            )

            # Overlay probability as brightness
            ax.imshow(
                strip_probs, aspect="auto", cmap="gray", alpha=0.3, vmin=0.0, vmax=1.0
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if s == 0:
                ax.set_title(f"Hop {h}", fontsize=13, fontweight="bold")

        # Add text preview on the left
        preview = prompts[s]
        truncated = preview[:80] + "..." if len(preview) > 80 else preview
        label = f"Example {s}:\n{truncated}"
        axes[s, 0].set_ylabel(
            label, rotation=0, labelpad=40, va="center", fontsize=11, ha="right"
        )

    # Add legend for expert types
    if id_to_type:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=type_cmap(i))
            for i in range(len(id_to_type))
        ]
        labels = id_to_type
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=11,
            frameon=True,
        )

    plt.suptitle(
        f"Token Routing by Expert Type - Step {step}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.12)

    wandb.log({"routing/token_type_grid": wandb.Image(fig), "step": step})
    plt.close(fig)


def print_example_routing(
    example_stats: Dict[str, Any],
    batch_stats: Dict[str, Any],
    tok,
    *,
    num_tokens: int = 10,
    num_examples: int = 2,
) -> None:
    """Print detailed routing patterns for selected examples.

    Shows top experts and probabilities for each token at each hop,
    formatted for clarity.

    Parameters
    ----------
    example_stats : Dict[str, Any]
        Per-example routing details.
    batch_stats : Dict[str, Any]
        Batch-level statistics.
    tok : Tokenizer
        Tokenizer for decoding.
    num_tokens : int
        Number of tokens to show per example.
    num_examples : int
        Number of examples to print.
    """
    probs = jax.device_get(example_stats["probs"])  # (H, S, T, E)
    ids = jax.device_get(example_stats["ids"])  # (S, T)
    mask = jax.device_get(example_stats["mask"])  # (S, T)

    expert_labels: List[str] = batch_stats.get("expert_labels", [])
    H, S, T, E = probs.shape

    print("\n" + "=" * 80)
    print("DETAILED ROUTING PATTERNS")
    print("=" * 80)

    for s in range(min(S, num_examples)):
        # Get valid tokens
        valid_mask = mask[s] > 0
        valid_ids = ids[s][valid_mask]
        num_valid = len(valid_ids)

        # Decode text
        try:
            text = tok.decode(valid_ids.tolist(), skip_special_tokens=True)
            # Also get individual tokens for display
            tokens = [tok.decode([tid], skip_special_tokens=True) for tid in valid_ids]
        except Exception:
            text = "[Decode Error]"
            tokens = [f"[{tid}]" for tid in valid_ids]

        print(f"\n{'='*60}")
        print(f"Example {s+1}/{S}")
        print(f"{'='*60}")
        print(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(
            f"Tokens ({num_valid} total): {' | '.join(tokens[:20])}{'...' if num_valid > 20 else ''}"
        )

        # Show routing for each hop
        for h in range(H):
            print(f"\n--- Hop {h} ---")

            hop_probs = probs[h, s]  # (T, E)
            shown = min(num_tokens, num_valid)

            # Create a formatted table
            print(
                f"{'Token':<15} {'Top Expert':<20} {'Prob':<8} {'2nd Expert':<20} {'Prob':<8}"
            )
            print("-" * 75)

            for t_idx in range(shown):
                if not valid_mask[t_idx]:
                    continue

                # Get token text (truncate if needed)
                token_text = tokens[t_idx] if t_idx < len(tokens) else f"[{t_idx}]"
                if len(token_text) > 12:
                    token_text = token_text[:12] + "..."

                # Get top experts
                p = hop_probs[t_idx]
                top_idx = jnp.argsort(p)[-3:][::-1]
                top_p = p[top_idx]

                # Get expert names
                names = []
                for e in top_idx:
                    e_int = int(e)
                    if e_int < len(expert_labels):
                        names.append(expert_labels[e_int])
                    else:
                        names.append(f"Expert_{e_int}")

                # Format row
                print(
                    f"{token_text:<15} {names[0]:<20} {top_p[0]:<8.3f} "
                    f"{names[1]:<20} {top_p[1]:<8.3f}"
                )

            if num_valid > shown:
                print(f"... ({num_valid - shown} more tokens)")


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
    """Evaluate routing and log visualizations to W&B.

    Parameters
    ----------
    model : Model
        DNA model.
    batch : Dict[str, Any]
        Input batch.
    key : jax.Array
        Random key.
    gumbel_tau : float
        Gumbel tau.
    router_temp : float
        Router temperature.
    select_temp : float
        Selection temperature.
    tok : Tokenizer
        Tokenizer.
    step : int
        Training step.
    num_examples : int
        Examples for detailed analysis.
    max_tokens_grid : int
        Max tokens in grid visualization.

    Returns
    -------
    batch_stats : Dict[str, Any]
        Batch-level statistics.
    example_stats : Dict[str, Any]
        Per-example statistics.
    """
    # Evaluate routing patterns
    batch_stats, example_stats = evaluate_heatmap(
        model,
        batch,
        key,
        gumbel_tau,
        router_temp,
        select_temp,
        num_examples=num_examples,
    )

    # Create visualizations
    plot_routing_heatmap(batch_stats, step)
    plot_token_type_grid(
        example_stats, batch_stats, tok, step=step, max_tokens=max_tokens_grid
    )

    # Optionally print detailed routing
    if step % 1000 == 0:  # Print less frequently
        print_example_routing(
            example_stats, batch_stats, tok, num_tokens=8, num_examples=1
        )

    return batch_stats, example_stats


def evaluate_token_flow_single(
    model,
    ids_1t: jnp.ndarray,  # (T,)
    mask_1t: jnp.ndarray,  # (T,)
    *,
    key: jax.Array,
    model_kwargs: dict,  # fixed-key dict; values may be (1,) arrays
    topk: int = 2,
):
    logits, stats = model(
        ids_1t,
        key=key,
        inference=True,
        attention_mask=mask_1t,
        **model_kwargs,
    )
    H = len(stats)
    meta = get_expert_metadata(model)  # must contain 'expert_type_ids', 'id_to_type'
    type_ids = jnp.asarray(meta["expert_type_ids"])  # (E,)

    topk_idx, topk_prob, topk_type = [], [], []
    for h in range(H):
        p_te = stats[h]["routing_probs"]  # (T, E)
        idx = jnp.argsort(p_te, axis=-1)[:, -topk:][:, ::-1]  # (T, K) largest→smallest
        prob = jnp.take_along_axis(p_te, idx, axis=-1)  # (T, K)
        typ = jnp.take(type_ids, idx)  # (T, K)
        topk_idx.append(idx)
        topk_prob.append(prob)
        topk_type.append(typ)

    return {
        "topk_idx": jnp.stack(topk_idx, axis=0),  # (H, T, K)
        "topk_prob": jnp.stack(topk_prob, axis=0),  # (H, T, K)
        "topk_type": jnp.stack(topk_type, axis=0),  # (H, T, K)
        "mask": mask_1t,
        "ids": ids_1t,
        "meta": meta,
    }


def plot_token_flow_single(
    flow: dict,
    tok,
    *,
    step: int,
    max_tokens: int = 96,
    title: str = "Token Routing Flow (top-2 per hop)",
    show_probs_as_alpha: bool = False,
):
    # host copies
    topk_idx = np.asarray(flow["topk_idx"])  # (H, T, K)
    topk_type = np.asarray(flow["topk_type"])  # (H, T, K)
    topk_prob = np.asarray(flow["topk_prob"])  # (H, T, K)
    mask = np.asarray(flow["mask"]).astype(bool)  # (T,)
    ids = np.asarray(flow["ids"])
    meta = flow["meta"]

    H, T, K = topk_idx.shape

    valid_pos = np.where(mask)[0]
    if valid_pos.size == 0:
        return
    Tvis = min(valid_pos.shape[0], max_tokens)
    vis_idx = valid_pos[:Tvis]
    ids_vis = ids[vis_idx]

    try:
        preview = tok.decode(ids_vis.tolist(), skip_special_tokens=True)
    except Exception:
        preview = "[decode error]"
    if len(preview) > 120:
        preview = preview[:117] + "..."

    # build grids
    grid_types = np.zeros((H * K, Tvis), dtype=np.int32)
    grid_indices = np.zeros_like(grid_types, dtype=np.int32)
    grid_alpha = np.ones_like(grid_types, dtype=np.float32)

    for h in range(H):
        for r in range(K):  # r=0 top1, r=1 top2
            row = h * K + r
            grid_types[row, :] = topk_type[h, vis_idx, r]
            grid_indices[row, :] = topk_idx[h, vis_idx, r]
            grid_alpha[row, :] = np.clip(topk_prob[h, vis_idx, r], 0.05, 1.0)

    # map type ids → colors (A/F/I). Extend if you have more.
    id_to_type = list(
        meta["id_to_type"]
    )  # e.g., ["Attention","FeedForward","Identity"]
    abbrev = {
        name: (
            "A" if "ttent" in name else "F" if "Feed" in name or "MLP" in name else "I"
        )
        for name in id_to_type
    }
    base = plt.get_cmap("tab10")
    colors = [base(i % 10) for i in range(len(id_to_type))]
    cmap_types = ListedColormap(colors)

    # figure sizing & density-aware text
    fig_h = max(5.0, 0.5 * H * K + 2.0)
    fig_w = max(12.0, 0.12 * Tvis + 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # draw colored type layer
    im = ax.imshow(
        grid_types,
        aspect="auto",
        cmap=cmap_types,
        vmin=0,
        vmax=max(1, len(id_to_type) - 1),
        interpolation="nearest",
    )

    # optional probability alpha overlay (light→low prob, opaque→high prob)
    if show_probs_as_alpha:
        # gray overlay with alpha = 1 - prob, to de-emphasize low-confidence cells
        ax.imshow(
            np.ones_like(grid_alpha), aspect="auto", cmap="gray", alpha=1.0 - grid_alpha
        )

    # overlay module id text: e.g., A3, F7, I1
    # shrink font if many tokens
    show_text = Tvis <= 120
    if show_text:
        # choose white text; if you want dynamic contrast you can add luminance check
        # label from type abbrev + module index
        for row in range(H * K):
            for col in range(Tvis):
                type_name = id_to_type[int(grid_types[row, col])]
                label = f"{abbrev[type_name]}{int(grid_indices[row, col])}"
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7 if Tvis > 80 else 8,
                    color="white",
                )

    # axes styling
    yticks = []
    ylabels = []
    for h in range(H):
        for r in range(K):
            yticks.append(h * K + r)
            ylabels.append(f"Hop {h} • r{r+1}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xticks([])
    ax.set_title(f"{title}\nstep {step} • preview: {preview}", fontsize=13, pad=10)

    # legend for types
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap_types(i)) for i in range(len(id_to_type))
    ]
    fig.legend(
        handles,
        id_to_type,
        loc="upper center",
        ncol=min(len(id_to_type), 6),
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    wandb.log({"routing/token_flow_single": wandb.Image(fig), "step": step})
    plt.close(fig)


def log_routing_visuals_single(
    model,
    batch,
    *,
    key,
    tok,
    step: int,
    model_kwargs: dict,
    token_max: int = 96,
):
    # choose the first valid example (or any strategy you like)
    ids_1t = batch["input_ids"][0]
    mask_1t = batch["attention_mask"][0]
    flow = evaluate_token_flow_single(
        model,
        ids_1t,
        mask_1t,
        key=key,
        model_kwargs=model_kwargs,
        topk=2,
    )
    plot_token_flow_single(flow, tok, step=step, max_tokens=token_max)
    return flow
