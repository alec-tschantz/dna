# evaluate.py
"""Evaluation utilities for DNA model."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
import matplotlib.pyplot as plt

from dna import Model, sample


# ============================== Evaluation Step ========================== #


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
    """Compute loss for evaluation."""
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
    """Single evaluation step."""
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


# ============================== Heatmap Evaluation ======================= #


@eqx.filter_jit
def evaluate_heatmap(
    model: Model,
    batch: Dict[str, Any],
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate routing patterns for visualization.

    Returns:
    - batch_stats: Averaged routing statistics across the batch
    - example_stats: Routing statistics for individual examples
    """
    ids = batch["input_ids"]  # (B, T)
    mask = batch["attention_mask"]  # (B, T)
    B = ids.shape[0]
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

    logits, stats = jax.vmap(fwd, in_axes=(0, 0, 0))(ids, mask, keys)

    n_hops = len(stats)
    n_experts = (
        len(model.groups) * len(model.groups[0]["idx"])
        if hasattr(model, "groups")
        else 16
    )

    batch_importance = []
    for hop_stats in stats:
        avg_importance = jnp.mean(hop_stats["importance_mean"], axis=0)  # (E,)
        batch_importance.append(avg_importance)

    example_routing = []
    if len(stats) > 0 and "routing_probs" in stats[0]:
        for i in range(min(5, B)):
            ex_routing = []
            for hop_stats in stats:
                probs = hop_stats["routing_probs"][i]  # (T, E)
                ex_routing.append(probs)
            example_routing.append(ex_routing)

    batch_stats = {
        "importance_matrix": jnp.stack(batch_importance),
        "n_hops": n_hops,
        "n_experts": n_experts,
    }

    example_stats = {
        "routing": example_routing,
        "ids": ids[:5],
        "mask": mask[:5],
    }

    return batch_stats, example_stats


def plot_routing_heatmap(batch_stats: Dict[str, Any], step: int):
    """Create and log routing heatmap to wandb."""
    importance = jax.device_get(batch_stats["importance_matrix"])  # (n_hops, n_experts)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(importance.T, aspect="auto", cmap="hot", interpolation="nearest")

    ax.set_xlabel("Hop (Layer)", fontsize=12)
    ax.set_ylabel("Expert Module", fontsize=12)
    ax.set_title(f"Routing Heatmap - Step {step}", fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average Importance", rotation=270, labelpad=20)

    ax.set_xticks(range(importance.shape[0]))
    ax.set_xticklabels([f"Hop {i}" for i in range(importance.shape[0])])

    # TODO: dont assume fixed expert structure
    expert_labels = []
    for i in range(importance.shape[1]):
        if i % 3 == 0:
            expert_labels.append(f"Attn_{i//3}")
        elif i % 3 == 1:
            expert_labels.append(f"FFN_{i//3}")
        else:
            expert_labels.append(f"Id_{i//3}")

    ax.set_yticks(range(importance.shape[1]))
    ax.set_yticklabels(expert_labels, fontsize=8)

    plt.tight_layout()

    wandb.log({"routing/heatmap": wandb.Image(fig), "step": step})
    plt.close(fig)


def print_example_routing(example_stats: Dict[str, Any], tok):
    if not example_stats["routing"]:
        print("No per-token routing probabilities available.")
        print("Note: Model needs modification to return routing_probs in stats.")
        return

    print("\n" + "=" * 60)
    print("Example Routing Patterns")
    print("=" * 60)

    for ex_idx, ex_routing in enumerate(example_stats["routing"][:1]):
        ids = jax.device_get(example_stats["ids"][ex_idx])
        mask = jax.device_get(example_stats["mask"][ex_idx])

        # Decode tokens
        valid_ids = ids[mask > 0]
        tokens = tok.convert_ids_to_tokens(valid_ids.tolist())

        print(f"\n--- Example {ex_idx + 1} ---")
        print(f"Tokens: {' '.join(tokens[:20])}...")  # Show first 20 tokens

        for hop_idx, hop_probs in enumerate(ex_routing):
            hop_probs = jax.device_get(hop_probs)  # (T, E)

            # Show top experts for first few tokens
            print(f"\nHop {hop_idx}:")
            for t_idx in range(min(10, len(valid_ids))):  # First 10 tokens
                if mask[t_idx] > 0:
                    probs = hop_probs[t_idx]
                    top_experts = jnp.argsort(probs)[-3:][::-1]  # Top 3
                    top_probs = probs[top_experts]

                    expert_names = []
                    for e in top_experts:
                        if e % 3 == 0:
                            expert_names.append(f"Attn_{e//3}")
                        elif e % 3 == 1:
                            expert_names.append(f"FFN_{e//3}")
                        else:
                            expert_names.append(f"Id_{e//3}")

                    print(
                        f"  Token {t_idx:2d} '{tokens[t_idx]:15s}': "
                        f"{expert_names[0]}={top_probs[0]:.3f}, "
                        f"{expert_names[1]}={top_probs[1]:.3f}, "
                        f"{expert_names[2]}={top_probs[2]:.3f}"
                    )


# ============================== Text Generation ========================== #


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
    prompts: List[str] | None = None,
    n_examples: int = 5,
) -> None:
    """Generate example text completions."""
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
        prompt_ids = jnp.array(tok.encode(p), dtype=jnp.int32)  # (T0,)
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


# ============================== Routing Metrics ========================== #


def routing_metrics_from_stats(
    stats_tuple_batched: Tuple[Dict[str, Any], ...],
    *,
    prefix: str,
) -> Dict[str, float]:
    """Extract routing metrics from statistics."""
    hop_logs = []

    for hop in stats_tuple_batched:
        # Shapes after vmap: load -> (B, E); scalars -> (B,)
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

        # Utilization per batch: fraction of experts with load>0
        util_b = jnp.mean((load > 0).astype(jnp.float32), axis=1)  # (B,)

        # Load dispersion per batch: std of normalized loads across experts
        denom = jnp.sum(load, axis=1, keepdims=True) + 1e-9
        load_norm = load / denom  # (B, E)
        load_std_b = jnp.std(load_norm, axis=1)  # (B,)

        # Reduce over batch
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
        )
        hop_logs.append(hop_log)

    # Average across hops
    def _mean_key(k: str) -> float:
        vals = [h[k] for h in hop_logs]
        return float(sum(vals) / max(len(vals), 1))

    return {
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
    }
