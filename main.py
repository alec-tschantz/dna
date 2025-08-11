"""Main training script for DNA model."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from dna import Model, Attention, FeedForward, Identity
from dna.eval import (
    eval_step,
    generate_examples,
    routing_metrics_from_stats,
    # new consolidated visual logger
    log_routing_visuals,
)


# ============================== Config =================================== #


@dataclass
class Config:
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 8
    n_modules: int = 16
    topk: int = 2
    capacity: int = 64
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0
    router_temp: float = 1.0
    select_temp: float = 1.0
    gumbel_tau: float = 1.0
    batch_size: int = 64
    seq_len: int = 256
    steps: int = 20_000
    warmup: int = 3_000
    lr_peak: float = 1e-4
    wd: float = 0.01
    clip: float = 1.0
    seed: int = 42
    eval_every: int = 250
    log_every: int = 10
    eval_samples: int = 5_000
    example_every: int = 250
    n_examples: int = 5
    gen_len: int = 100
    heatmap_every: int = 100


cfg: Config = tyro.cli(Config)


# ============================== Data loading ============================= #


def load_tinystories(tok, seq_len: int, split: str = "train"):
    """Load TinyStories dataset."""
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    def _proc(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"]  # (B, T)
        attn_mask = enc["attention_mask"]  # (B, T)
        eos_id = tok.eos_token_id
        for i in range(input_ids.shape[0]):
            row = input_ids[i]
            idx = np.where(row == eos_id)[0]
            if idx.size > 0:
                eos_pos = int(idx[0])
                attn_mask[i, eos_pos + 1 :] = 0
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    return ds.map(_proc, batched=True, batch_size=1024, remove_columns=["text"])


# ============================== Schedules/metrics ======================== #


def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
    """Cosine learning rate schedule with warmup."""
    warm = jnp.minimum(step / cfg.warmup, 1.0)
    lr = cfg.lr_peak * warm
    decay_steps = jnp.maximum(cfg.steps - cfg.warmup, 1)
    progress = jnp.clip((step - cfg.warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step >= cfg.warmup, cfg.lr_peak * cos, lr).astype(jnp.float32)


def count_params(tree) -> int:
    """Count parameters in a pytree."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(x.size for x in leaves))


def l2_tree_norm(tree) -> float:
    """Compute L2 norm of all parameters."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq + 1e-12))


def router_l2_norm(model: Model) -> float:
    """Compute L2 norm of router parameters."""
    if hasattr(model, "routers"):
        leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
        if not leaves:
            return 0.0
        sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
        return float(jnp.sqrt(sq + 1e-12))
    return 0.0


# ============================== Model factory ============================ #


def make_modules(
    *,
    d_model: int,
    n_heads: int,
    n: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    """Create a collection of expert modules."""
    ks = jax.random.split(key, n)
    mods: List[eqx.Module] = []
    for i in range(n):
        t = i % 3
        if t == 0:
            mods.append(Attention(d_model, n_heads, dropout, key=ks[i]))
        elif t == 1:
            mods.append(FeedForward(d_model, mlp_mult, dropout, key=ks[i]))
        else:
            mods.append(Identity())
    return tuple(mods)


def make_backbone(
    *,
    d_model: int,
    n_heads: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    """Create backbone modules (optional pre-routing layers)."""
    # TODO:
    ks = jax.random.split(key, 2)
    attn = Attention(d_model, n_heads, dropout, key=ks[0])
    ff = FeedForward(d_model, mlp_mult, dropout, key=ks[1])
    return (ff,)


def build_model(key: jax.Array) -> Model:
    """Build the complete DNA model."""
    k_mods, k_bb, k_model = jax.random.split(key, 3)
    mods = make_modules(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n=cfg.n_modules,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        key=k_mods,
    )
    backbone = make_backbone(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        key=k_bb,
    )
    return Model(
        modules=mods,
        vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        capacity=cfg.capacity,
        topk=cfg.topk,
        n_hops=cfg.n_hops,
        dropout=cfg.dropout,
        rope_base=cfg.rope_base,
        backbone=backbone,
        key=k_model,
    )


# ============================== Batch helpers =========================== #


def _normalize_to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape {arr.shape}")


def sample_batch(stream_it, bsz: int) -> Dict[str, jnp.ndarray]:
    """Sample a batch from the data stream."""
    ids_buf, mask_buf, total = [], [], 0
    while total < bsz:
        ex = next(stream_it)
        ids = _normalize_to_2d(ex["input_ids"])
        mask = _normalize_to_2d(ex["attention_mask"])
        ids_buf.append(ids)
        mask_buf.append(mask)
        total += ids.shape[0]
    ids = np.concatenate(ids_buf, axis=0)[:bsz].astype(np.int32)  # (B, T)
    mask = np.concatenate(mask_buf, axis=0)[:bsz].astype(np.int32)  # (B, T)
    return {"input_ids": jnp.array(ids), "attention_mask": jnp.array(mask)}


def batch_seq_stats(mask: jnp.ndarray) -> Tuple[float, int, int, float]:
    """Compute sequence length statistics for a batch."""
    m = mask[:, 1:]
    lens = jnp.sum(m, axis=1)  # (B,)
    return (
        float(jnp.mean(lens)),
        int(jnp.min(lens)),
        int(jnp.max(lens)),
        float(jnp.mean((cfg.seq_len - 1) - lens)),
    )


# ============================== Forward / Loss =========================== #


def compute_loss(
    model: Model,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    inference: bool = False,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]]:
    """Compute cross-entropy loss."""
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
            gumbel_tau=cfg.gumbel_tau,
            router_temperature=cfg.router_temp,
            select_temperature=cfg.select_temp,
        )
        return logits, stats

    logits, stats = jax.vmap(fwd, in_axes=(0, 0, 0))(ids, mask, keys)
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = mask[:, 1:]

    raw = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift)
    loss = (raw * mask_shift).sum() / jnp.maximum(mask_shift.sum(), 1)
    return loss, (logits_shift, labels_shift, mask_shift, stats)


# ============================== Training step ============================ #


@eqx.filter_jit

def train_step(
    model: Model,
    opt: optax.GradientTransformation,
    opt_state,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
):
    """Single training step."""
    (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, batch, key, inference=False)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    preds = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = ((preds == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


# ============================== Initial stats ============================ #


def print_initial_stats(
    model: Model,
    first_batch: Dict[str, Any],
    *,
    step0_log_to_wandb: bool = True,
) -> None:
    """Print initial model and data statistics."""
    n_params = count_params(model)
    lmean, lmin, lmax, pmean = batch_seq_stats(first_batch["attention_mask"])
    print("\n" + "=" * 40)
    print("Initial stats")
    print("=" * 40)
    print(f"Params: {n_params:,}")
    print(f"Capacity: {cfg.capacity}  TopK: {cfg.topk}  Hops: {cfg.n_hops}")
    print(f"Seq len mean/min/max (T-1): {lmean:.1f} / {lmin} / {lmax}")
    print(f"Pad mean (T-1): {pmean:.1f}")
    if step0_log_to_wandb:
        wandb.log(
            {
                "n_params": n_params,
                "capacity": cfg.capacity,
                "topk": cfg.topk,
                "hops": cfg.n_hops,
                "seq/len_mean": lmean,
                "seq/len_min": lmin,
                "seq/len_max": lmax,
                "seq/pad_mean": pmean,
                "step": 0,
            }
        )


# ============================== Main training loop ======================= #


def main():
    """Main training loop."""
    wandb.init(project="dna", name="dna", config=asdict(cfg))

    # Initialize tokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # Setup data iterators
    train_it = iter(load_tinystories(tok, cfg.seq_len, "train"))
    val_it = iter(load_tinystories(tok, cfg.seq_len, "validation"))

    # Build model
    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = build_model(mk)

    # Print initial statistics
    first_batch = sample_batch(train_it, cfg.batch_size)
    print_initial_stats(model, first_batch)

    # Setup optimizer
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=cfg.wd,
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Training loop
    for step in range(cfg.steps + 1):
        key, sk = jax.random.split(key)
        batch = sample_batch(train_it, cfg.batch_size)

        # Training step
        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt, opt_state, batch, key=sk
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Logging
        if step % cfg.log_every == 0:
            stats_host = jax.tree_util.tree_map(jax.device_get, stats)

            train_metrics = {
                "train/loss": float(loss),
                "train/acc": float(acc),
                "train/grad_norm": float(gnorm),
                "train/lr": float(lr_schedule(jnp.array(step))),
                "train/step_ms": dt_ms,
                "train/tok_s": cfg.batch_size * cfg.seq_len / (dt_ms / 1000.0 + 1e-9),
                "w_norm/global": l2_tree_norm(model),
                "w_norm/routers": router_l2_norm(model),
                "step": step,
            }

            route_log = routing_metrics_from_stats(stats_host, prefix="train")
            wandb.log({**train_metrics, **route_log})

        # Evaluation
        if step % cfg.eval_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            val_batch = sample_batch(val_it, cfg.eval_samples // cfg.seq_len)
            val_loss, val_acc = eval_step(
                model,
                val_batch,
                key=ek,
                gumbel_tau=cfg.gumbel_tau,
                router_temp=cfg.router_temp,
                select_temp=cfg.select_temp,
            )

            wandb.log(
                {
                    "eval/loss": float(val_loss),
                    "eval/acc": float(val_acc),
                    "step": step,
                }
            )

        # Generate examples
        if step % cfg.example_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            generate_examples(
                model,
                tok,
                key=ek,
                gen_len=cfg.gen_len,
                per_prompt=1,
                router_temp=cfg.router_temp,
                select_temp=cfg.select_temp,
                gumbel_tau=cfg.gumbel_tau,
                prompts=None,  # Will use default prompts
                n_examples=cfg.n_examples,
            )

        # Routing heatmaps & multi-sample comparisons
        if step % cfg.heatmap_every == 0 and step > 0:
            key, hk = jax.random.split(key)
            heatmap_batch = sample_batch(train_it, min(32, cfg.batch_size))
            # Consolidated helper logs the figures to W&B and returns stats
            batch_stats, example_stats = log_routing_visuals(
                model,
                heatmap_batch,
                key=hk,
                gumbel_tau=cfg.gumbel_tau,
                router_temp=cfg.router_temp,
                select_temp=cfg.select_temp,
                tok=tok,
                step=step,
                num_examples=4,
                max_tokens_grid=128,
            )

    
if __name__ == "__main__":
    main()
