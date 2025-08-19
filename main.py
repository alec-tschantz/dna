# ================================ main.py ================================
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple, List, Optional

import time
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from transformers import AutoTokenizer

from dna import DNA, Dense, ModelKwargs
from dna.routing import RouterRegistry
from dna.dataloader import load_dataset_stream, sample_batch
from logs import (
    log_initial_stats,
    log_train_step,
    log_module_and_router_diagnostics,
    run_eval_suite,
    log_checkpoint,
)


@dataclass
class Config:
    """Training configuration with improved defaults and validation."""

    # Architecture
    model_type: str = "dna"
    router_type: str = "sequence"
    norm_probs: bool = False
    norm_after_capacity: bool = False
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 8
    topk: int = 2
    capacity: int = 64
    mlp_mult: int = 4
    dropout: float = 0.0
    rope_base: float = 10_000.0

    n_att_modules: int = 6
    n_ff_modules: int = 6
    n_id_modules: int = 0

    # Backbone
    backbone: Tuple[str, ...] = ("feedforward",)

    # Data
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    batch_size: int = 64
    seq_len: int = 256

    # Training
    steps: int = 40_000
    warmup: int = 1_000
    lr_peak: float = 3e-4
    wd: float = 0.01
    clip: float = 1.0
    seed: int = 0
    grad_accum: int = 4

    # Routing temperatures
    router_temp: float = 1.0
    select_temp: float = 1.0
    gumbel_tau: float = 0.0

    # Logging/eval
    wandb_project: str = "dna-slurm-v4"
    eval_every: int = 200
    log_every: int = 10
    stats_every: int = 100
    n_examples: int = 5
    gen_len: int = 200
    eval_samples: int = 8192

    # Checkpoints
    save_every: int = 5000
    ckpt_dir: str = "checkpoints"

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert (
            self.d_model // self.n_heads
        ) % 2 == 0, "Head dimension must be even for RoPE"
        assert self.grad_accum >= 1, "grad_accum must be at least 1"


def build_model(cfg: Config, key: jax.Array) -> eqx.Module:
    """Build model based on configuration."""
    if cfg.model_type.lower() == "dna":
        return DNA.from_config(
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_att=cfg.n_att_modules,
            n_ff=cfg.n_ff_modules,
            n_id=cfg.n_id_modules,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            router_type=cfg.router_type,
            capacity=cfg.capacity,
            topk=cfg.topk,
            n_hops=cfg.n_hops,
            norm_probs=cfg.norm_probs,
            norm_after_capacity=cfg.norm_after_capacity,
            backbone=cfg.backbone,
            key=key,
        )
    elif cfg.model_type.lower() == "dense":
        return Dense(
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_hops,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            key=key,
        )
    else:
        raise ValueError(f"Unknown model_type '{cfg.model_type}'")


# Optimized loss computation with better memory usage
@eqx.filter_jit
def compute_loss(
    model: eqx.Module,
    batch: Dict[str, jnp.ndarray],
    key: jax.Array,
    *,
    inference: bool,
    model_kwargs: ModelKwargs,
    return_stats: bool = False,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]]:
    """Compute loss with improved memory efficiency."""
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    # Single vmap for efficiency
    def forward(x, m, k):
        logits, stats = model(
            x,
            key=k,
            inference=inference,
            mask=m,
            **model_kwargs.to_dict(),
            return_stats=return_stats,
        )
        return logits, stats

    logits, stats = jax.vmap(forward)(ids, mask, keys)

    # Compute loss on shifted sequences
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = mask[:, 1:]

    raw_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_shift, labels_shift
    )
    loss = jnp.sum(raw_loss * mask_shift) / jnp.maximum(mask_shift.sum(), 1.0)

    return loss, (logits_shift, labels_shift, mask_shift, stats)


# Optimized gradient accumulation
class GradientAccumulator:
    """Efficient gradient accumulation with minimal tree operations."""

    @staticmethod
    @eqx.filter_jit
    def accumulate_step(
        model: eqx.Module,
        batch: Dict[str, jnp.ndarray],
        accum_state: Tuple[Any, jnp.ndarray, jnp.ndarray],
        key: jax.Array,
        model_kwargs: ModelKwargs,
        return_stats: bool = False,
    ) -> Tuple[Tuple[Any, jnp.ndarray, jnp.ndarray], Any]:
        """Single accumulation step."""
        grads_accum, loss_accum, acc_accum = accum_state

        (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
            compute_loss, has_aux=True
        )(
            model,
            batch,
            key,
            inference=False,
            model_kwargs=model_kwargs,
            return_stats=return_stats,
        )

        # Efficient tree addition
        grads_accum = jax.tree_map(
            lambda a, g: a + g if a is not None else g, grads_accum, grads
        )

        predictions = jnp.argmax(logits, axis=-1)
        valid = mask > 0
        acc = jnp.sum((predictions == labels) & valid) / jnp.maximum(valid.sum(), 1.0)

        return (grads_accum, loss_accum + loss, acc_accum + acc), stats


@eqx.filter_jit
def train_step_single(
    model: eqx.Module,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    key: jax.Array,
    model_kwargs: ModelKwargs,
    return_stats: bool = False,
) -> Tuple[eqx.Module, optax.OptState, jnp.ndarray, jnp.ndarray, Any, jnp.ndarray, Any]:
    """Single training step without accumulation."""
    (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(
        model,
        batch,
        key,
        inference=False,
        model_kwargs=model_kwargs,
        return_stats=return_stats,
    )

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    predictions = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = jnp.sum((predictions == labels) & valid) / jnp.maximum(valid.sum(), 1.0)
    gnorm = optax.global_norm(grads)

    return model, opt_state, loss, acc, stats, gnorm, (grads if return_stats else None)


@eqx.filter_jit
def eval_step(
    model: eqx.Module,
    batch: Dict[str, jnp.ndarray],
    key: jax.Array,
    model_kwargs: ModelKwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
    """Evaluation step."""
    loss, (logits, labels, mask, stats) = compute_loss(
        model, batch, key, inference=True, model_kwargs=model_kwargs, return_stats=True
    )
    acc = jnp.sum((jnp.argmax(logits, -1) == labels) & (mask > 0)) / jnp.maximum(
        mask.sum(), 1.0
    )
    return loss, acc, stats


def create_lr_schedule(warmup: int, steps: int, lr_peak: float):
    """Create learning rate schedule function."""

    def schedule(step):
        step = jnp.array(step, dtype=jnp.float32)
        warm_progress = jnp.minimum(step / warmup, 1.0)

        # Linear warmup
        warmup_lr = lr_peak * warm_progress

        # Cosine decay
        decay_steps = jnp.maximum(steps - warmup, 1)
        decay_progress = jnp.clip((step - warmup) / decay_steps, 0.0, 1.0)
        decay_lr = lr_peak * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))

        return jnp.where(step < warmup, warmup_lr, decay_lr)

    return schedule


def main():
    cfg = tyro.cli(Config)

    # Initialize wandb
    run_name = (
        f"s{cfg.seed}-{cfg.model_type}-{cfg.dataset_name.split('/')[-1]}"
        f"-h{cfg.n_hops}-k{cfg.topk}-c{cfg.capacity}-r{cfg.router_type}"
        f"-d{cfg.d_model}-n{cfg.n_att_modules}-n{cfg.n_ff_modules}-i{cfg.n_id_modules}"
        f"-b{''.join(cfg.backbone)}-wd{cfg.wd}-l{cfg.lr_peak}-g{cfg.gumbel_tau}"
        f"-d{cfg.dropout}-ga{cfg.grad_accum}"
    )
    wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))

    # Setup tokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # Setup data streams
    train_stream = load_dataset_stream(
        cfg.dataset_name,
        tok,
        cfg.seq_len,
        split="train",
        config=cfg.dataset_config,
        seed=cfg.seed,
    )
    val_stream = load_dataset_stream(
        cfg.dataset_name,
        tok,
        cfg.seq_len,
        split="validation",
        config=cfg.dataset_config,
        seed=cfg.seed + 1,
    )

    # Initialize model
    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    model = build_model(cfg, model_key)

    # Model kwargs with type safety
    model_kwargs = ModelKwargs(
        gumbel_tau=cfg.gumbel_tau,
        router_temp=cfg.router_temp,
        select_temp=cfg.select_temp,
    )

    # Log initial stats
    first_batch = sample_batch(train_stream, cfg.batch_size)
    log_initial_stats(model, first_batch, cfg, stream=train_stream)

    # Setup optimizer
    schedule_fn = create_lr_schedule(cfg.warmup, cfg.steps, cfg.lr_peak)
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=schedule_fn, b1=0.9, b2=0.95, eps=1e-8, weight_decay=cfg.wd
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Training loop
    accumulator = GradientAccumulator()

    for step in range(cfg.steps + 1):
        want_stats = step % cfg.stats_every == 0
        t0 = time.perf_counter()

        if cfg.grad_accum == 1:
            # Single step (no accumulation)
            batch = sample_batch(train_stream, cfg.batch_size)
            key, step_key = jax.random.split(key)
            model, opt_state, loss, acc, stats, gnorm, grads = train_step_single(
                model,
                opt,
                opt_state,
                batch,
                key=step_key,
                model_kwargs=model_kwargs,
                return_stats=want_stats,
            )
        else:
            # Gradient accumulation
            accum_state = (None, jnp.array(0.0), jnp.array(0.0))
            stats = None

            for micro in range(cfg.grad_accum):
                batch = sample_batch(train_stream, cfg.batch_size)
                key, step_key = jax.random.split(key)
                return_stats = want_stats and (micro == cfg.grad_accum - 1)

                accum_state, micro_stats = accumulator.accumulate_step(
                    model, batch, accum_state, step_key, model_kwargs, return_stats
                )
                if return_stats:
                    stats = micro_stats

            grads_accum, loss_accum, acc_accum = accum_state
            avg_grads = jax.tree_map(lambda g: g / cfg.grad_accum, grads_accum)

            updates, opt_state = opt.update(avg_grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            loss = loss_accum / cfg.grad_accum
            acc = acc_accum / cfg.grad_accum
            gnorm = optax.global_norm(avg_grads)
            grads = avg_grads if want_stats else None

        # Logging
        if step % cfg.log_every == 0:
            log_train_step(
                step=step,
                cfg=cfg,
                schedule_fn=schedule_fn,
                model=model,
                loss=loss,
                acc=acc,
                gnorm=gnorm,
                step_time_ms=(time.perf_counter() - t0) * 1000.0,
                model_kwargs=model_kwargs,
            )

        if step % cfg.stats_every == 0 and step > 0:
            log_module_and_router_diagnostics(
                model=model, grads=grads, stats_tuple=stats, step=step, prefix="stats"
            )

        if step % cfg.eval_every == 0 and step > 0:
            key = run_eval_suite(
                step=step,
                cfg=cfg,
                model=model,
                eval_step_fn=eval_step,
                val_stream=val_stream,
                key=key,
                tok=tok,
                model_kwargs_train=model_kwargs,
                sample_batch_fn=sample_batch,
            )

        if step % cfg.save_every == 0:
            log_checkpoint(
                run_name=run_name,
                cfg=cfg,
                step=step,
                model=model,
                opt_state=opt_state,
                lr_value=float(schedule_fn(step)),
            )

        wandb.log({"global_step": step}, step=step, commit=True)

    wandb.finish()


if __name__ == "__main__":
    main()