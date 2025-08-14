# main.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import time
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from transformers import AutoTokenizer

from dna import DNA, Dense, Attention, FeedForward, Identity
from dna.dataloader import load_dataset_stream, sample_batch
from logs import (
    log_initial_stats,
    log_train_step,
    run_eval_suite,
)

# ------------------------------ config ------------------------------ #


@dataclass
class Config:
    # architecture
    model_type: str = "dna"
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 6
    n_modules: int = 16
    topk: int = 2
    capacity: int = 64
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0

    router_temp: float = 1.0
    select_temp: float = 1.0
    gumbel_tau: float = 1.0

    # data
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: str | None = None
    batch_size: int = 32
    seq_len: int = 256

    # training
    steps: int = 20_000
    warmup: int = 2_000
    lr_peak: float = 3e-4
    wd: float = 0.1
    clip: float = 1.0
    seed: int = 0

    # logging/eval
    wandb_project: str = "dna-model-v2"
    eval_every: int = 100
    log_every: int = 10
    eval_samples: int = 2048
    n_examples: int = 5
    gen_len: int = 200


# ------------------------------ builders ------------------------------ #


def make_modules(
    *,
    d_model: int,
    n_heads: int,
    n_modules: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    n_att = n_modules // 2
    n_ff = n_modules // 2
    # n_id = n_modules // 3
    keys = list(jax.random.split(key, n_modules))
    keys_att = keys[:n_att]
    keys_ff = keys[n_att:]
    attn = [Attention(d_model, n_heads, dropout, key=k) for k in keys_att]
    ffn = [FeedForward(d_model, mlp_mult, dropout, key=k) for k in keys_ff]
    # ident = [Identity() for _ in range(n_id)]
    return tuple(attn + ffn)


def make_backbone(
    *, d_model: int, n_heads: int, mlp_mult: int, dropout: float, key: jax.Array
) -> Tuple[eqx.Module, ...]:
    k1, k2 = jax.random.split(key)
    return (
        Attention(d_model, n_heads, dropout, key=k1),
        FeedForward(d_model, mlp_mult, dropout, key=k2),
    )


def build_model(cfg: Config, key: jax.Array) -> eqx.Module:
    mt = cfg.model_type.lower()
    if mt == "dna":
        km, kb, kmodel = jax.random.split(key, 3)
        modules = make_modules(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_modules=cfg.n_modules,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            key=km,
        )
        backbone = make_backbone(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            key=kb,
        )
        return DNA(
            modules=modules,
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            capacity=cfg.capacity,
            topk=cfg.topk,
            n_hops=cfg.n_hops,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            backbone=backbone,
            key=kmodel,
        )
    elif mt == "dense":
        return Dense(
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_hops,  #
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            key=key,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}' (use 'dna' or 'dense')."
        )


# ------------------------------ training fns ------------------------------ #


def compute_loss(
    model: eqx.Module,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    inference: bool,
    model_kwargs: Dict[str, jax.Array],
):
    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def forward(x, m, k):
        logits, stats = model(x, key=k, inference=inference, mask=m, **model_kwargs)
        return logits, stats

    logits, stats = jax.vmap(forward, in_axes=(0, 0, 0))(ids, mask, keys)
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = mask[:, 1:]
    raw_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_shift, labels_shift
    )
    total_loss = (raw_loss * mask_shift).sum()
    total_tokens = jnp.maximum(mask_shift.sum(), 1.0)
    loss = total_loss / total_tokens
    return loss, (logits_shift, labels_shift, mask_shift, stats)


@eqx.filter_jit
def train_step(
    model: eqx.Module,
    opt: optax.GradientTransformation,
    opt_state,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    model_kwargs: Dict[str, jax.Array],
):
    (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, batch, key, inference=False, model_kwargs=model_kwargs)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    predictions = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    correct = (predictions == labels) & valid
    acc = correct.sum() / jnp.maximum(valid.sum(), 1)
    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


@eqx.filter_jit
def eval_step(
    model: eqx.Module,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    model_kwargs: Dict[str, jax.Array],
) -> Tuple[float, float]:
    loss, (logits, labels, mask, _stats) = compute_loss(
        model, batch, key, inference=True, model_kwargs=model_kwargs
    )
    predictions = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    correct = (predictions == labels) & valid
    acc = correct.sum() / jnp.maximum(valid.sum(), 1)
    return loss, acc


def lr_schedule(
    step: jnp.ndarray, warmup: int, steps: int, lr_peak: float
) -> jnp.ndarray:
    warm = jnp.minimum(step / warmup, 1.0)
    lr = lr_peak * warm
    decay_steps = jnp.maximum(steps - warmup, 1)
    progress = jnp.clip((step - warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step >= warmup, lr_peak * cos, lr).astype(jnp.float32)


# ------------------------------ main ------------------------------ #


def main():
    cfg: Config = tyro.cli(Config)

    run_name = f"{cfg.model_type}-{cfg.dataset_name.split('/')[-1]}-h{cfg.n_hops}"
    if cfg.model_type.lower() == "dna":
        run_name += f"-k{cfg.topk}-c{cfg.capacity}"

    wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

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

    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    model = build_model(cfg, model_key)

    model_kwargs: Dict[str, jax.Array] = {
        "gumbel_tau": jnp.array([cfg.gumbel_tau], dtype=jnp.float32),
        "router_temp": jnp.array([cfg.router_temp], dtype=jnp.float32),
        "select_temp": jnp.array([cfg.select_temp], dtype=jnp.float32),
    }

    first_batch = sample_batch(train_stream, cfg.batch_size)
    log_initial_stats(
        model,
        first_batch,
        seq_len=cfg.seq_len,
        capacity=cfg.capacity,
        topk=cfg.topk,
        n_hops=cfg.n_hops,
        model_type=cfg.model_type,
    )

    schedule_fn = lambda step: lr_schedule(
        jnp.array(step), cfg.warmup, cfg.steps, cfg.lr_peak
    )

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=schedule_fn, b1=0.9, b2=0.95, eps=1e-15, weight_decay=cfg.wd
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    print(f"\nStarting training for {cfg.steps} steps...")
    print("=" * 60)

    for step in range(cfg.steps + 1):
        batch = sample_batch(train_stream, cfg.batch_size)

        key, step_key = jax.random.split(key)
        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt, opt_state, batch, key=step_key, model_kwargs=model_kwargs
        )
        step_time_ms = (time.perf_counter() - t0) * 1000.0

        if step % cfg.log_every == 0:
            log_train_step(
                step=step,
                cfg=cfg,
                schedule_fn=schedule_fn,
                model=model,
                loss=loss,
                acc=acc,
                gnorm=gnorm,
                step_time_ms=step_time_ms,
                stats=stats,
                model_kwargs=model_kwargs,
            )
            if step % (cfg.log_every * 10) == 0:
                print(
                    f"Step {step:5d} | Loss: {float(loss):.4f} | Acc: {float(acc):.4f} | "
                    f"LR: {float(schedule_fn(step)):.2e} | Time: {step_time_ms:.1f}ms"
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


if __name__ == "__main__":
    main()
