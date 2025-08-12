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
from utils import (
    lr_schedule,
    count_params,
    l2_tree_norm,
    router_l2_norm,
    batch_seq_stats,
    print_initial_stats,
    generate_examples,
    routing_metrics_from_stats,
    log_routing_visuals,
    log_routing_visuals_single,
)


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
    warmup: int = 2_000
    lr_peak: float = 1e-4
    wd: float = 0.01
    clip: float = 1.0
    seed: int = 42
    eval_every: int = 100
    log_every: int = 10
    eval_samples: int = 16_384
    example_every: int = 250
    n_examples: int = 5
    gen_len: int = 200
    heatmap_every: int = 100


def load_tinystories(tok, seq_len: int, split: str = "train"):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    def _process(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        eos_id = tok.eos_token_id
        for i in range(input_ids.shape[0]):
            row = input_ids[i]
            pos = np.where(row == eos_id)[0]
            if pos.size > 0:
                first_eos = int(pos[0])
                attn_mask[i, first_eos + 1 :] = 0
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    return ds.map(_process, batched=True, batch_size=1024, remove_columns=["text"])


def _normalize_to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape: {arr.shape}")


def sample_batch(stream_it, batch_size: int) -> Dict[str, jnp.ndarray]:
    ids_buffer, mask_buffer, total = [], [], 0
    while total < batch_size:
        example = next(stream_it)
        ids = _normalize_to_2d(example["input_ids"])
        mask = _normalize_to_2d(example["attention_mask"])
        ids_buffer.append(ids)
        mask_buffer.append(mask)
        total += ids.shape[0]
    ids = np.concatenate(ids_buffer, axis=0)[:batch_size].astype(np.int32)
    mask = np.concatenate(mask_buffer, axis=0)[:batch_size].astype(np.int32)
    return {"input_ids": jnp.array(ids), "attention_mask": jnp.array(mask)}


def make_modules(
    *,
    d_model: int,
    n_heads: int,
    n_modules: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    keys = jax.random.split(key, n_modules)
    modules: List[eqx.Module] = []
    for i in range(n_modules):
        t = i % 3
        if t == 0:
            modules.append(Attention(d_model, n_heads, dropout, key=keys[i]))
        elif t == 1:
            modules.append(FeedForward(d_model, mlp_mult, dropout, key=keys[i]))
        else:
            modules.append(Identity())
    return tuple(modules)


def make_backbone(
    *, d_model: int, n_heads: int, mlp_mult: int, dropout: float, key: jax.Array
) -> Tuple[eqx.Module, ...]:
    ff = FeedForward(d_model, mlp_mult, dropout, key=key)
    return (ff,)


def build_model(cfg: Config, key: jax.Array) -> Model:
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
    return Model(
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


def compute_loss(
    model: Model,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    inference: bool,
    model_kwargs: Dict[str, jax.Array],
):
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def forward(x, m, k):
        logits, stats = model(
            x,
            key=k,
            inference=inference,
            attention_mask=m,
            **model_kwargs,
        )
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
    model: Model,
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
    model: Model,
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


def main():
    cfg: Config = tyro.cli(Config)

    wandb.init(
        project="dna-model",
        name=f"dna-h{cfg.n_hops}-k{cfg.topk}-c{cfg.capacity}",
        config=asdict(cfg),
    )

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    train_stream = iter(load_tinystories(tok, cfg.seq_len, "train"))
    val_stream = iter(load_tinystories(tok, cfg.seq_len, "validation"))

    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    model = build_model(cfg, model_key)

    first_batch = sample_batch(train_stream, cfg.batch_size)
    print_initial_stats(
        model,
        first_batch,
        cfg.seq_len,
        cfg.capacity,
        cfg.topk,
        cfg.n_hops,
    )

    schedule_fn = lambda step: lr_schedule(
        jnp.array(step), cfg.warmup, cfg.steps, cfg.lr_peak
    )

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=schedule_fn,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=cfg.wd,
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    def model_kwargs_for_step(step: int) -> Dict[str, jax.Array]:
        return {
            "gumbel_tau": jnp.array([cfg.gumbel_tau], dtype=jnp.float32),
            "router_temperature": jnp.array([cfg.router_temp], dtype=jnp.float32),
            "select_temperature": jnp.array([cfg.select_temp], dtype=jnp.float32),
        }

    print(f"\nStarting training for {cfg.steps} steps...")
    print("=" * 60)

    for step in range(cfg.steps + 1):
        batch = sample_batch(train_stream, cfg.batch_size)
        model_kwargs = model_kwargs_for_step(step)

        key, step_key = jax.random.split(key)
        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt, opt_state, batch, key=step_key, model_kwargs=model_kwargs
        )
        step_time_ms = (time.perf_counter() - t0) * 1000.0

        if step % cfg.log_every == 0:
            stats_host = jax.tree_util.tree_map(jax.device_get, stats)
            rt = float(model_kwargs["router_temperature"][0])
            st = float(model_kwargs["select_temperature"][0])
            gt = float(model_kwargs["gumbel_tau"][0])

            train_metrics = {
                "train/loss": float(loss),
                "train/acc": float(acc),
                "train/grad_norm": float(gnorm),
                "train/lr": float(schedule_fn(step)),
                "train/step_ms": step_time_ms,
                "train/tok_per_sec": cfg.batch_size
                * cfg.seq_len
                / (step_time_ms / 1000.0),
                "weights/global_norm": l2_tree_norm(model),
                "weights/router_norm": router_l2_norm(model),
                "temps/router": rt,
                "temps/select": st,
                "temps/gumbel": gt,
                "step": step,
            }
            routing_metrics = routing_metrics_from_stats(
                stats_host, prefix="train", capacity=cfg.capacity
            )
            wandb.log({**train_metrics, **routing_metrics})

            if step % (cfg.log_every * 10) == 0:
                print(
                    f"Step {step:5d} | Loss: {loss:.4f} | Acc: {acc:.4f} | "
                    f"LR: {schedule_fn(step):.2e} | Time: {step_time_ms:.1f}ms"
                )

        if step % cfg.eval_every == 0 and step > 0:
            key, eval_key = jax.random.split(key)
            eval_batch = sample_batch(val_stream, cfg.eval_samples // cfg.seq_len)
            val_loss, val_acc = eval_step(
                model,
                eval_batch,
                key=eval_key,
                model_kwargs=model_kwargs_for_step(step),
            )
            wandb.log(
                {"eval/loss": float(val_loss), "eval/acc": float(val_acc), "step": step}
            )
            print(f"  [Eval] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if step % cfg.example_every == 0 and step > 0:
            key, gen_key = jax.random.split(key)
            rt = float(model_kwargs["router_temperature"][0])
            st = float(model_kwargs["select_temperature"][0])
            gt = float(model_kwargs["gumbel_tau"][0])
            generate_examples(
                model,
                tok,
                key=gen_key,
                gen_len=cfg.gen_len,
                per_prompt=1,
                router_temp=rt,
                select_temp=st,
                gumbel_tau=gt,
                prompts=None,
                n_examples=cfg.n_examples,
            )

        if step % cfg.heatmap_every == 0 and step > 0:
            key, vis_key = jax.random.split(key)
            vis_batch = sample_batch(train_stream, min(32, cfg.batch_size))
            rt = float(model_kwargs["router_temperature"][0])
            st = float(model_kwargs["select_temperature"][0])
            gt = float(model_kwargs["gumbel_tau"][0])
            _batch_stats, _example_stats = log_routing_visuals(
                model,
                vis_batch,
                key=vis_key,
                gumbel_tau=gt,
                router_temp=rt,
                select_temp=st,
                tok=tok,
                step=step,
                num_examples=4,
                max_tokens_grid=128,
            )

            _flow = log_routing_visuals_single(
                model,
                vis_batch,
                key=vis_key,
                tok=tok,
                step=step,
                model_kwargs=model_kwargs,
                token_max=96,
            )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
