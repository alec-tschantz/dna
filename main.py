from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple, List

import time
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from transformers import AutoTokenizer

from dna import DNA, Dense, Attention, FeedForward, Identity
from dna.routing import Router, CosineRouter, SequenceRouter
from dna.dataloader import load_dataset_stream, sample_batch
from logs import log_initial_stats, log_train_step, run_eval_suite, log_checkpoint


# ------------------------------ config ------------------------------ #


@dataclass
class Config:
    # architecture
    model_type: str = "dna"
    router_type: str = "sequence"
    norm_probs: bool = False
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 8
    topk: int = 2
    capacity: int = 64
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0

    # routing temperatures
    router_temp: float = 1.0
    select_temp: float = 1.0
    gumbel_tau: float = 0.0

    # module pool (routed)
    n_att_modules: int = 4
    n_ff_modules: int = 4
    n_id_modules: int = 0

    # backbone
    backbone: Tuple[str] = ("feedforward",)

    # data
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: str | None = None
    batch_size: int = 64
    seq_len: int = 256

    # training
    steps: int = 20_000
    warmup: int = 2000
    lr_peak: float = 2.5e-4
    wd: float = 0.1
    clip: float = 1.0
    seed: int = 0

    # logging/eval
    wandb_project: str = "dna-slurm"
    eval_every: int = 200
    log_every: int = 10
    n_examples: int = 5
    gen_len: int = 200
    eval_samples: int = 8192

    # checkpoints
    save_every: int = 1000
    ckpt_dir: str = "checkpoints"


# ------------------------------ builders ------------------------------ #


def make_modules(
    *,
    d_model: int,
    n_heads: int,
    n_att: int,
    n_ff: int,
    n_id: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    total_param = n_att + n_ff
    keys = list(jax.random.split(key, total_param)) if total_param > 0 else []
    att = [Attention(d_model, n_heads, dropout, key=k) for k in keys[:n_att]]
    ff = [FeedForward(d_model, mlp_mult, dropout, key=k) for k in keys[n_att:]]
    ids = [Identity() for _ in range(n_id)]
    return tuple(att + ff + ids)


def make_backbone(
    *,
    d_model: int,
    n_heads: int,
    mlp_mult: int,
    dropout: float,
    backbone: List[str],
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    """Build backbone exactly as specified in `backbone` list."""
    keys = list(jax.random.split(key, len(backbone))) if backbone else []
    out: List[eqx.Module] = []
    for i, layer_type in enumerate(backbone):
        if layer_type.lower() == "attention":
            out.append(Attention(d_model, n_heads, dropout, key=keys[i]))
        elif layer_type.lower() == "feedforward":
            out.append(FeedForward(d_model, mlp_mult, dropout, key=keys[i]))
        else:
            raise ValueError(f"Unknown backbone layer type '{layer_type}'")
    return tuple(out)


def make_router_cls(router_type: str):
    cfg = {"default": Router, "cosine": CosineRouter, "sequence": SequenceRouter}
    return cfg[router_type]


def build_model(cfg: Config, key: jax.Array) -> eqx.Module:
    if cfg.model_type.lower() == "dna":
        km, kb, kmodel = jax.random.split(key, 3)
        modules = make_modules(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_att=cfg.n_att_modules,
            n_ff=cfg.n_ff_modules,
            n_id=cfg.n_id_modules,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            key=km,
        )
        backbone = make_backbone(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            backbone=cfg.backbone,
            key=kb,
        )
        return DNA(
            modules=modules,
            router_cls=make_router_cls(cfg.router_type),
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            capacity=cfg.capacity,
            topk=cfg.topk,
            n_hops=cfg.n_hops,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            norm_probs=cfg.norm_probs,
            backbone=backbone,
            key=kmodel,
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
def train_step(model, opt, opt_state, batch, *, key, model_kwargs):
    (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, batch, key, inference=False, model_kwargs=model_kwargs)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    predictions = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = ((predictions == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


@eqx.filter_jit
def eval_step(model, batch, *, key, model_kwargs):
    loss, (logits, labels, mask, _) = compute_loss(
        model, batch, key, inference=True, model_kwargs=model_kwargs
    )
    denom = jnp.maximum(mask.sum(), 1)
    acc = ((jnp.argmax(logits, -1) == labels) & (mask > 0)).sum() / denom
    return loss, acc


# def lr_schedule(step, warmup, steps, lr_peak):
#     warm = jnp.minimum(step / warmup, 1.0)
#     lr = lr_peak * warm
#     decay_steps = jnp.maximum(steps - warmup, 1)
#     progress = jnp.clip((step - warmup) / decay_steps, 0.0, 1.0)
#     cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
#     return jnp.where(step >= warmup, lr_peak * cos, lr).astype(jnp.float32)

def lr_schedule(step, warmup, steps, lr_peak):
    warm = jnp.minimum(step / warmup, 1.0)
    lr = lr_peak * warm
    return lr.astype(jnp.float32)

# ------------------------------ main ------------------------------ #


def main():
    cfg: Config = tyro.cli(Config)

    run_name = (
        f"{cfg.model_type}-{cfg.dataset_name.split('/')[-1]}"
        f"-h{cfg.n_hops}-k{cfg.topk}-c{cfg.capacity}-r{cfg.router_type}"
        f"-d{cfg.d_model}-n{cfg.n_att_modules}-n{cfg.n_ff_modules}-i{cfg.n_id_modules}"
        f"-b{cfg.backbone}-s{cfg.seed}-l{cfg.lr_peak}-g{cfg.gumbel_tau}"
    )
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

    model_kwargs = {
        "gumbel_tau": jnp.array([cfg.gumbel_tau], dtype=jnp.float32),
        "router_temp": jnp.array([cfg.router_temp], dtype=jnp.float32),
        "select_temp": jnp.array([cfg.select_temp], dtype=jnp.float32),
    }

    first_batch = sample_batch(train_stream, cfg.batch_size)
    log_initial_stats(model, first_batch, cfg, stream=train_stream)

    schedule_fn = lambda step: lr_schedule(
        jnp.array(step), cfg.warmup, cfg.steps, cfg.lr_peak
    )
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=schedule_fn, b1=0.9, b2=0.95, eps=1e-8, weight_decay=cfg.wd
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    for step in range(cfg.steps + 1):
        batch = sample_batch(train_stream, cfg.batch_size)
        key, step_key = jax.random.split(key)
        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt, opt_state, batch, key=step_key, model_kwargs=model_kwargs
        )
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
                stats=stats,
                model_kwargs=model_kwargs,
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


if __name__ == "__main__":
    main()
