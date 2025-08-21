# train.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import json, time
from datetime import datetime

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tyro
import wandb
from transformers import AutoTokenizer
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

from dna import DNA, Attention, FeedForward, Router
from dataset import load_dataset_stream, sample_batch

f32 = jnp.float32


# ------------------------------ config ------------------------------ #


@dataclass
class Config:
    # model
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 8
    topk: int = 2
    dropout: float = 0.0
    rope_base: float = 10_000.0
    n_attn_modules: int = 8
    n_ff_modules: int = 8

    # data
    batch_size: int = 64
    seq_len: int = 256
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None

    # training
    steps: int = 20_000
    warmup: int = 1_000
    lr_peak: float = 3e-4
    wd: float = 1e-3
    clip: float = 1.0
    seed: int = 0

    # routing
    router_temp: float = 1.0
    gumbel_tau: float = 0.0

    # logging/eval
    wandb_project: str = "dna-sharded"
    eval_every: int = 500
    log_every: int = 10
    n_examples: int = 3
    gen_len: int = 200
    eval_samples: int = 1024

    # checkpoints
    save_every: int = 5_000
    ckpt_dir: str = "checkpoints"

    # sharding
    batch_shards: int = 2  # 'data'
    expert_shards: int = 4  # 'expert'


# ------------------------------ mesh & sharding ------------------------------ #


def make_mesh(n_data: int, n_expert: int) -> Mesh:
    devs = jax.devices()
    need = n_data * n_expert
    assert len(devs) >= need, f"need at least {need} devices, got {len(devs)}"
    dm = mesh_utils.create_device_mesh((n_data, n_expert), devices=devs[:need])
    return Mesh(dm, ("data", "expert"))


def shard_expert_params(model: DNA, mesh: Mesh) -> DNA:
    """Shard stacked expert params on 'expert'; replicate everything else."""
    stacked_leaves = []
    for g in model.groups:
        stacked_leaves += jax.tree.leaves(g.params)
    stacked_ids = {id(x) for x in stacked_leaves}

    def place(a):
        if not isinstance(a, jnp.ndarray):
            return a
        if id(a) in stacked_ids:
            spec = NamedSharding(mesh, P("expert", *([None] * (a.ndim - 1))))
        else:
            spec = NamedSharding(mesh, P())
        return jax.device_put(a, spec)

    return jax.tree.map(place, model)


def params_sharding_pytree(params):
    return jax.tree.map(
        lambda x: x.sharding if isinstance(x, jax.Array) else None, params
    )


# ------------------------------ model build ------------------------------ #


def build_model(cfg: Config, key: jax.Array) -> DNA:
    k_mods, k_rtrs, k_bb, k_dna = jax.random.split(key, 4)

    total = cfg.n_attn_modules + cfg.n_ff_modules
    k_list = jax.random.split(k_mods, total)
    mods = [
        Attention(cfg.d_model, cfg.n_heads, cfg.dropout, key=k_list[i])
        for i in range(cfg.n_attn_modules)
    ]
    mods += [
        FeedForward(cfg.d_model, 4, cfg.dropout, key=k_list[cfg.n_attn_modules + i])
        for i in range(cfg.n_ff_modules)
    ]

    routers = tuple(
        Router(cfg.d_model, total, cfg.topk, cfg.dropout, key=k)
        for k in jax.random.split(k_rtrs, cfg.n_hops)
    )

    backbone = (FeedForward(cfg.d_model, 4, cfg.dropout, key=k_bb),)

    return DNA(
        modules=tuple(mods),
        routers=routers,
        vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        rope_base=cfg.rope_base,
        backbone=backbone,
        key=k_dna,
    )


# ------------------------------ schedule ------------------------------ #


def lr_schedule(step, warmup, steps, lr_peak):
    warm = jnp.minimum(step / warmup, 1.0)
    lr = lr_peak * warm
    decay_steps = jnp.maximum(steps - warmup, 1)
    progress = jnp.clip((step - warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step >= warmup, lr_peak * cos, lr).astype(f32)


# ------------------------------ loss (vmapped over B) ------------------------------ #


def loss_and_aux(
    params,
    static,
    batch: Dict[str, jnp.ndarray],
    key,
    *,
    inference: bool,
    model_kwargs: Dict[str, jnp.ndarray],
):
    model = eqx.combine(params, static)
    ids = batch["input_ids"]  # [B,T]
    msk = batch["attention_mask"]  # [B,T]
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def f(x, m, k):
        return model(x, key=k, inference=inference, mask=m, **model_kwargs)  # [T,V]

    logits = jax.vmap(f)(ids, msk, keys)  # [B,T,V]

    # next-token loss
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = msk[:, 1:]
    raw = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift)
    tot = (raw * mask_shift).sum()
    denom = jnp.maximum(mask_shift.sum(), 1.0)
    loss = tot / denom

    return loss, (logits_shift, labels_shift, mask_shift)


# ------------------------------ eval ------------------------------ #


def eval_model(
    params,
    static,
    mesh: Mesh,
    params_in_shardings,
    val_stream,
    *,
    cfg: Config,
    model_kwargs: Dict[str, jnp.ndarray],
    key,
) -> Tuple[float, float]:
    """Average loss/acc over cfg.eval_samples tokens."""
    eval_batches = max(1, cfg.eval_samples // cfg.batch_size)

    def _eval_step(params, batch, key):
        (loss, (logits, labels, mask)) = pjit(
            lambda pr, ids, msk, k: loss_and_aux(
                pr,
                static,
                {"input_ids": ids, "attention_mask": msk},
                k,
                inference=True,
                model_kwargs=model_kwargs,
            ),
            in_shardings=(params_in_shardings, P("data", None), P("data", None), None),
            out_shardings=None,
        )(params, batch["input_ids"], batch["attention_mask"], key)
        acc = ((jnp.argmax(logits, -1) == labels) & (mask > 0)).sum() / jnp.maximum(
            mask.sum(), 1
        )
        return loss, acc

    loss_sum, acc_sum = 0.0, 0.0
    with mesh:
        for _ in range(eval_batches):
            batch_np = sample_batch(val_stream, cfg.batch_size)
            batch = {
                "input_ids": jax.device_put(
                    batch_np["input_ids"], NamedSharding(mesh, P("data", None))
                ),
                "attention_mask": jax.device_put(
                    batch_np["attention_mask"], NamedSharding(mesh, P("data", None))
                ),
            }
            key, sub = jax.random.split(key)
            l, a = _eval_step(params, batch, sub)
            loss_sum += float(l)
            acc_sum += float(a)
    return loss_sum / eval_batches, acc_sum / eval_batches


# ------------------------------ checkpoint ------------------------------ #


def save_ckpt(
    *, run_name: str, cfg: Config, step: int, params, opt_state, lr_value: float
):
    out = Path(cfg.ckpt_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out / f"params_{step}.eqx", params)
    eqx.tree_serialise_leaves(out / f"opt_{step}.eqx", opt_state)
    with open(out / f"config_{step}.json", "w") as f:
        json.dump({**asdict(cfg), "step": step, "lr": float(lr_value)}, f, indent=2)
    print(f"[ckpt] saved at step {step} -> {out}")


# ------------------------------ main ------------------------------ #


def main():
    cfg: Config = tyro.cli(Config)

    # mesh
    assert cfg.batch_size % cfg.batch_shards == 0
    assert cfg.n_attn_modules % cfg.expert_shards == 0
    assert cfg.n_ff_modules % cfg.expert_shards == 0
    mesh = make_mesh(cfg.batch_shards, cfg.expert_shards)
    print(
        "Mesh:",
        mesh.devices.shape,
        "platforms:",
        {d.platform for d in mesh.devices.flat},
    )

    # tokenizer & data
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

    # model
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model = jax.random.split(key)
    model = build_model(cfg, k_model)
    model = shard_expert_params(model, mesh)

    # params/static & shardings
    params, static = eqx.partition(model, eqx.is_inexact_array)
    params_in_shardings = params_sharding_pytree(params)

    # optax
    schedule = lambda step: lr_schedule(step, cfg.warmup, cfg.steps, cfg.lr_peak)
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=cfg.wd
        ),
    )
    opt_state = opt.init(params)

    # router kwargs as arrays (to avoid retraces)
    model_kwargs = {
        "router_temp": jnp.array(cfg.router_temp, dtype=f32),
        "gumbel_tau": jnp.array(cfg.gumbel_tau, dtype=f32),
    }

    # ---------------- pjit-ed train step (updates params) ---------------- #
    def _train_step(pr, os, ids, msk, k):
        (loss, (logits, labels, mask)), grads = eqx.filter_value_and_grad(
            loss_and_aux, has_aux=True
        )(
            pr,
            static,
            {"input_ids": ids, "attention_mask": msk},
            k,
            inference=False,
            model_kwargs=model_kwargs,
        )

        updates, os = opt.update(grads, os, pr)
        pr = eqx.apply_updates(pr, updates)

        # metrics
        acc = ((jnp.argmax(logits, -1) == labels) & (mask > 0)).sum() / jnp.maximum(
            mask.sum(), 1
        )
        gnorm = optax.global_norm(grads)
        wnorm = optax.global_norm(pr)
        return pr, os, loss, acc, gnorm, wnorm

    step_pjit = pjit(
        _train_step,
        in_shardings=(
            params_in_shardings,
            None,
            P("data", None),
            P("data", None),
            None,
        ),
        out_shardings=(params_in_shardings, None, P(), P(), P(), P()),
    )

    # logging
    run_name = f"dna-att{cfg.n_attn_modules}-ff{cfg.n_ff_modules}-h{cfg.n_hops}-k{cfg.topk}-s{cfg.seed}"
    wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))
    global_tokens = 0
    t0_global = time.time()

    with mesh:
        for step in range(cfg.steps + 1):
            t_step = time.perf_counter()

            # batch -> 'data' sharded
            batch_np = sample_batch(train_stream, cfg.batch_size)
            batch = {
                "input_ids": jax.device_put(
                    batch_np["input_ids"], NamedSharding(mesh, P("data", None))
                ),
                "attention_mask": jax.device_put(
                    batch_np["attention_mask"], NamedSharding(mesh, P("data", None))
                ),
            }

            key, k_step = jax.random.split(key)
            params, opt_state, loss, acc, gnorm, wnorm = step_pjit(
                params, opt_state, batch["input_ids"], batch["attention_mask"], k_step
            )

            # tokens/sec etc.
            tokens = int(batch_np["attention_mask"].sum())
            global_tokens += tokens
            if step % cfg.log_every == 0:
                elapsed = time.time() - t0_global
                wandb.log(
                    {
                        "train/loss": float(loss),
                        "train/acc": float(acc),
                        "train/grad_norm": float(gnorm),
                        "train/weight_norm": float(wnorm),
                        "train/lr": float(schedule(step)),
                        "train/tokens_per_sec": global_tokens / max(elapsed, 1e-6),
                        "train/step_ms": (time.perf_counter() - t_step) * 1000,
                    },
                    step=step,
                )
                print(
                    f"step {step:6d} | loss {float(loss):.4f} | acc {float(acc):.4f} | "
                    f"lr {float(schedule(step)):.6f} | t/ms {(time.perf_counter()-t_step)*1000:.1f}"
                )

            if step % cfg.eval_every == 0 and step > 0:
                key, k_eval = jax.random.split(key)
                eval_loss, eval_acc = eval_model(
                    params,
                    static,
                    mesh,
                    params_in_shardings,
                    val_stream,
                    cfg=cfg,
                    model_kwargs=model_kwargs,
                    key=k_eval,
                )
                wandb.log({"eval/loss": eval_loss, "eval/acc": eval_acc}, step=step)
                print(f"[eval] step {step}: loss {eval_loss:.4f} | acc {eval_acc:.4f}")

            if step % cfg.save_every == 0 and step > 0:
                save_ckpt(
                    run_name=run_name,
                    cfg=cfg,
                    step=step,
                    params=params,
                    opt_state=opt_state,
                    lr_value=float(schedule(step)),
                )

    wandb.finish()


if __name__ == "__main__":
    main()
