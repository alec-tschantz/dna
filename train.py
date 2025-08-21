# train.py
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import tyro
import wandb
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit

from model import DNA, Attention, FeedForward, Identity, Router
from dataloader import setup_tokenizer_and_streams, sample_batch
from utils import (
    lr_schedule,
    build_sample_fn,
    make_mesh,
    shard_expert_params,
    sharding_pytree,
    replicate_scalars_to_mesh,
    save_ckpt,
)


f32 = jnp.float32


# ------------------------------ config ------------------------------ #


@dataclass
class Config:
    # model
    vocab_size: int = 50_257
    d_model: int = 1024
    n_heads: int = 16
    n_hops: int = 8
    topk: int = 1
    dropout: float = 0.2
    rope_base: float = 10_000.0

    n_attn_modules: int = 8
    n_ff_modules: int = 8
    n_id_modules = 8

    # data
    batch_size: int = 128
    seq_len: int = 256
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None

    # training
    steps: int = 20_000
    warmup: int = 1_000
    lr_peak: float = 3e-4
    wd: float = 0.1
    clip: float = 1.0
    seed: int = 0

    # routing
    router_temp: float = 1.0
    gumbel_tau: float = 0.0

    # logging/eval
    wandb_project: str = "dna-tiny-stories"
    eval_every: int = 200
    log_every: int = 10
    gen_len: int = 200
    eval_samples: int = 512

    # checkpoints
    save_every: int = 5_000
    ckpt_dir: str = "checkpoints"

    # sharding
    batch_shards: int = 4  # 'data'
    expert_shards: int = 2  # 'expert'


# ------------------------------ model build ------------------------------ #


def build_model(cfg: Config, key: jax.Array) -> DNA:
    k_mods, k_rtrs, k_bb, k_dna, k_perm = jax.random.split(key, 5)

    total = cfg.n_attn_modules + cfg.n_ff_modules + cfg.n_id_modules
    k_list = jax.random.split(k_mods, cfg.n_attn_modules + cfg.n_ff_modules)
    attns = [
        Attention(cfg.d_model, cfg.n_heads, cfg.dropout, key=k_list[i])
        for i in range(cfg.n_attn_modules)
    ]
    ffs = [
        FeedForward(cfg.d_model, 4, cfg.dropout, key=k_list[cfg.n_attn_modules + i])
        for i in range(cfg.n_ff_modules)
    ]
    ids = [Identity() for _ in range(cfg.n_id_modules)]

    mods = attns + ffs + ids
    perm = list(map(int, jax.random.permutation(k_perm, len(mods)).tolist()))
    mods = [mods[i] for i in perm]

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

    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = msk[:, 1:]

    raw = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift)
    tot = (raw * mask_shift).sum()
    denom = jnp.maximum(mask_shift.sum(), 1.0)
    loss = tot / denom

    return loss, (logits_shift, labels_shift, mask_shift)


# ------------------------------ compiled steps ------------------------------ #


def build_train_step(
    *, static, opt, params_in_shardings, opt_state_in_shardings, model_kwargs
):
    """Compile a single train step that updates params and returns metrics."""

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

        denom = jnp.maximum(mask.sum(), 1)
        acc = ((jnp.argmax(logits, -1) == labels) & (mask > 0)).sum() / denom
        gnorm = optax.global_norm(grads)
        wnorm = optax.global_norm(pr)
        return pr, os, loss, acc, gnorm, wnorm

    return pjit(
        _train_step,
        in_shardings=(
            params_in_shardings,  # params
            opt_state_in_shardings,  # opt_state
            P("data", None),  # ids
            P("data", None),  # mask
            None,  # rng key
        ),
        out_shardings=(
            params_in_shardings,  # updated params
            opt_state_in_shardings,  # updated opt_state
            P(),
            P(),
            P(),
            P(),  # scalars
        ),
    )


def build_eval_step(*, static, params_in_shardings, model_kwargs):
    """Compile a pure eval step that returns (loss, logits, labels, mask)."""

    def _eval_once(pr, ids, msk, k):
        return loss_and_aux(
            pr,
            static,
            {"input_ids": ids, "attention_mask": msk},
            k,
            inference=True,
            model_kwargs=model_kwargs,
        )

    return pjit(
        _eval_once,
        in_shardings=(params_in_shardings, P("data", None), P("data", None), None),
        out_shardings=None,
    )


def build_sample_step(
    *,
    static,
    params_in_shardings,
    max_new: int,
    pad_id: int,
    eos_id: int,
    temperature: float,
):
    scan_sample = build_sample_fn(
        static, max_new=max_new, pad_id=pad_id, eos_id=eos_id, temperature=temperature
    )
    return pjit(
        scan_sample,
        in_shardings=(
            params_in_shardings,  # params
            P("data", None),  # prompt_ids [B, T0]
            P("data"),  # prompt_lens [B]
            None,  # rng
            P(),  # router_temp (scalar)
            P(),  # gumbel_tau (scalar)
        ),
        out_shardings=P("data", None),  # toks [B, T0 + max_new]
    )


# ------------------------------ eval (loop + generation) ------------------------------ #


def eval_model(
    params,
    mesh: Mesh,
    *,
    cfg: Config,
    eval_pjit,
    sample_pjit,
    val_stream,
    tok,
    key,
) -> Tuple[float, float]:
    """Average loss/acc over cfg.eval_samples tokens and print a few generations."""
    eval_batches = max(1, cfg.eval_samples // cfg.batch_size)
    loss_sum, acc_sum = 0.0, 0.0

    with mesh:

        # ------------ eval set -------------
        for _ in range(eval_batches):
            bnp = sample_batch(val_stream, cfg.batch_size)
            batch = {
                "input_ids": jax.device_put(
                    bnp["input_ids"], NamedSharding(mesh, P("data", None))
                ),
                "attention_mask": jax.device_put(
                    bnp["attention_mask"], NamedSharding(mesh, P("data", None))
                ),
            }

            key, sub = jax.random.split(key)
            (loss, (logits, labels, mask)) = eval_pjit(
                params, batch["input_ids"], batch["attention_mask"], sub
            )
            # TODO: return from train step
            denom = jnp.maximum(mask.sum(), 1)
            acc = ((jnp.argmax(logits, -1) == labels) & (mask > 0)).sum() / denom
            loss_sum += float(loss)
            acc_sum += float(acc)

        avg_loss = loss_sum / eval_batches
        avg_acc = acc_sum / eval_batches

        # ------------ generation -------------
        prompts = [
            "once upon a time",
            "the little robot",
            "in a quiet forest",
            "the dog was on fire",
        ]
        pad_id = int(tok.pad_token_id)
        eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else pad_id

        ids_list = [tok.encode(p) for p in prompts]
        lens = jnp.array([len(x) for x in ids_list], dtype=jnp.int32)
        T0 = int(max(lens.tolist()) if len(lens) else 1)
        B = len(ids_list)
        prompt_ids_np = np.full((B, T0), pad_id, dtype=np.int32)
        for i, seq in enumerate(ids_list):
            prompt_ids_np[i, : len(seq)] = seq

        prompt_ids = jax.device_put(prompt_ids_np, NamedSharding(mesh, P("data", None)))
        prompt_lens = jax.device_put(lens, NamedSharding(mesh, P("data")))
        key, sub = jax.random.split(key)
        toks = sample_pjit(
            params,
            prompt_ids,
            prompt_lens,
            sub,
            jnp.asarray(cfg.router_temp, f32),
            jnp.asarray(cfg.gumbel_tau, f32),
        )

        toks = jax.device_get(toks)
        print("\n[eval/generate]")
        for i, p in enumerate(prompts):
            seq = toks[i].tolist()
            if eos_id in seq:
                seq = seq[: seq.index(eos_id) + 1]
            text = tok.decode(seq, skip_special_tokens=True)
            print(f"prompt: {p}\n→ {text}\n")

    return avg_loss, avg_acc


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

    # data
    tok, train_stream, val_stream = setup_tokenizer_and_streams(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        seq_len=cfg.seq_len,
    )

    # model
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model = jax.random.split(key)
    model = build_model(cfg, k_model)
    model = shard_expert_params(model, mesh)

    # params/static & shardings
    params, static = eqx.partition(model, eqx.is_inexact_array)
    params_in_shardings = sharding_pytree(params)

    # optax
    schedule = lambda step: lr_schedule(step, cfg.warmup, cfg.steps, cfg.lr_peak)
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=cfg.wd
        ),
    )

    with mesh:
        opt_state = opt.init(params)
        opt_state = replicate_scalars_to_mesh(opt_state, mesh)
    opt_state_in_shardings = sharding_pytree(opt_state)

    model_kwargs = {
        "router_temp": jnp.array(cfg.router_temp, dtype=f32),
        "gumbel_tau": jnp.array(cfg.gumbel_tau, dtype=f32),
    }

    step_pjit = build_train_step(
        static=static,
        opt=opt,
        params_in_shardings=params_in_shardings,
        opt_state_in_shardings=opt_state_in_shardings,
        model_kwargs=model_kwargs,
    )
    eval_pjit = build_eval_step(
        static=static,
        params_in_shardings=params_in_shardings,
        model_kwargs=model_kwargs,
    )
    sample_pjit = build_sample_step(
        static=static,
        params_in_shardings=params_in_shardings,
        max_new=int(cfg.gen_len),
        pad_id=int(tok.pad_token_id),
        eos_id=(
            int(tok.eos_token_id)
            if tok.eos_token_id is not None
            else int(tok.pad_token_id)
        ),
        temperature=0.8,
    )

    # logging
    run_name = f"dna-att{cfg.n_attn_modules}-ff{cfg.n_ff_modules}-id{cfg.n_id_modules}"
    run_name += f"-h{cfg.n_hops}-k{cfg.topk}-bs{cfg.batch_size}-s{cfg.seed}"
    wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))
    t0_global = time.time()

    # -------------------- main loop -------------------- #
    with mesh:
        for step in range(cfg.steps + 1):
            t_step = time.perf_counter()

            bnp = sample_batch(train_stream, cfg.batch_size)
            batch = {
                "input_ids": jax.device_put(
                    bnp["input_ids"], NamedSharding(mesh, P("data", None))
                ),
                "attention_mask": jax.device_put(
                    bnp["attention_mask"], NamedSharding(mesh, P("data", None))
                ),
            }

            key, k_step = jax.random.split(key)
            params, opt_state, loss, acc, gnorm, wnorm = step_pjit(
                params, opt_state, batch["input_ids"], batch["attention_mask"], k_step
            )

            if step % cfg.log_every == 0:
                wandb.log(
                    {
                        "train/loss": float(loss),
                        "train/acc": float(acc),
                        "train/grad_norm": float(gnorm),
                        "train/weight_norm": float(wnorm),
                        "train/lr": float(schedule(step)),
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
                    mesh,
                    cfg=cfg,
                    eval_pjit=eval_pjit,
                    sample_pjit=sample_pjit,
                    val_stream=val_stream,
                    tok=tok,
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
