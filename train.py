# train.py
from __future__ import annotations

import os
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
from datasets import load_dataset
from transformers import AutoTokenizer
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.experimental.multihost_utils import sync_global_devices

from model import DNA, Attention, FeedForward, Router
from dataloader import setup_tokenizer_and_streams, sample_batch

f32 = jnp.float32


# ------------------------------ config ------------------------------ #


@dataclass
class Config:
    # model
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 8
    n_hops: int = 12
    topk: int = 2
    dropout: float = 0.2
    rope_base: float = 10_000.0
    n_attn_modules: int = 12
    n_ff_modules: int = 12

    # data
    batch_size: int = 128
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
    wandb_project: str = "dna-tiny-stories"
    eval_every: int = 500
    log_every: int = 10
    gen_len: int = 200
    eval_samples: int = 512

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


def _sharding_pytree(params):
    return jax.tree.map(
        lambda x: x.sharding if isinstance(x, jax.Array) else None, params
    )


def _replicate_scalars_to_mesh(tree, mesh):
    """Place all scalar jax.Arrays onto the mesh with full replication."""

    def place(x):
        if isinstance(x, jax.Array) and x.shape == ():  # scalar leaf
            return jax.device_put(x, NamedSharding(mesh, P()))
        return x

    return jax.tree.map(place, tree)


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


def _scan_sample_fn(
    static, *, max_new: int, pad_id: int, eos_id: int, temperature: float
):
    """Return a pjit-friendly sampler using lax.scan with static shape."""

    def _scan_sample(
        params,
        prompt_ids: jnp.ndarray,  # [T0]
        key: jax.Array,
        router_temp: jnp.ndarray,  # scalar f32
        gumbel_tau: jnp.ndarray,  # scalar f32
    ) -> jnp.ndarray:  # [T0 + max_new]
        model = eqx.combine(params, static)

        # T0 is part of the prompt_ids shape and thus static per-compile
        t0 = prompt_ids.shape[0]
        total_len = t0 + max_new  # max_new is STATIC via closure

        toks = jnp.full((total_len,), pad_id, dtype=jnp.int32)
        toks = toks.at[:t0].set(prompt_ids)

        force_greedy = jnp.asarray(temperature, f32) <= 0

        def step(carry, _):
            toks, cur, done, k = carry
            k, sub = jax.random.split(k)

            attn_mask = jnp.arange(total_len, dtype=jnp.int32) < cur

            logits = model(
                toks,
                key=sub,
                inference=True,
                mask=attn_mask,
                router_temp=router_temp,
                gumbel_tau=gumbel_tau,
            )  # [T,V]

            vocab = logits.shape[-1]
            cur_idx = cur - jnp.asarray(1, jnp.int32)
            last_logits = jax.lax.dynamic_slice(logits, (cur_idx, 0), (1, vocab))[0]

            def pick_greedy(lg):
                return jnp.argmax(lg, axis=-1).astype(jnp.int32)

            def pick_sample(rng, lg):
                scaled = lg / jnp.clip(jnp.asarray(temperature, f32), 1e-6, None)
                return jax.random.categorical(rng, scaled).astype(jnp.int32)

            nxt = jax.lax.cond(
                force_greedy,
                lambda _: pick_greedy(last_logits),
                lambda rng: pick_sample(rng, last_logits),
                operand=sub,
            )
            nxt = jax.lax.select(done, jnp.asarray(pad_id, jnp.int32), nxt)

            toks = toks.at[cur].set(nxt)
            done = done | (nxt == eos_id)
            return (toks, cur + 1, done, k), None

        (toks, _, _, _), _ = jax.lax.scan(
            step,
            (toks, jnp.asarray(t0, jnp.int32), jnp.asarray(False), key),
            None,
            length=max_new,
        )
        return toks

    return _scan_sample


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
    tok,  
) -> Tuple[float, float]:
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
            l, a = _eval_step(params, batch, sub)
            loss_sum += float(l)
            acc_sum += float(a)

        avg_loss = loss_sum / eval_batches
        avg_acc = acc_sum / eval_batches

        # generation (pjit + scan) with static knobs baked in
        pad_id = int(tok.pad_token_id)
        eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else pad_id
        temperature = 0.8

        scan_sample = _scan_sample_fn(
            static,
            max_new=int(cfg.gen_len),
            pad_id=pad_id,
            eos_id=eos_id,
            temperature=float(temperature),
        )

        sample_pjit = pjit(
            scan_sample,
            in_shardings=(  # params, prompt_ids, key, router_temp, gumbel_tau
                params_in_shardings,
                P(),
                P(),
                P(),
                P(),
            ),
            out_shardings=P(),
        )

        prompts = [
            "once upon a time",
            "the little robot",
            "in a quiet forest",
        ]

        print("\n[eval/generate]")
        for p in prompts:
            prompt_ids = jnp.array(tok.encode(p), dtype=jnp.int32)
            prompt_ids = jax.device_put(prompt_ids, NamedSharding(mesh, P()))
            key, sub = jax.random.split(key)
            toks = sample_pjit(
                params,
                prompt_ids,
                sub,
                model_kwargs["router_temp"],
                model_kwargs["gumbel_tau"],
            )
            seq = jax.device_get(toks).tolist()
            if eos_id in seq:
                seq = seq[: seq.index(eos_id) + 1]
            print(f"prompt: {p}\n→ {tok.decode(seq, skip_special_tokens=True)}\n")

    return avg_loss, avg_acc


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
    params_in_shardings = _sharding_pytree(params)

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
        opt_state = _replicate_scalars_to_mesh(opt_state, mesh)
    opt_state_in_shardings = _sharding_pytree(opt_state)

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
        denom = jnp.maximum(mask.sum(), 1)
        acc = ((jnp.argmax(logits, -1) == labels) & (mask > 0)).sum() / denom
        gnorm = optax.global_norm(grads)
        wnorm = optax.global_norm(pr)
        return pr, os, loss, acc, gnorm, wnorm

    step_pjit = pjit(
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

    # logging
    run_name = f"dna-att{cfg.n_attn_modules}-ff{cfg.n_ff_modules}-h{cfg.n_hops}-k{cfg.topk}-s{cfg.seed}"
    wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))
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
            if step % cfg.log_every == 0:
                elapsed = time.time() - t0_global
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
                    static,
                    mesh,
                    params_in_shardings,
                    val_stream,
                    cfg=cfg,
                    model_kwargs=model_kwargs,
                    key=k_eval,
                    tok=tok,
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
