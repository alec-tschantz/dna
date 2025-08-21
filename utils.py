# utils.py
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
import wandb
from transformers import AutoTokenizer
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.experimental.multihost_utils import sync_global_devices

f32 = jnp.float32


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


def sharding_pytree(params):
    return jax.tree.map(
        lambda x: x.sharding if isinstance(x, jax.Array) else None, params
    )


def replicate_scalars_to_mesh(tree, mesh):
    def place(x):
        if isinstance(x, jax.Array) and x.shape == ():  # scalar leaf
            return jax.device_put(x, NamedSharding(mesh, P()))
        return x

    return jax.tree.map(place, tree)


def lr_schedule(step, warmup, steps, lr_peak):
    warm = jnp.minimum(step / warmup, 1.0)
    lr = lr_peak * warm
    decay_steps = jnp.maximum(steps - warmup, 1)
    progress = jnp.clip((step - warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step >= warmup, lr_peak * cos, lr).astype(f32)


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


def build_sample_fn(
    static, *, max_new: int, pad_id: int, eos_id: int, temperature: float
):
    def _scan_sample(
        params,
        prompt_ids: jnp.ndarray,  # [B, T0]
        prompt_lens: jnp.ndarray,  # [B]
        key: jax.Array,
        router_temp: jnp.ndarray,
        gumbel_tau: jnp.ndarray,
    ) -> jnp.ndarray:
        model = eqx.combine(params, static)
        B, T0 = prompt_ids.shape
        total_len = T0 + max_new

        toks = jnp.full((B, total_len), pad_id, dtype=jnp.int32)
        toks = toks.at[:, :T0].set(prompt_ids)

        cur = prompt_lens.astype(jnp.int32)  # [B]
        done = jnp.zeros((B,), dtype=jnp.bool_)  # [B]
        force_greedy = jnp.asarray(temperature, f32) <= 0

        def step(carry, _):
            toks, cur, done, k = carry
            ks = jax.random.split(k, B + 1)
            subkeys, k = ks[:-1], ks[-1]

            ar = jnp.arange(total_len, dtype=jnp.int32)[None, :]
            attn_mask = ar < cur[:, None]

            def run_one(t_i, m_i, k_i):
                logits = model(
                    t_i,
                    key=k_i,
                    inference=True,
                    mask=m_i,
                    router_temp=router_temp,
                    gumbel_tau=gumbel_tau,
                )  # [T, V]
                return logits

            logits = jax.vmap(run_one)(toks, attn_mask, subkeys)  # [B, T, V]
            V = logits.shape[-1]
            last_idx = cur - jnp.asarray(1, jnp.int32)

            def last_row(lg, idx):
                return jax.lax.dynamic_slice(lg, (idx, 0), (1, V))[0]

            last_logits = jax.vmap(last_row)(logits, last_idx)  # [B, V]

            def pick_greedy(lg):
                return jnp.argmax(lg, axis=-1).astype(jnp.int32)

            def pick_sample(rngs, lg):
                scaled = lg / jnp.clip(jnp.asarray(temperature, f32), 1e-6, None)
                return jax.vmap(
                    lambda r, z: jax.random.categorical(r, z).astype(jnp.int32)
                )(rngs, scaled)

            nxt = jax.lax.cond(
                force_greedy,
                lambda _: pick_greedy(last_logits),
                lambda rngs: pick_sample(rngs, last_logits),
                operand=subkeys,
            )
            nxt = jnp.where(done, jnp.asarray(pad_id, jnp.int32), nxt)

            toks = toks.at[jnp.arange(B), cur].set(nxt)
            done = done | (nxt == eos_id)
            cur = cur + 1
            return (toks, cur, done, k), None

        (toks, _, _, _), _ = jax.lax.scan(
            step,
            (toks, cur, done, key),
            None,
            length=max_new,
        )
        return toks

    return _scan_sample
