#!/usr/bin/env python3
import os, subprocess
from dataclasses import dataclass
from typing import List, Tuple

import jax, jax.numpy as jnp
import equinox as eqx
import optax

from dna import DNA, Attention, FeedForward, Identity
from dna.routing import SequenceRouter


# -------------------- Args --------------------
@dataclass
class Args:
    batch_size: int = 32
    seq_len: int = 256
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 12
    n_att_modules: int = 6
    n_ff_modules: int = 6
    n_id_modules: int = 0
    mlp_mult: int = 4
    capacity: int = 64
    topk: int = 2
    dropout: float = 0.0
    vocab_size: int = 50_257
    backbone: Tuple[str] = ("feedforward",)


# -------------------- GPU helpers --------------------
def gpu_used_mb(device_index=0):
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .splitlines()
        )
        return int(out[device_index])
    except Exception:
        return None


def gpu_total_mb(device_index=0):
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .splitlines()
        )
        return int(out[device_index])
    except Exception:
        return None


def pct(x, total):
    return f"{(x/total)*100:.1f}%" if x is not None and total else "-"


# -------------------- Builders --------------------
def make_modules(*, d_model, n_heads, n_att, n_ff, n_id, mlp_mult, dropout, key):
    total = n_att + n_ff
    keys = list(jax.random.split(key, total)) if total > 0 else []
    att = [Attention(d_model, n_heads, dropout, key=k) for k in keys[:n_att]]
    ff = [FeedForward(d_model, mlp_mult, dropout, key=k) for k in keys[n_att:]]
    ids = [Identity() for _ in range(n_id)]
    return tuple(att + ff + ids)


def make_backbone(*, d_model, n_heads, mlp_mult, dropout, backbone, key):
    if not backbone:
        return ()
    keys = list(jax.random.split(key, len(backbone)))
    out = []
    for i, layer_type in enumerate(backbone):
        lt = layer_type.lower()
        if lt == "attention":
            out.append(Attention(d_model, n_heads, dropout, key=keys[i]))
        elif lt == "feedforward":
            out.append(FeedForward(d_model, mlp_mult, dropout, key=keys[i]))
        else:
            raise ValueError(f"Unknown backbone layer type '{layer_type}'")
    return tuple(out)


def build_dna(cfg: Args, key):
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
        backbone=cfg.backbone or [],
        key=kb,
    )
    return DNA(
        modules=modules,
        router_cls=SequenceRouter,
        vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        capacity=cfg.capacity,
        topk=cfg.topk,
        n_hops=cfg.n_hops,
        dropout=cfg.dropout,
        rope_base=10_000.0,
        norm_probs=False,
        backbone=backbone,
        key=kmodel,
    )


# -------------------- Loss and steps --------------------
def compute_loss(model, ids, mask, key):
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    model_kwargs = {
        "gumbel_tau": jnp.array([1.0]),
        "router_temp": jnp.array([1.0]),
        "select_temp": jnp.array([1.0]),
    }

    def forward(x, m, k):
        logits, stats = model(x, key=k, inference=False, mask=m, **model_kwargs)
        return logits, stats

    logits, stats = jax.vmap(forward, in_axes=(0, 0, 0))(ids, mask, keys)
    labels = ids[:, 1:]
    logits = logits[:, :-1]
    mask = mask[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return (loss * mask).sum() / jnp.maximum(mask.sum(), 1)


@eqx.filter_jit
def forward_only(model, ids, mask, key):
    return compute_loss(model, ids, mask, key)


@eqx.filter_jit
def train_step(model, opt, opt_state, ids, mask, key):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, ids, mask, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# -------------------- Memory test --------------------
def main():
    cfg = Args()
    total_mb = gpu_total_mb() or 1
    key = jax.random.PRNGKey(0)

    model = build_dna(cfg, key)
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    B, T = cfg.batch_size, cfg.seq_len
    ids = jax.random.randint(key, (B, T), 0, cfg.vocab_size, dtype=jnp.int32)
    mask = jnp.ones((B, T), dtype=jnp.float32)

    def fmt(label, used_mb, delta_mb=None):
        if delta_mb is None:
            return f"{label:<20} {used_mb:>6} MB ({pct(used_mb, total_mb)})"
        sign = "+" if delta_mb >= 0 else "-"
        return f"{label:<20} {used_mb:>6} MB  ({sign}{abs(delta_mb):>4} MB, {pct(delta_mb, total_mb)})"

    start = gpu_used_mb()

    print("\n=== GPU Memory Usage ===")
    print(f"Total GPU memory: {total_mb} MB")

    # Model + optimizer state
    _ = jax.device_put(eqx.filter(model, eqx.is_array))
    _ = jax.device_put(opt_state)
    after_state = gpu_used_mb()
    print(fmt("Model + Opt state", after_state, after_state - start))

    # Forward pass (compile + run)
    loss = forward_only(model, ids, mask, key)
    loss.block_until_ready()
    after_fwd = gpu_used_mb()
    print(fmt("Fwd activations", after_fwd, after_fwd - after_state))

    # Backward pass (compile + run)
    model, opt_state, loss = train_step(model, opt, opt_state, ids, mask, key)
    loss.block_until_ready()
    after_bwd = gpu_used_mb()
    print(fmt("Gradients (bwd)", after_bwd, after_bwd - after_fwd))

    print(fmt("Total after bwd", after_bwd))
    print("========================\n")


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
