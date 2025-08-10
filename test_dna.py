# demo_test.py
# -----------------------------------------------------------------------------
# Small, self-contained demo:
#   • builds a model
#   • does a single forward on a batch via vmap(model)
#   • computes a simple next-token loss
#   • runs JIT and gradients through everything
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from dna.model import build_default_model


def _nll_for_sequence(logits_tv: jnp.ndarray, targets_t: jnp.ndarray) -> jnp.ndarray:
    """Standard next-token NLL (teacher forcing, shift by 1); mean over valid tokens."""
    T, V = logits_tv.shape
    # shift: predict token t from t-1; ignore t=0
    logits_pred = logits_tv[:-1]                    # (T-1, V)
    targets = targets_t[1:]                         # (T-1,)
    logp = logits_pred - jax.nn.logsumexp(logits_pred, axis=-1, keepdims=True)
    nll = -jnp.take_along_axis(logp, targets[:, None], axis=-1).squeeze(-1)
    return nll.mean()


def make_model_and_data(
    B: int = 4,
    T: int = 32,
    V: int = 1024,
    *,
    key=jax.random.PRNGKey(0),
):
    model = build_default_model(
        vocab=V,
        d_model=128,
        n_heads=4,
        n_experts=6,
        capacity=16,
        topk=2,
        n_hops=3,
        mlp_mult=4,
        dropout=0.1,
        rope_base=10_000.0,
        num_backbone=1,
        key=key,
    )
    k_data, = jax.random.split(key, 1)
    ids_bt = jax.random.randint(k_data, (B, T), 0, V, dtype=jnp.int32)
    return model, ids_bt


def forward_batch(model, ids_bt, *, key, inference: bool) -> Tuple[jnp.ndarray, dict]:
    """Runs a batched forward via VMAP over sequences; returns logits and a few hop stats."""
    B, T = ids_bt.shape

    def _one(ids_t, k):
        logits_tv, hop_stats = model(ids_t, key=k, inference=inference)
        # Return just a couple of scalar stats per hop for easy aggregation
        scalars = {f"hop{h}_util": s["util_frac"] for h, s in enumerate(hop_stats)}
        return logits_tv, scalars

    keys = jax.random.split(key, B)
    logits_btv, stats_b = jax.vmap(_one)(ids_bt, keys)
    # Merge stats by averaging across batch
    def _mean_stat(dicts):
        keys = list(dicts[0].keys())
        return {k: jnp.mean(jnp.stack([d[k] for d in dicts])) for k in keys}

    return logits_btv, _mean_stat(list(stats_b))


def loss_fn(model, ids_bt, *, key) -> jnp.ndarray:
    logits_btv, _ = forward_batch(model, ids_bt, key=key, inference=False)
    # Mean sequence NLL across batch
    nll_b = jax.vmap(_nll_for_sequence)(logits_btv, ids_bt)
    return nll_b.mean()


# JIT + grad demo ------------------------------------------------------------

def run_demo():
    key = jax.random.PRNGKey(42)
    model, ids_bt = make_model_and_data(B=3, T=24, V=512, key=key)

    # JIT forward + loss
    jit_loss = jax.jit(lambda m, x, k: loss_fn(m, x, key=k))

    # Filter params for differentiation
    value_and_grad = eqx.filter_value_and_grad(jit_loss)

    # A single step of value+grad (gradients w.r.t. parameters only)
    loss, grads = value_and_grad(model, ids_bt, key)

    # Also demonstrate an eval (no gumbel exploration)
    eval_logits, eval_stats = forward_batch(model, ids_bt, key=key, inference=True)

    return {
        "loss": loss,
        "mean_util_per_hop": eval_stats,
        "eval_logits_shape": eval_logits.shape,
        # grads is a pytree; typically you'd pass it into an Optax optimizer
    }


if __name__ == "__main__":
    out = run_demo()
    # Printing small summary; real training would integrate with Optax, etc.
    for k, v in out.items():
        print(k, (v.shape if hasattr(v, "shape") else v))