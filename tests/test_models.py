from __future__ import annotations
from typing import Tuple, Dict, Any

import math
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Int, Float, Array, Bool

from dna.model import Model
from dna.modules import Attention, FeedForward, Identity
from dna.sample import sample


def make_small_dna(
    *,
    vocab: int = 32,
    d_model: int = 32,
    n_heads: int = 4,
    n_experts: int = 4,
    capacity: int = 4,
    topk: int = 2,
    n_hops: int = 2,
    dropout: float = 0.0,
    rope_base: float = 10_000.0,
    key: jax.Array,
) -> Model:
    k_mods = jax.random.split(key, n_experts)
    mods = []
    for i in range(n_experts):
        if i % 3 == 0:
            mods.append(Attention(d_model, n_heads, dropout, key=k_mods[i]))
        elif i % 3 == 1:
            mods.append(FeedForward(d_model, 2, dropout, key=k_mods[i]))
        else:
            mods.append(Identity())
    k_model = jax.random.split(key, 1)[0]
    model = Model(
        modules=tuple(mods),
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        capacity=capacity,
        topk=topk,
        n_hops=n_hops,
        dropout=dropout,
        rope_base=rope_base,
        backbone=None,
        key=k_model,
    )
    return model


def rand_tokens(T: int, V: int, *, key: jax.Array) -> Int[Array, "T"]:
    return jax.random.randint(key, (T,), minval=0, maxval=V, dtype=jnp.int32)


def test_forward_shapes_and_stats():
    key = jax.random.PRNGKey(0)
    V, T = 48, 12
    model = make_small_dna(vocab=V, key=key)
    key, k_tok, k_call = jax.random.split(key, 3)
    ids = rand_tokens(T, V, key=k_tok)
    attn_mask = jnp.ones((T,), dtype=bool)
    logits, stats_all = model(
        ids,
        key=k_call,
        inference=True,
        attention_mask=attn_mask,
        gumbel_tau=1.0,
        router_temperature=1.0,
        select_temperature=None,
    )
    assert logits.shape == (T, V)
    assert isinstance(stats_all, tuple) and len(stats_all) == len(model.routers)
    required = {
        "load",
        "importance_sum",
        "importance_mean",
        "rho_mean",
        "rho_min",
        "rho_max",
        "entropy_mean",
        "entropy_min",
        "entropy_max",
        "util",
        "load_std",
        "cap_drop_frac_edges",
        "selected_edges",
        "kept_edges",
        "eff_topk_mean",
        "eff_topk_min",
        "eff_topk_max",
        "cap_util_mean",
        "cap_util_min",
        "cap_util_max",
    }
    for st in stats_all:
        assert required.issubset(st.keys())
        assert (st["load"] <= model.capacity).all()
        for k in [
            "rho_mean",
            "rho_min",
            "rho_max",
            "entropy_mean",
            "entropy_min",
            "entropy_max",
            "cap_drop_frac_edges",
            "cap_util_mean",
            "cap_util_min",
            "cap_util_max",
        ]:
            assert jnp.all(jnp.isfinite(jnp.asarray(st[k])))


def test_vmap_batch():
    key = jax.random.PRNGKey(1)
    V, T, B = 32, 10, 3
    model = make_small_dna(vocab=V, key=key)

    def run(ids, k):
        logits, _ = model(
            ids, key=k, inference=True, attention_mask=jnp.ones((T,), bool)
        )
        return logits

    k_ids = jax.random.split(key, B)
    batch_ids = jnp.stack([rand_tokens(T, V, key=ki) for ki in k_ids], axis=0)
    k_calls = jax.random.split(jax.random.PRNGKey(2), B)
    batched = jax.vmap(run)(batch_ids, k_calls)
    assert batched.shape == (B, T, V)


def test_jit_wrapper_matches_eager():
    key = jax.random.PRNGKey(2)
    V, T = 40, 9
    model = make_small_dna(vocab=V, key=key)
    ids = rand_tokens(T, V, key=jax.random.PRNGKey(21))
    am = jnp.ones((T,), dtype=bool)

    def f(m, ids, k):
        logits, _ = m(ids, key=k, inference=True, attention_mask=am)
        return logits

    f_jit = eqx.filter_jit(f)
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    out_eager = f(model, ids, k1)
    out_jit = f_jit(model, ids, k2)
    assert jnp.allclose(out_eager, out_jit, atol=1e-6)


def test_generate_loop():
    key = jax.random.PRNGKey(3)
    V, T0 = 50, 7
    model = make_small_dna(vocab=V, key=key)
    prompt = rand_tokens(T0, V, key=jax.random.PRNGKey(33))
    out = sample(
        model,
        prompt,
        max_new_tokens=8,
        key=jax.random.PRNGKey(44),
        temperature=0.0,
        greedy=True,
        router_temperature=1.0,
        select_temperature=None,
        gumbel_tau=1.0,
        pad_id=0,
        eos_id=None,
    )
    assert out.shape[0] == T0 + 8
    assert out.dtype == jnp.int32
    assert int(out.max()) < V and int(out.min()) >= 0


def test_grads_finite():
    key = jax.random.PRNGKey(4)
    V, T = 32, 12
    model = make_small_dna(vocab=V, key=key, dropout=0.0)
    ids = rand_tokens(T, V, key=jax.random.PRNGKey(123))
    targets = rand_tokens(T, V, key=jax.random.PRNGKey(124))
    am = jnp.ones((T,), dtype=bool)

    def loss_fn(m: Model, k):
        logits, _ = m(ids, key=k, inference=False, attention_mask=am)
        logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        nll = -jnp.take_along_axis(logp, targets[:, None], axis=-1).squeeze(-1)
        return nll.mean()

    val_and_grad = eqx.filter_value_and_grad(loss_fn)
    loss, grads = val_and_grad(model, jax.random.PRNGKey(5))
    assert jnp.isfinite(loss)
    array_grads = eqx.filter(grads, eqx.is_array)
    for leaf in jax.tree_util.tree_leaves(array_grads):
        assert jnp.all(jnp.isfinite(leaf))


def test_capacity_and_topk_invariants():
    key = jax.random.PRNGKey(6)
    V, T = 64, 16
    capacity, topk = 3, 2
    model = make_small_dna(vocab=V, capacity=capacity, topk=topk, key=key)
    ids = rand_tokens(T, V, key=jax.random.PRNGKey(77))
    logits, stats_all = model(ids, key=jax.random.PRNGKey(78), inference=True)
    for st in stats_all:
        assert int(st["load"].max()) <= capacity
        assert float(st["eff_topk_mean"]) <= topk + 1e-5


if __name__ == "__main__":
    pytest.main([__file__])
