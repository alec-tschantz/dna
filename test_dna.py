#!/usr/bin/env python3
import math

import equinox as eqx
import jax
import jax.numpy as jnp

from dna.dna import DNA
from dna import generate


def build_toy_model(key):
    return DNA(
        vocab=257,
        d_model=64,
        n_heads=8,
        n_modules=6,
        capacity=4,
        topk=2,
        n_hops=3,
        mlp_mult=2,
        dropout=0.0,
        rope_base=10_000.0,
        num_backbone=2,
        key=key,
    )


def test_forward_shapes_and_invariants():
    key = jax.random.PRNGKey(0)
    k_model, k_ids = jax.random.split(key)
    model = build_toy_model(k_model)

    T = 16
    V = 257
    ids = jax.random.randint(k_ids, (T,), 0, V, dtype=jnp.int32)
    amask = jnp.concatenate(
        [jnp.ones((12,), dtype=jnp.int32), jnp.zeros((4,), dtype=jnp.int32)]
    )

    logits, stats = model(ids, key=k_model, inference=False, attention_mask=amask)
    assert logits.shape == (T, V)
    assert isinstance(stats, tuple) and len(stats) == 3

    fwd = eqx.filter_jit(
        lambda m, x, k, msk: m(x, key=k, inference=False, attention_mask=msk)
    )
    logits_jit, stats_jit = fwd(model, ids, k_model, amask)
    assert jnp.allclose(logits, logits_jit, atol=1e-5, rtol=1e-5)

    for hop in stats_jit:
        load = hop["load"]
        importance = hop["importance"]
        cap = 4
        E = load.shape[0]

        assert load.ndim == 1
        assert importance.shape == (E,)
        assert jnp.all((load >= 0) & (load <= cap))

        selected_edges = int(hop["selected_edges"])
        kept_edges = int(hop["kept_edges"])
        assert 0 <= kept_edges <= selected_edges

        cdrop = float(hop["cap_drop_frac_edges"])
        assert 0.0 <= cdrop <= 1.0 + 1e-6

        for kname in ("rho_mean", "rho_min", "rho_max"):
            val = float(hop[kname])
            assert 0.0 <= val <= 1.0 + 1e-6

        for kname in ("entropy_mean", "entropy_min", "entropy_max"):
            val = float(hop[kname])
            assert 0.0 <= val <= 1.0 + 1e-6

        util = float(hop["util"])
        load_std = float(hop["load_std"])
        assert 0.0 <= util <= 1.0 + 1e-6
        assert 0.0 <= load_std <= 1.0 + 1e-6
        assert jnp.all(importance >= 0)

    print("forward/JIT invariants OK")


def test_grads_flow_and_are_finite():
    key = jax.random.PRNGKey(1)
    k_model, k_ids = jax.random.split(key)
    model = build_toy_model(k_model)

    T = 12
    V = 257
    ids = jax.random.randint(k_ids, (T,), 0, V, dtype=jnp.int32)
    amask = jnp.concatenate(
        [jnp.ones((10,), dtype=jnp.int32), jnp.zeros((2,), dtype=jnp.int32)]
    )

    def loss_fn(m: DNA, x, k, msk):
        logits, _ = m(x, key=k, inference=False, attention_mask=msk)
        labels = jnp.roll(x, shift=-1)
        logits = logits[:-1]
        labels = labels[1:]
        msk_shift = msk[1:].astype(bool)
        nll_tok = -jax.nn.log_softmax(logits, axis=-1)[jnp.arange(T - 1), labels]
        nll = jnp.sum(nll_tok * msk_shift) / jnp.maximum(jnp.sum(msk_shift), 1)
        return nll

    value_and_grad = eqx.filter_value_and_grad(eqx.filter_jit(loss_fn))
    loss, grads = value_and_grad(model, ids, k_model, amask)

    def finite_tree(tree):
        leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
        if not leaves:
            return True
        return bool(
            jnp.all(jnp.concatenate([jnp.ravel(jnp.isfinite(x)) for x in leaves]))
        )

    assert math.isfinite(float(loss))
    assert finite_tree(grads)

    gnorm = jnp.sqrt(
        sum(
            jnp.sum(g**2)
            for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        )
    )
    assert float(gnorm) > 0.0
    print("grads/JIT OK")


def test_generate_shapes_prefix_and_greedy_equivalence():
    key = jax.random.PRNGKey(2)
    k_model, k_prompt, k_gen = jax.random.split(key, 3)
    model = build_toy_model(k_model)

    V = 257
    pad_id = 0
    eos_id = 0

    prompt_len = 7
    prompt_ids = jax.random.randint(k_prompt, (prompt_len,), 0, V, dtype=jnp.int32)

    max_new = 9
    total_len = prompt_len + max_new

    toks = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new,
        temperature=0.0,
        key=k_gen,
        biases=None,
        gumbel=False,
        gumbel_tau=1.0,
        router_temp=1.0,
        greedy=True,
        pad_id=pad_id,
        eos_id=eos_id,
    )

    assert toks.shape == (total_len,)
    assert toks.dtype == jnp.int32
    assert jnp.all(toks[:prompt_len] == prompt_ids)

    attn_mask = jnp.ones((prompt_len,), dtype=jnp.int32)
    logits, _ = model(prompt_ids, key=k_model, inference=True, attention_mask=attn_mask)
    expected_next = jnp.argmax(logits[-1]).astype(jnp.int32)
    assert int(toks[prompt_len]) == int(expected_next)

    gen_jit = eqx.filter_jit(
        lambda m, p, k: generate(
            m,
            p,
            max_new_tokens=max_new,
            temperature=0.0,
            key=k,
            biases=None,
            gumbel=False,
            gumbel_tau=1.0,
            router_temp=1.0,
            greedy=True,
            pad_id=pad_id,
            eos_id=eos_id,
        )
    )
    toks_jit = gen_jit(model, prompt_ids, k_gen)
    assert jnp.array_equal(toks, toks_jit)

    print("generate shapes/prefix/greedy/JIT OK")


if __name__ == "__main__":
    test_forward_shapes_and_invariants()
    test_grads_flow_and_are_finite()
    test_generate_shapes_prefix_and_greedy_equivalence()
