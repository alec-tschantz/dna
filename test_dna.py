import math
import jax
import jax.numpy as jnp
import equinox as eqx

from dna.dna import DNA


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
    key, mk, fk = jax.random.split(key, 3)
    model = build_toy_model(mk)
    T = 16
    V = 257
    ids = jax.random.randint(fk, (T,), 0, V, dtype=jnp.int32)
    amask = jnp.concatenate(
        [jnp.ones((12,), dtype=jnp.int32), jnp.zeros((4,), dtype=jnp.int32)]
    )
    logits, stats = model(ids, key=mk, inference=False, attention_mask=amask)
    assert logits.shape == (T, V)
    assert isinstance(stats, tuple) and len(stats) == 3
    fwd = eqx.filter_jit(
        lambda m, x, k, msk: m(x, key=k, inference=False, attention_mask=msk)
    )
    logits_jit, stats_jit = fwd(model, ids, mk, amask)
    assert jnp.allclose(logits, logits_jit, atol=1e-5, rtol=1e-5)
    for hop in stats_jit:
        load = hop["load"]
        importance = hop["importance"]
        cap = 4
        E = load.shape[0]
        assert load.ndim == 1 and importance.shape == (E,)
        assert jnp.all((load >= 0) & (load <= cap))
        selected_edges = int(hop["selected_edges"])
        kept_edges = int(hop["kept_edges"])
        assert 0 <= kept_edges <= selected_edges
        cdrop = float(hop["cap_drop_frac_edges"])
        assert 0.0 <= cdrop <= 1.0 + 1e-6
        for kname in ["rho_mean", "rho_min", "rho_max"]:
            val = float(hop[kname])
            assert 0.0 <= val <= 1.0 + 1e-6
        for kname in ["entropy_mean", "entropy_min", "entropy_max"]:
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
    key, mk, fk = jax.random.split(key, 3)
    model = build_toy_model(mk)
    T = 12
    V = 257
    ids = jax.random.randint(fk, (T,), 0, V, dtype=jnp.int32)
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
    loss, grads = value_and_grad(model, ids, mk, amask)

    def finite_tree(tree):
        return jax.tree_util.tree_reduce(
            lambda a, b: a & jnp.all(jnp.isfinite(b)),
            jax.tree.map(
                lambda x: x if isinstance(x, jnp.ndarray) else jnp.array(0.0),
                eqx.filter(grads, eqx.is_array),
            ),
            True,
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


if __name__ == "__main__":
    test_forward_shapes_and_invariants()
    test_grads_flow_and_are_finite()
