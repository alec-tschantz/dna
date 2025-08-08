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
    ids = jax.random.randint(fk, (T,), 0, 257, dtype=jnp.int32)

    # Plain call
    logits, stats = model(ids, key=mk, inference=False)
    V = 257
    assert logits.shape == (T, V)
    assert isinstance(stats, tuple) and len(stats) == 3

    # JIT call
    fwd = eqx.filter_jit(lambda m, x, k: m(x, key=k, inference=False))
    logits_jit, stats_jit = fwd(model, ids, mk)
    assert jnp.allclose(logits, logits_jit, atol=1e-5, rtol=1e-5)

    # Invariants per hop
    for hop in stats_jit:
        load = hop["load"]               # [E]
        importance = hop["importance"]   # [E]
        cap = 4
        E = load.shape[0]

        # shapes
        assert load.ndim == 1 and importance.shape == (E,)

        # capacity respected
        assert jnp.all((load >= 0) & (load <= cap))

        # edge accounting
        selected_edges = int(hop["selected_edges"])
        kept_edges = int(hop["kept_edges"])
        assert 0 <= kept_edges <= selected_edges

        # drop fraction in [0,1]
        cdrop = float(hop["cap_drop_frac_edges"])
        assert 0.0 <= cdrop <= 1.0 + 1e-6

        # rho stats in [0,1]
        for kname in ["rho_mean", "rho_min", "rho_max"]:
            val = float(hop[kname])
            assert 0.0 <= val <= 1.0 + 1e-6

        # entropy stats in [0,1]
        for kname in ["entropy_mean", "entropy_min", "entropy_max"]:
            val = float(hop[kname])
            assert 0.0 <= val <= 1.0 + 1e-6

        # utilization & dispersion sane
        util = float(hop["util"])
        load_std = float(hop["load_std"])
        assert 0.0 <= util <= 1.0 + 1e-6
        assert 0.0 <= load_std <= 1.0 + 1e-6

        # importance non-negative
        assert jnp.all(importance >= 0)

    print("forward/JIT invariants OK")


def test_grads_flow_and_are_finite():
    key = jax.random.PRNGKey(1)
    key, mk, fk = jax.random.split(key, 3)
    model = build_toy_model(mk)

    T = 12
    V = 257
    ids = jax.random.randint(fk, (T,), 0, V, dtype=jnp.int32)

    def loss_fn(m: DNA, x, k):
        logits, _ = m(x, key=k, inference=False)  # [T, V]
        # teacher-forcing next-token NLL without masking (just for test)
        labels = jnp.roll(x, shift=-1)
        logits = logits[:-1]
        labels = labels[1:]
        nll = jnp.mean(
            -jax.nn.log_softmax(logits, axis=-1)[jnp.arange(T - 1), labels]
        )
        return nll

    # JIT + grads
    value_and_grad = eqx.filter_value_and_grad(eqx.filter_jit(loss_fn))
    loss, grads = value_and_grad(model, ids, mk)

    # Finite and some signal present
    def finite_tree(tree):
        return jax.tree_util.tree_reduce(
            lambda a, b: a & jnp.all(jnp.isfinite(b)),
            jax.tree_map(lambda x: x if isinstance(x, jnp.ndarray) else jnp.array(0.0), eqx.filter(grads, eqx.is_array)),
            True,
        )

    assert math.isfinite(float(loss))
    assert finite_tree(grads)

    # Global norm > 0 (some grad flows)
    gnorm = jnp.sqrt(
        sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)))
    )
    assert float(gnorm) > 0.0

    print("grads/JIT OK")


if __name__ == "__main__":
    test_forward_shapes_and_invariants()
    test_grads_flow_and_are_finite()
