# test_pjit_dna.py
from __future__ import annotations
from dataclasses import dataclass
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import tyro
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils

from dna import DNA, Attention, FeedForward, LinearRouter, f32


# ----------------------------- Mesh & sharding ------------------------------

def make_mesh(n_data: int, n_expert: int) -> Mesh:
    devs = jax.devices()
    if len(devs) == 1:
        dm = mesh_utils.create_device_mesh((1, 1), devices=devs)
    else:
        want, have = n_data * n_expert, len(devs)
        if want > have:
            n_data = min(n_data, have)
            while n_data > 1 and (have % n_data) != 0:
                n_data -= 1
            n_expert = max(1, have // n_data)
        dm = mesh_utils.create_device_mesh((n_data, n_expert), devices=devs)
    return Mesh(dm, ("data", "expert"))


def shard_expert_params(model: DNA, mesh: Mesh) -> DNA:
    """Shard stacked expert params on 'expert' when divisible; else replicate."""
    expert_axis = mesh.shape["expert"]
    grouped_leaves = []
    for g in model.groups:
        grouped_leaves += jax.tree.leaves(g.params)
    ids = {id(x): True for x in grouped_leaves}

    def put(a):
        if not isinstance(a, jnp.ndarray):
            return a
        if id(a) in ids and a.ndim >= 1 and expert_axis > 1 and (a.shape[0] % expert_axis == 0):
            spec = NamedSharding(mesh, P("expert", *([None] * (a.ndim - 1))))
        else:
            spec = NamedSharding(mesh, P())
        return jax.device_put(a, spec)

    return jax.tree.map(put, model)


def params_sharding_pytree(params):
    return jax.tree.map(lambda x: x.sharding if isinstance(x, jax.Array) else None, params)


# --------------------------------- IO / utils --------------------------------

def build_io(B: int, T: int, V: int, *, key):
    ids = jax.random.randint(key, (B, T), 0, V, dtype=jnp.int32)
    mask = jax.random.uniform(key, (B, T)) > 0.1
    return ids, mask


def tree_all_finite(tree) -> bool:
    def _leaf(acc, x):
        return acc & (jnp.all(jnp.isfinite(x)) if isinstance(x, jnp.ndarray) else True)
    return jax.tree_util.tree_reduce(_leaf, tree, True)


# --------------------------------- Model build --------------------------------

def build_model_and_routers(
    *,
    vocab: int,
    d_model: int,
    n_heads: int,
    n_hops: int,
    topk: int,
    dropout: float,
    rope_base: float,
    n_attn: int,
    n_ff: int,
    key: jax.Array,
) -> DNA:
    k_mods, k_routers = jax.random.split(key, 2)
    klist = jax.random.split(k_mods, n_attn + n_ff)
    mods = [Attention(d_model, n_heads, dropout, key=klist[i]) for i in range(n_attn)]
    mods += [FeedForward(d_model, 4, dropout, key=klist[n_attn + i]) for i in range(n_ff)]

    total_experts = n_attn + n_ff
    router_keys = jax.random.split(k_routers, n_hops)
    routers = tuple(LinearRouter(d_model, total_experts, topk, dropout, key=k) for k in router_keys)

    return DNA(
        modules=tuple(mods),
        routers=routers,
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        rope_base=rope_base,
        backbone=None,
        key=jax.random.PRNGKey(0),  # used for embedding init
    )


# --------------------------------- CLI config --------------------------------

@dataclass
class Args:
    # data / shapes
    batch_size: int = 64
    seq_len: int = 128
    vocab: int = 32000

    # model
    d_model: int = 256
    n_heads: int = 8
    n_hops: int = 2
    topk: int = 2
    dropout: float = 0.1
    rope_base: float = 10_000.0

    # experts
    n_attn: int = 8
    n_ff: int = 8

    # mesh
    batch_shards: int = 4     # 'data' axis
    expert_shards: int = 2    # 'expert' axis

    seed: int = 0


# --------------------------------- main --------------------------------------

def main():
    args = tyro.cli(Args)
    mesh = make_mesh(args.batch_shards, args.expert_shards)
    print("Mesh:", mesh.devices.shape, "platforms:", {d.platform for d in mesh.devices.flat})

    # sanity checks (group-wise divisibility keeps sharding simple)
    assert args.batch_size % args.batch_shards == 0, "batch_size must be divisible by batch_shards"
    assert args.n_attn % args.expert_shards == 0, "n_attn must be divisible by expert_shards"
    assert args.n_ff % args.expert_shards == 0, "n_ff must be divisible by expert_shards"

    model = build_model_and_routers(
        vocab=args.vocab, d_model=args.d_model, n_heads=args.n_heads, n_hops=args.n_hops,
        topk=args.topk, dropout=args.dropout, rope_base=args.rope_base,
        n_attn=args.n_attn, n_ff=args.n_ff, key=jax.random.PRNGKey(args.seed)
    )
    model = shard_expert_params(model, mesh)

    total_experts = sum(int(g.idx.shape[0]) for g in model.groups)
    experts_per_shard = sum(int(g.idx.shape[0]) // args.expert_shards for g in model.groups)
    print(f"Total experts: {total_experts} | expert_shards: {args.expert_shards}")
    print(f"Experts per expert-shard: ~{experts_per_shard}")
    print(f"Batch per data-shard: {args.batch_size // args.batch_shards}")

    params, static = eqx.partition(model, eqx.is_inexact_array)
    params_in_shardings = params_sharding_pytree(params)

    key0 = jax.random.PRNGKey(args.seed + 1)
    ids, msk = build_io(args.batch_size, args.seq_len, args.vocab, key=key0)

    with mesh:
        in_ids = jax.device_put(ids, NamedSharding(mesh, P("data", None)))
        in_msk = jax.device_put(msk, NamedSharding(mesh, P("data", None)))

        # ---------- forward (B,T) via vmap over T-only model ----------
        def fwd(params, ids, mask, key):
            mdl = eqx.combine(params, static)
            B = ids.shape[0]
            keys = jax.random.split(key, B)
            call_one = lambda s_ids, s_mask, k: mdl(
                s_ids, key=k, inference=False, mask=s_mask, router_temp=1.0, gumbel_tau=0.5
            )  # [T,V]
            return jax.vmap(call_one)(ids, mask, keys)  # [B,T,V]

        pjitted_fwd = pjit(
            fwd,
            in_shardings=(params_in_shardings, P("data", None), P("data", None), None),
            out_shardings=P("data", None, None),
        )

        # timings
        t0 = time.perf_counter()
        logits = pjitted_fwd(params, in_ids, in_msk, jax.random.PRNGKey(123))
        logits.block_until_ready()
        t1 = time.perf_counter()
        print("logits:", logits.shape, logits.addressable_shards[0].data.dtype)
        print(f"[FWD] first-call (compile+run): {t1 - t0:.3f}s")

        t2 = time.perf_counter()
        _ = pjitted_fwd(params, in_ids, in_msk, jax.random.PRNGKey(124)).block_until_ready()
        t3 = time.perf_counter()
        print(f"[FWD] steady-state run: {t3 - t2:.3f}s")

        # ---------- loss ----------
        tgt = jax.random.randint(key0, (args.batch_size, args.seq_len), 0, args.vocab, dtype=jnp.int32)
        tgt = jax.device_put(tgt, NamedSharding(mesh, P("data", None)))

        def loss_fn(params, ids, mask, key):
            y = fwd(params, ids, mask, key)                 # [B,T,V]
            logp = jax.nn.log_softmax(y.astype(f32), axis=-1)
            gathered = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
            valid = mask.astype(f32)
            return -(gathered * valid).sum() / jnp.clip(valid.sum(), 1.0)

        pjitted_loss = pjit(
            loss_fn,
            in_shardings=(params_in_shardings, P("data", None), P("data", None), None),
            out_shardings=P(),
        )

        t4 = time.perf_counter()
        val = pjitted_loss(params, in_ids, in_msk, jax.random.PRNGKey(999))
        float(val)
        t5 = time.perf_counter()
        print("loss:", float(val))
        print(f"[LOSS] first-call (compile+run): {t5 - t4:.3f}s")

        # ---------- value_and_grad (wrapped) ----------
        def loss_and_grad_wrapped(params, ids, mask, key):
            v, g = eqx.filter_value_and_grad(loss_fn)(params, ids, mask, key)
            return v, g

        pjitted_vg = pjit(
            loss_and_grad_wrapped,
            in_shardings=(params_in_shardings, P("data", None), P("data", None), None),
            out_shardings=(P(), params_in_shardings),
        )

        t6 = time.perf_counter()
        loss_val, grads = pjitted_vg(params, in_ids, in_msk, jax.random.PRNGKey(1001))
        jax.tree.map(lambda x: x.block_until_ready() if isinstance(x, jax.Array) else None, grads)
        t7 = time.perf_counter()
        print("loss (vg):", float(loss_val))
        print("grads finite:", bool(tree_all_finite(grads)))
        print(f"[LOSS+GRAD] first-call (compile+run): {t7 - t6:.3f}s")

        t8 = time.perf_counter()
        loss_val2, _ = pjitted_vg(params, in_ids, in_msk, jax.random.PRNGKey(1002))
        float(loss_val2)
        t9 = time.perf_counter()
        print(f"[LOSS+GRAD] steady-state run: {t9 - t8:.3f}s")


if __name__ == "__main__":
    main()
