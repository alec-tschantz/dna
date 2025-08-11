from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from dna import Model, Attention, FeedForward, Identity, sample


# ============================== Config =================================== #


@dataclass
class Config:
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_hops: int = 6
    n_modules: int = 16
    topk: int = 2
    capacity: int = 128
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0
    router_temp: float = 1.5
    select_temp: float = 1.75
    gumbel_tau: float = 1.2
    batch_size: int = 32
    seq_len: int = 256
    steps: int = 20_000
    warmup: int = 2_000
    lr_peak: float = 2.5e-4
    wd: float = 0.01
    clip: float = 1.0
    seed: int = 42
    eval_every: int = 250
    log_every: int = 10
    eval_samples: int = 5_000
    example_every: int = 250
    n_examples: int = 5
    gen_len: int = 100


cfg: Config = tyro.cli(Config)


# ============================== Data loading ============================= #


def load_tinystories(tok, seq_len: int, split: str = "train"):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    def _proc(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"]  # (B, T)
        attn_mask = enc["attention_mask"]  # (B, T)
        eos_id = tok.eos_token_id
        for i in range(input_ids.shape[0]):
            row = input_ids[i]
            idx = np.where(row == eos_id)[0]
            if idx.size > 0:
                eos_pos = int(idx[0])
                attn_mask[i, eos_pos + 1 :] = 0
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    return ds.map(_proc, batched=True, batch_size=1024, remove_columns=["text"])


# ============================== Schedules/metrics ======================== #


def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
    warm = jnp.minimum(step / cfg.warmup, 1.0)
    lr = cfg.lr_peak * warm
    decay_steps = jnp.maximum(cfg.steps - cfg.warmup, 1)
    progress = jnp.clip((step - cfg.warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step >= cfg.warmup, cfg.lr_peak * cos, lr).astype(jnp.float32)


def count_params(tree) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(x.size for x in leaves))


def l2_tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return float(jnp.sqrt(sq + 1e-12))


def router_l2_norm(model: Model) -> float:
    if hasattr(model, "routers"):
        leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
        if not leaves:
            return 0.0
        sq = sum(jnp.sum(jnp.square(x)) for x in leaves)
        return float(jnp.sqrt(sq + 1e-12))
    return 0.0


# ============================== Model factory ============================ #


def make_modules(
    *,
    d_model: int,
    n_heads: int,
    n: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    ks = jax.random.split(key, n)
    mods: List[eqx.Module] = []
    for i in range(n):
        t = i % 3
        if t == 0:
            mods.append(Attention(d_model, n_heads, dropout, key=ks[i]))
        elif t == 1:
            mods.append(FeedForward(d_model, mlp_mult, dropout, key=ks[i]))
        else:
            mods.append(Identity())
    return tuple(mods)


def make_backbone(
    *,
    d_model: int,
    n_heads: int,
    mlp_mult: int,
    dropout: float,
    key: jax.Array,
) -> Tuple[eqx.Module, ...]:
    ks = jax.random.split(key, 2)
    attn = Attention(d_model, n_heads, dropout, key=ks[0])
    ff = FeedForward(d_model, mlp_mult, dropout, key=ks[1])
    return (attn, ff)


def build_model(key: jax.Array) -> Model:
    k_mods, k_bb, k_model = jax.random.split(key, 3)
    mods = make_modules(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n=cfg.n_modules,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        key=k_mods,
    )
    backbone = make_backbone(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        key=k_bb,
    )
    return Model(
        modules=mods,
        vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        capacity=cfg.capacity,
        topk=cfg.topk,
        n_hops=cfg.n_hops,
        dropout=cfg.dropout,
        rope_base=cfg.rope_base,
        backbone=backbone,
        key=k_model,
    )


# ============================== Batch helpers =========================== #


def _normalize_to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape {arr.shape}")


def sample_batch(stream_it, bsz: int) -> Dict[str, jnp.ndarray]:
    ids_buf, mask_buf, total = [], [], 0
    while total < bsz:
        ex = next(stream_it)
        ids = _normalize_to_2d(ex["input_ids"])
        mask = _normalize_to_2d(ex["attention_mask"])
        ids_buf.append(ids)
        mask_buf.append(mask)
        total += ids.shape[0]
    ids = np.concatenate(ids_buf, axis=0)[:bsz].astype(np.int32)  # (B, T)
    mask = np.concatenate(mask_buf, axis=0)[:bsz].astype(np.int32)  # (B, T)
    return {"input_ids": jnp.array(ids), "attention_mask": jnp.array(mask)}


def batch_seq_stats(mask: jnp.ndarray) -> Tuple[float, int, int, float]:
    m = mask[:, 1:]
    lens = jnp.sum(m, axis=1)  # (B,)
    return (
        float(jnp.mean(lens)),
        int(jnp.min(lens)),
        int(jnp.max(lens)),
        float(jnp.mean((cfg.seq_len - 1) - lens)),
    )


# ============================== Forward / Loss =========================== #


def compute_loss(
    model: Model,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    inference: bool = False,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]]:
    ids = batch["input_ids"]  # (B, T)
    mask = batch["attention_mask"]  # (B, T)
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def fwd(x, m, k):
        logits, stats = model(
            x,  # (T,)
            key=k,
            inference=inference,
            attention_mask=m,  # (T,)
            gumbel_tau=cfg.gumbel_tau,
            router_temperature=cfg.router_temp,
            select_temperature=cfg.select_temp,
        )
        return logits, stats

    logits, stats = jax.vmap(fwd, in_axes=(0, 0, 0))(ids, mask, keys)  # (B,T,V)
    logits_shift = logits[:, :-1]  # (B,T-1,V)
    labels_shift = ids[:, 1:]  # (B,T-1)
    mask_shift = mask[:, 1:]  # (B,T-1)

    raw = optax.softmax_cross_entropy_with_integer_labels(
        logits_shift, labels_shift
    )  # (B, T-1)
    loss = (raw * mask_shift).sum() / jnp.maximum(mask_shift.sum(), 1)
    return loss, (logits_shift, labels_shift, mask_shift, stats)


# ============================== Train / Eval steps ======================= #


@eqx.filter_jit
def train_step(
    model: Model,
    opt: optax.GradientTransformation,
    opt_state,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
):
    (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, batch, key, inference=False)

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    preds = jnp.argmax(logits, axis=-1)  # (B, T-1)
    valid = mask > 0
    acc = ((preds == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


@eqx.filter_jit
def eval_step(model: Model, batch: Dict[str, Any], *, key: jax.Array):
    loss, (logits, labels, mask, _stats) = compute_loss(
        model, batch, key, inference=True
    )
    preds = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = ((preds == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    return loss, acc


# ============================== Routing metrics ========================== #


def _avg_over_batch(x) -> jnp.ndarray:
    """Average first axis (batch) -> keep remaining dims."""
    x = jnp.asarray(x)
    return jnp.mean(x, axis=0)


def routing_metrics_from_stats(
    stats_tuple_batched: Tuple[Dict[str, Any], ...],
    *,
    prefix: str,
) -> Dict[str, float]:
    hop_logs = []

    for hop in stats_tuple_batched:
        # Shapes after vmap: load -> (B, E); scalars -> (B,)
        load = jnp.asarray(hop["load"])  # (B, E)
        rho_mean = jnp.asarray(hop["rho_mean"])  # (B,)
        entropy_mean = jnp.asarray(hop["entropy_mean"])  # (B,)
        cap_drop = jnp.asarray(hop["cap_drop_frac_edges"])  # (B,)
        eff_topk_mean = jnp.asarray(hop["eff_topk_mean"])  # (B,)
        eff_topk_min = jnp.asarray(hop["eff_topk_min"])  # (B,)
        eff_topk_max = jnp.asarray(hop["eff_topk_max"])  # (B,)
        cap_util_mean = jnp.asarray(hop["cap_util_mean"])  # (B,)
        cap_util_min = jnp.asarray(hop["cap_util_min"])  # (B,)
        cap_util_max = jnp.asarray(hop["cap_util_max"])  # (B,)

        # Utilization per batch: fraction of experts with load>0
        util_b = jnp.mean((load > 0).astype(jnp.float32), axis=1)  # (B,)

        # Load dispersion per batch: std of normalized loads across experts
        denom = jnp.sum(load, axis=1, keepdims=True) + 1e-9
        load_norm = load / denom  # (B, E)
        load_std_b = jnp.std(load_norm, axis=1)  # (B,)

        # Reduce over batch
        hop_log = dict(
            rho_mean=float(jnp.mean(rho_mean)),
            entropy_mean=float(jnp.mean(entropy_mean)),
            util=float(jnp.mean(util_b)),
            load_std=float(jnp.mean(load_std_b)),
            cap_drop_frac=float(jnp.mean(cap_drop)),
            eff_topk_mean=float(jnp.mean(eff_topk_mean)),
            eff_topk_min=float(jnp.mean(eff_topk_min)),
            eff_topk_max=float(jnp.mean(eff_topk_max)),
            cap_util_mean=float(jnp.mean(cap_util_mean)),
            cap_util_min=float(jnp.mean(cap_util_min)),
            cap_util_max=float(jnp.mean(cap_util_max)),
        )
        hop_logs.append(hop_log)

    # Average across hops
    def _mean_key(k: str) -> float:
        vals = [h[k] for h in hop_logs]
        return float(sum(vals) / max(len(vals), 1))

    return {
        f"routing/{prefix}/rho_mean": _mean_key("rho_mean"),
        f"routing/{prefix}/entropy_mean": _mean_key("entropy_mean"),
        f"routing/{prefix}/util": _mean_key("util"),
        f"routing/{prefix}/load_std": _mean_key("load_std"),
        f"routing/{prefix}/cap_drop_frac": _mean_key("cap_drop_frac"),
        f"routing/{prefix}/eff_topk_mean": _mean_key("eff_topk_mean"),
        f"routing/{prefix}/eff_topk_min": _mean_key("eff_topk_min"),
        f"routing/{prefix}/eff_topk_max": _mean_key("eff_topk_max"),
        f"routing/{prefix}/cap_util_mean": _mean_key("cap_util_mean"),
        f"routing/{prefix}/cap_util_min": _mean_key("cap_util_min"),
        f"routing/{prefix}/cap_util_max": _mean_key("cap_util_max"),
    }


# ============================== Examples (prints) ======================== #


def generate_examples(
    model: Model,
    tok,
    *,
    key: jax.Array,
    gen_len: int = 100,
    per_prompt: int = 1,
    prompts: List[str] | None = None,
) -> None:
    if prompts is None:
        prompts = [
            "One day, ",
            "Once upon a time, ",
            "In a small town, ",
            "Long ago, ",
            "On a sunny morning, ",
        ]

    print("\n" + "=" * 40)
    print("Generated Examples")
    print("=" * 40)

    for p in prompts[: cfg.n_examples]:
        prompt_ids = jnp.array(tok.encode(p), dtype=jnp.int32)  # (T0,)
        key, *subs = jax.random.split(key, per_prompt + 1)
        subs = jnp.stack(subs)

        @jax.vmap
        def _sample(k):
            return sample(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=gen_len,
                temperature=0.8,
                key=k,
                router_temperature=cfg.router_temp,
                select_temperature=cfg.select_temp,
                gumbel_tau=cfg.gumbel_tau,
                greedy=False,
                pad_id=tok.pad_token_id,
                eos_id=tok.pad_token_id,
            )

        toks = _sample(subs)
        for seq in jax.device_get(toks):
            seq = list(seq)
            if tok.eos_token_id in seq:
                seq = seq[: seq.index(tok.eos_token_id) + 1]
            text = tok.decode(seq, skip_special_tokens=True)
            print(f"[{p}] {text}")
            print("-" * 40)


# ============================== Initial stats print ====================== #


def print_initial_stats(
    model: Model,
    first_batch: Dict[str, Any],
    *,
    step0_log_to_wandb: bool = True,
) -> None:
    n_params = count_params(model)
    lmean, lmin, lmax, pmean = batch_seq_stats(first_batch["attention_mask"])
    print("\n" + "=" * 40)
    print("Initial stats")
    print("=" * 40)
    print(f"Params: {n_params:,}")
    print(f"Capacity: {cfg.capacity}  TopK: {cfg.topk}  Hops: {cfg.n_hops}")
    print(f"Seq len mean/min/max (T-1): {lmean:.1f} / {lmin} / {lmax}")
    print(f"Pad mean (T-1): {pmean:.1f}")
    if step0_log_to_wandb:
        wandb.log(
            {
                "n_params": n_params,
                "capacity": cfg.capacity,
                "topk": cfg.topk,
                "hops": cfg.n_hops,
                "seq/len_mean": lmean,
                "seq/len_min": lmin,
                "seq/len_max": lmax,
                "seq/pad_mean": pmean,
                "step": 0,
            }
        )


# ============================== Main ==================================== #


def main():
    wandb.init(project="dna", name="dna", config=asdict(cfg))

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    train_it = iter(load_tinystories(tok, cfg.seq_len, "train"))
    val_it = iter(load_tinystories(tok, cfg.seq_len, "validation"))

    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = build_model(mk)

    first_batch = sample_batch(train_it, cfg.batch_size)
    print_initial_stats(model, first_batch)

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=cfg.wd,
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    for step in range(cfg.steps + 1):
        key, sk = jax.random.split(key)
        batch = sample_batch(train_it, cfg.batch_size)

        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt, opt_state, batch, key=sk
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if step % cfg.log_every == 0:
            stats_host = jax.tree_util.tree_map(jax.device_get, stats)

            train_metrics = {
                "train/loss": float(loss),
                "train/acc": float(acc),
                "train/grad_norm": float(gnorm),
                "train/lr": float(lr_schedule(jnp.array(step))),
                "train/step_ms": dt_ms,
                "train/tok_s": cfg.batch_size * cfg.seq_len / (dt_ms / 1000.0 + 1e-9),
                "w_norm/global": l2_tree_norm(model),
                "w_norm/routers": router_l2_norm(model),
                "step": step,
            }

            route_log = routing_metrics_from_stats(stats_host, prefix="train")

            wandb.log({**train_metrics, **route_log})

        if step % cfg.eval_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            val_batch = sample_batch(val_it, cfg.eval_samples // cfg.seq_len)
            val_loss, val_acc = eval_step(model, val_batch, key=ek)

            lmean, lmin, lmax, pmean = batch_seq_stats(val_batch["attention_mask"])

            wandb.log(
                {
                    "eval/loss": float(val_loss),
                    "eval/acc": float(val_acc),
                    "step": step,
                }
            )

        if step % cfg.example_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            generate_examples(
                model,
                tok,
                key=ek,
                gen_len=cfg.gen_len,
                per_prompt=1,
            )


if __name__ == "__main__":
    main()
