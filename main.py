import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from dna import Dense, DNA, generate


@dataclass
class Config:
    model_type: str = "dna"
    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 8
    n_hops: int = 6
    n_backbone: int = 2
    n_modules: int = 6
    topk: int = 2
    capacity: int = 128
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0
    router_temp: float = 1.0
    gumbel: bool = True
    gumbel_tau: float = 0.3
    identity_bias_init: float = 0.0
    bias_lr: float = 0.02
    id_target_frac: float = 0.1
    bias_clip: float = 1.0
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
    eval_samples: int = 2000
    example_every: int = 250
    n_examples: int = 5
    gen_len: int = 100


cfg: Config = tyro.cli(Config)


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
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        eos_id = tok.eos_token_id
        for i in range(input_ids.shape[0]):
            row = input_ids[i]
            idx = np.where(row == eos_id)[0]
            if idx.size > 0:
                eos_pos = int(idx[0])
                attn_mask[i, eos_pos + 1 :] = 0
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    return ds.map(_proc, batched=True, batch_size=1024, remove_columns=["text"])


def _normalize_to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape {arr.shape}")


def sample_batch(stream_it, bsz: int):
    ids_buf, mask_buf, total = [], [], 0
    while total < bsz:
        ex = next(stream_it)
        ids = _normalize_to_2d(ex["input_ids"])
        mask = _normalize_to_2d(ex["attention_mask"])
        ids_buf.append(ids)
        mask_buf.append(mask)
        total += ids.shape[0]
    ids = np.concatenate(ids_buf, axis=0)[:bsz].astype(np.int32)
    mask = np.concatenate(mask_buf, axis=0)[:bsz].astype(np.int32)
    return {"input_ids": jnp.array(ids), "attention_mask": jnp.array(mask)}


def lr_schedule(step: jnp.ndarray):
    warm = jnp.minimum(step / cfg.warmup, 1.0)
    lr = cfg.lr_peak * warm
    decay_steps = jnp.maximum(cfg.steps - cfg.warmup, 1)
    progress = jnp.clip((step - cfg.warmup) / decay_steps, 0.0, 1.0)
    cos = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step >= cfg.warmup, cfg.lr_peak * cos, lr).astype(jnp.float32)


def compute_loss(
    model,
    batch: Dict[str, Any],
    biases_id: jnp.ndarray,
    key,
    *,
    inference: bool = False,
):
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def fwd(x, m, k):
        return model(
            x,
            key=k,
            inference=inference,
            attention_mask=m,
            biases=biases_id,
            gumbel=(cfg.gumbel and not inference),
            gumbel_tau=cfg.gumbel_tau,
            temp=cfg.router_temp,
        )

    logits, stats = jax.vmap(fwd)(ids, mask, keys)
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = mask[:, 1:]
    raw_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_shift, labels_shift
    )
    loss = (raw_loss * mask_shift).sum() / jnp.maximum(mask_shift.sum(), 1)
    return loss, (logits_shift, labels_shift, mask_shift, stats)


@eqx.filter_jit
def train_step(model, opt_state, batch, biases_id, *, key):
    (loss, (logits, labels, mask, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, batch, biases_id, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    preds = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = ((preds == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


@eqx.filter_jit
def eval_step(model, batch, biases_id, *, key):
    loss, (logits, labels, mask, _) = compute_loss(
        model, batch, biases_id, key, inference=True
    )
    preds = jnp.argmax(logits, axis=-1)
    valid = mask > 0
    acc = ((preds == labels) & valid).sum() / jnp.maximum(valid.sum(), 1)
    return loss, acc


def init_biases_id(n_hops: int, init_identity_bias: float) -> jnp.ndarray:
    return jnp.full((n_hops,), init_identity_bias, dtype=jnp.float32)


def update_biases_from_stats_id(
    biases_id: jnp.ndarray,
    stats_tuple,
    *,
    topk: int,
    target_skip_frac: float,
    n_experts_total: int,
    lr: float,
    clip: float | None = None,
) -> jnp.ndarray:
    if lr == 0.0:
        return biases_id
    b = biases_id
    for s, hop in enumerate(stats_tuple):
        id_count = jnp.sum(hop["id_topk_count"].astype(jnp.int32)).astype(jnp.float32)
        total_routes = jnp.sum(hop["total_routes"].astype(jnp.int32)).astype(
            jnp.float32
        )
        cbar = total_routes / jnp.maximum(n_experts_total, 1)
        target = target_skip_frac * float(topk) * cbar
        step = lr * jnp.sign(target - id_count)
        b = b.at[s].add(step)
    if clip is not None and clip > 0:
        b = jnp.clip(b, -clip, clip)
    return b


def l2_tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    sq = sum([jnp.sum(jnp.square(x)) for x in leaves]) if leaves else jnp.array(0.0)
    return float(jnp.sqrt(sq + 1e-12))


def router_l2_norm(model: DNA) -> float:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum([jnp.sum(jnp.square(x)) for x in leaves])
    return float(jnp.sqrt(sq + 1e-12))


def log_metrics(
    step: int,
    metrics: Dict[str, float],
    stats_tuple,
    biases_id: jnp.ndarray,
    prefix: str,
):
    log = {f"{prefix}/{k}": v for k, v in metrics.items()}
    log["step"] = step
    if stats_tuple and isinstance(stats_tuple[0], dict):
        rho_means, id_rates, ent_means, load_stds, utils, cap_drop_fracs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        def _mean_over_batch(x):
            return jnp.mean(x, axis=0)

        for hop in stats_tuple:
            load = _mean_over_batch(hop["load"])
            load_std = float(jnp.std(load / jnp.clip(jnp.sum(load), 1.0)))
            utils.append(float(jnp.mean(load > 0)))
            rho_means.append(float(jnp.mean(_mean_over_batch(hop["rho_mean"]))))
            id_rates.append(float(jnp.mean(_mean_over_batch(hop["id_topk_rate"]))))
            ent_means.append(float(jnp.mean(_mean_over_batch(hop["entropy_mean"]))))
            cap_drop_fracs.append(
                float(jnp.mean(_mean_over_batch(hop["cap_drop_frac_edges"])))
            )
            load_stds.append(load_std)
        log.update(
            {
                f"routing/{prefix}/rho_mean": float(
                    sum(rho_means) / max(len(rho_means), 1)
                ),
                f"routing/{prefix}/id_rate": float(
                    sum(id_rates) / max(len(id_rates), 1)
                ),
                f"routing/{prefix}/entropy_mean": float(
                    sum(ent_means) / max(len(ent_means), 1)
                ),
                f"routing/{prefix}/load_std": float(
                    sum(load_stds) / max(len(load_stds), 1)
                ),
                f"routing/{prefix}/util": float(sum(utils) / max(len(utils), 1)),
                f"routing/{prefix}/cap_drop_frac": float(
                    sum(cap_drop_fracs) / max(len(cap_drop_fracs), 1)
                ),
            }
        )
    log["bias/id_mean"] = float(jnp.mean(biases_id))
    log["bias/id_min"] = float(jnp.min(biases_id))
    log["bias/id_max"] = float(jnp.max(biases_id))
    log["router/temp"] = float(cfg.router_temp)
    log["router/gumbel_tau"] = float(cfg.gumbel_tau)
    log["router/gumbel"] = float(cfg.gumbel)
    wandb.log(log)


def generate_examples(
    model,
    tok,
    prompts=None,
    per_prompt: int = 1,
    gen_len: int = 100,
    *,
    key,
    biases_id,
):
    if prompts is None:
        prompts = [
            "One day, ",
            "Once upon a time, ",
            "In a small town, ",
            "Long ago, ",
            "On a sunny morning, ",
        ]

    out_texts = []
    for p in prompts:
        prompt_ids = jnp.array(tok.encode(p), dtype=jnp.int32)
        key, *subs = jax.random.split(key, per_prompt + 1)
        subs = jnp.stack(subs)

        @jax.vmap
        def _sample(k):
            return generate(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=gen_len,
                temperature=0.8,
                key=k,
                biases=biases_id,
                gumbel=False,
                gumbel_tau=1.0,
                router_temp=cfg.router_temp,
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
            out_texts.append(f"[{p}] {text}")

    return out_texts


def build_model(key):
    if cfg.model_type == "dense":
        return Dense(
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            key=key,
        )
    return DNA(
        vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_modules=cfg.n_modules,
        capacity=cfg.capacity,
        topk=cfg.topk,
        n_hops=cfg.n_hops,
        mlp_mult=cfg.mlp_mult,
        dropout=cfg.dropout,
        rope_base=cfg.rope_base,
        num_backbone=cfg.n_backbone,
        key=key,
    )


def seq_len_stats(mask: jnp.ndarray):
    m = mask[:, 1:]
    lens = jnp.sum(m, axis=1)
    return (
        float(jnp.mean(lens)),
        int(jnp.min(lens)),
        int(jnp.max(lens)),
        float(jnp.mean((cfg.seq_len - 1) - lens)),
    )


def main():
    wandb.init(project="dense-moe", name=cfg.model_type, config=asdict(cfg))
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    train_it = iter(load_tinystories(tok, cfg.seq_len, "train"))
    val_it = iter(load_tinystories(tok, cfg.seq_len, "validation"))
    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = build_model(mk)
    biases_id = init_biases_id(cfg.n_hops, cfg.identity_bias_init)
    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    wandb.log({"n_params": n_params, "capacity": cfg.capacity, "step": 0})
    global opt
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=cfg.wd
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    for step in range(cfg.steps + 1):
        key, sk = jax.random.split(key)
        batch = sample_batch(train_it, cfg.batch_size)
        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt_state, batch, biases_id, key=sk
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        stats_host = jax.tree_util.tree_map(lambda x: jax.device_get(x), stats)
        biases_id = update_biases_from_stats_id(
            biases_id,
            stats_host,
            topk=cfg.topk,
            target_skip_frac=cfg.id_target_frac,
            n_experts_total=cfg.n_modules + 1,
            lr=cfg.bias_lr,
            clip=cfg.bias_clip,
        )

        if step % cfg.log_every == 0:
            lmean, lmin, lmax, pmean = seq_len_stats(batch["attention_mask"])
            metrics = {
                "loss": float(loss),
                "acc": float(acc),
                "lr": float(lr_schedule(jnp.array(step))),
                "grad_norm": float(gnorm),
                "step_ms": dt_ms,
                "tok_s": cfg.batch_size * cfg.seq_len / (dt_ms / 1000.0 + 1e-9),
                "w_norm/global": l2_tree_norm(model),
                "w_norm/routers": router_l2_norm(model),
                "seq/len_mean": lmean,
                "seq/len_min": lmin,
                "seq/len_max": lmax,
                "seq/pad_mean": pmean,
            }
            log_metrics(step, metrics, stats_host, biases_id, "train")

        if step % cfg.eval_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            val_batch = sample_batch(val_it, cfg.eval_samples // cfg.seq_len)
            val_loss, val_acc = eval_step(model, val_batch, biases_id, key=ek)
            lmean, lmin, lmax, pmean = seq_len_stats(val_batch["attention_mask"])
            wandb.log(
                {
                    "eval/loss": float(val_loss),
                    "eval/acc": float(val_acc),
                    "eval/seq/len_mean": lmean,
                    "eval/seq/len_min": lmin,
                    "eval/seq/len_max": lmax,
                    "eval/seq/pad_mean": pmean,
                    "step": step,
                }
            )

        if step % cfg.example_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            samples = generate_examples(
                model,
                tok,
                gen_len=cfg.gen_len,
                key=ek,
                biases_id=biases_id,
            )

            # Log to W&B
            wandb.log({"examples": "\n\n".join(samples), "step": step})

            # Also print to terminal
            print("\n" + "=" * 40)
            print(f"Step {step} — Generated Examples")
            print("=" * 40)
            for i, s in enumerate(samples, 1):
                print(f"[{i}] {s}")
                print("-" * 40)


if __name__ == "__main__":
    main()
