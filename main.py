import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from dna import Dense, DNA, generate


@dataclass
class Config:
    # ---------------- model ----------------
    model_type: str = "dna"
    vocab_size: int = 50_257
    d_model: int = 256
    n_heads: int = 16
    n_layers: int = 8
    n_hops: int = 8
    n_backbone: int = 2
    n_modules: int = 16
    topk: int = 1
    capacity: int = 32
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0
    identity_bias: float = 0.1

    # ---------------- training ----------------
    batch_size: int = 32
    seq_len: int = 256
    steps: int = 40_000
    warmup: int = 1_000
    lr_init: float = 1e-7
    lr_peak: float = 1e-6
    lr_final_mult: float = 0.1
    wd: float = 0.01
    clip: float = 1.0
    seed: int = 42

    # ---------------- logging / eval ----------------
    eval_every: int = 1_000
    log_every: int = 100
    eval_samples: int = 1_000
    example_every: int = 500
    n_examples: int = 5
    gen_len: int = 100


cfg: Config = tyro.cli(Config)

# ───────────────────────────────────── Dataset helpers ────────────────────────────────────── #


def load_tinystories(tok, seq_len: int, split="train"):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    def _proc(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="np",
        )
        return {"input_ids": enc["input_ids"]}

    return ds.map(_proc, batched=True, batch_size=1_024, remove_columns=["text"])


def sample_batch(it, bsz: int):
    batch = [next(it) for _ in range(bsz)]
    ids = jnp.stack([jnp.array(b["input_ids"], dtype=jnp.int32) for b in batch])
    return {"input_ids": ids}


# ───────────────────────────────────── LR schedule  ------────────────────────────────────── #


def lr_schedule(step: jnp.ndarray):
    lr = cfg.lr_init + (cfg.lr_peak - cfg.lr_init) * jnp.minimum(step / cfg.warmup, 1.0)
    plateau_steps = int(cfg.steps * 0.8)
    lr = jnp.where(step <= plateau_steps, lr, cfg.lr_peak)
    decay_progress = (step - plateau_steps) / max(1, cfg.steps - plateau_steps)
    decay_lr = cfg.lr_peak - decay_progress * cfg.lr_peak * (1 - cfg.lr_final_mult)
    lr = jnp.where(step > plateau_steps, decay_lr, lr)
    return lr


# ───────────────────────────────────── Loss & step fns ────────────────────────────────────── #


def compute_loss(model, batch: Dict[str, Any], key, *, inference: bool = False):
    ids = batch["input_ids"]  # [B, T]
    B = ids.shape[0]
    keys = jax.random.split(key, B)
    vmap_model = jax.vmap(lambda x, k: model(x, key=k, inference=inference))
    logits, stats = vmap_model(ids, keys)

    logits_shift, labels_shift = logits[:, :-1], ids[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift).mean()
    return loss, (logits_shift, labels_shift, stats)


@eqx.filter_jit
def train_step(model, opt_state, batch, *, key):
    (loss, (logits, labels, stats)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(model, batch, key)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    acc = (jnp.argmax(logits, -1) == labels).mean()
    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


@eqx.filter_jit
def eval_step(model, batch, *, key):
    loss, (logits, labels, _) = compute_loss(model, batch, key, inference=True)
    acc = (jnp.argmax(logits, -1) == labels).mean()
    return loss, acc


# ─────────────────────────────────────── Logging utils ────────────────────────────────────── #


def log_metrics(step: int, metrics: Dict[str, float], stats, prefix: str):
    log = {f"{prefix}/{k}": v for k, v in metrics.items()}
    log["step"] = step

    if stats and isinstance(stats[0], dict):
        for h, hop in enumerate(stats):
            load, imp = hop["load"].mean(0), hop["importance"].mean(0)
            ln = load / jnp.clip(load.sum(), 1.0)
            inorm = imp / jnp.clip(imp.sum(), 1.0)
            entropy = -(inorm * jnp.log(inorm + 1e-9)).sum() / math.log(inorm.shape[0])
            log.update(
                {
                    f"routing/{prefix}/hop{h}_load_std": float(jnp.std(ln)),
                    f"routing/{prefix}/hop{h}_imp_std": float(jnp.std(inorm)),
                    f"routing/{prefix}/hop{h}_dropped": float(hop["dropped"].mean()),
                    f"routing/{prefix}/hop{h}_util": float((load > 0).mean()),
                    f"routing/{prefix}/hop{h}_entropy": float(entropy),
                    # f"routing/{prefix}/hop{h}_conf": float(
                    #     hop["router_probs"].mean(0).max(-1).mean()
                    # ),
                }
            )
    wandb.log(log)


# ───────────────────────────────────── Generation util ────────────────────────────────────── #


def generate_examples(model, tok, prompt="One day, ", n=5, gen_len=50, *, key):
    prompt_ids = jnp.array(tok.encode(prompt), dtype=jnp.int32)
    key, *subs = jax.random.split(key, n + 1)
    subs = jnp.stack(subs)

    @jax.vmap
    def _sample(k):
        return generate(model, prompt_ids, gen_len, 1.0, key=k)

    out = _sample(subs)
    decoded = []
    for seq in jax.device_get(out):
        seq = list(seq)
        if tok.eos_token_id in seq:
            seq = seq[: seq.index(tok.eos_token_id) + 1]
        decoded.append(tok.decode(seq, skip_special_tokens=True))
    return decoded


# ───────────────────────────────────── Build model ────────────────────────────────────────── #


def build_model(key):
    if cfg.model_type == "dense":
        model = Dense(
            vocab=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            mlp_mult=cfg.mlp_mult,
            dropout=cfg.dropout,
            rope_base=cfg.rope_base,
            key=key,
        )
    else:
        model = DNA(
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
            identity_bias=cfg.identity_bias,
            key=key,
        )
    return model


# ────────────────────────────────────────── Main ─────────────────────────────────────────── #


def main():
    wandb.init(project=f"dense-moe", name=cfg.model_type, config=asdict(cfg))

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    train_it = iter(load_tinystories(tok, cfg.seq_len, "train"))
    val_it = iter(load_tinystories(tok, cfg.seq_len, "validation"))

    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    model = build_model(mk)

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    wandb.log({"n_params": n_params, "capacity": cfg.capacity, "step": 0})

    global opt
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-15,
            weight_decay=cfg.wd,
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # --- training loop ------------------------------------------------------
    for step in range(cfg.steps + 1):
        key, sk = jax.random.split(key)
        batch = sample_batch(train_it, cfg.batch_size)

        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt_state, batch, key=sk
        )
        dt_ms = (time.perf_counter() - t0) * 1000

        if step % cfg.log_every == 0:
            log_metrics(
                step,
                {
                    "loss": float(loss),
                    "acc": float(acc),
                    "lr": float(lr_schedule(jnp.array(step))),
                    "grad_norm": float(gnorm),
                    "step_ms": dt_ms,
                    "tok_s": cfg.batch_size * cfg.seq_len / (dt_ms / 1000 + 1e-9),
                },
                stats,
                "train",
            )

        if step % cfg.eval_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            val_batch = sample_batch(val_it, cfg.eval_samples // cfg.seq_len)
            val_loss, val_acc = eval_step(model, val_batch, key=ek)
            log_metrics(
                step, {"loss": float(val_loss), "acc": float(val_acc)}, [], "eval"
            )

        if step % cfg.example_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            samples = generate_examples(
                model, tok, n=cfg.n_examples, gen_len=cfg.gen_len, key=ek
            )
            print(f"\n========== SAMPLES @ step {step} ==========")
            for i, s in enumerate(samples):
                print(f"\n--- Example {i} ---\n{s}\n")
            wandb.log({"examples": "\n\n".join(samples), "step": step})


if __name__ == "__main__":
    main()
