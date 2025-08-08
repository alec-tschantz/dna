import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

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
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 8
    n_hops: int = 6
    n_backbone: int = 2
    n_modules: int = 16
    topk: int = 2
    capacity: int = 64  # per trainable expert (identity excluded)
    mlp_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0

    # Router controls
    router_temp: float = 1.0  # temperature for router softmax
    gumbel: bool = True  # use Gumbel-Top-k during training
    gumbel_tau: float = 1.0  # Gumbel noise scale

    # Bias control (trainer-driven, identity-only)
    identity_bias_init: float = 0.0  # initial identity bias per hop (scalar)
    bias_lr: float = 0.1  # additive sign-update step size
    id_target_frac: float = 0.0  # target fraction of routes that are identity
    bias_clip: float = 6.0  # clamp biases into [-bias_clip, +bias_clip]

    # ---------------- training ----------------
    batch_size: int = 32
    seq_len: int = 256
    steps: int = 20_000
    warmup: int = 2_000
    lr_peak: float = 2.5e-4
    wd: float = 0.01
    clip: float = 1.0
    seed: int = 42

    # ---------------- logging / eval ----------------
    eval_every: int = 250
    log_every: int = 10
    eval_samples: int = 1_000
    example_every: int = 250
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
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    return ds.map(_proc, batched=True, batch_size=1_024, remove_columns=["text"])


def sample_batch(it, bsz: int):
    batch = [next(it) for _ in range(bsz)]
    ids = jnp.stack([jnp.array(b["input_ids"], dtype=jnp.int32) for b in batch])
    amask = jnp.stack([jnp.array(b["attention_mask"], dtype=jnp.int32) for b in batch])
    return {"input_ids": ids, "attention_mask": amask}


# ───────────────────────────────────── LR schedule  ------────────────────────────────────── #


def lr_schedule(step: jnp.ndarray):
    warmup_ratio = jnp.minimum(step / cfg.warmup, 1.0)
    lr = cfg.lr_peak * warmup_ratio
    decay_steps = jnp.maximum(cfg.steps - cfg.warmup, 1)
    progress = jnp.clip((step - cfg.warmup) / decay_steps, 0.0, 1.0)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(step >= cfg.warmup, cfg.lr_peak * cosine_decay, lr)
    return jnp.array(lr, dtype=jnp.float32)


# ───────────────────────────────────── Loss & step fns ────────────────────────────────────── #


def compute_loss(
    model: DNA,
    batch: Dict[str, Any],
    biases_id: jnp.ndarray,
    key,
    *,
    inference: bool = False,
):
    ids = batch["input_ids"]  # [B, T]
    mask = batch["attention_mask"]  # [B, T] (1=token, 0=pad). Only for loss mask.

    B = ids.shape[0]
    keys = jax.random.split(key, B)

    def fwd(x, k):
        return model(
            x,
            key=k,
            inference=inference,
            biases=biases_id,  # [n_hops] identity-only biases
            gumbel=(cfg.gumbel and not inference),
            gumbel_tau=cfg.gumbel_tau,
            temp=cfg.router_temp,
        )

    vmap_model = jax.vmap(fwd)
    logits, stats = vmap_model(
        ids, keys
    )  # logits [B,T,V]; stats: tuple(hops) dicts with [B,...] arrays

    logits_shift = logits[:, :-1]  # [B, T-1, V]
    labels_shift = ids[:, 1:]  # [B, T-1]
    mask_shift = mask[:, 1:]  # [B, T-1]

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

    preds = jnp.argmax(logits, -1)
    correct = (preds == labels) * mask
    acc = correct.sum() / jnp.maximum(mask.sum(), 1)

    gnorm = optax.global_norm(grads)
    return model, opt_state, loss, acc, stats, gnorm


@eqx.filter_jit
def eval_step(model, batch, biases_id, *, key):
    loss, (logits, labels, mask, _) = compute_loss(
        model, batch, biases_id, key, inference=True
    )
    preds = jnp.argmax(logits, -1)
    correct = (preds == labels) * mask
    acc = correct.sum() / jnp.maximum(mask.sum(), 1)
    return loss, acc


# ───────────────────────────────────── Bias control ───────────────────────────────────────── #


def init_biases_id(n_hops: int, init_identity_bias: float) -> jnp.ndarray:
    """Return [n_hops] scalar identity biases (last expert per hop)."""
    return jnp.full((n_hops,), init_identity_bias, dtype=jnp.float32)


def update_biases_from_stats_id(
    biases_id: jnp.ndarray,
    stats_tuple,
    *,
    topk: int,  # k in the paper
    target_skip_frac: float,  # r in the paper (desired fraction of routes that go to identity)
    n_experts_total: int,  # |⋆| in the paper = total experts INCLUDING identity
    lr: float,  # u in the paper (update step size)
    clip: float | None = None,
) -> jnp.ndarray:
    """
    Paper-spec identity bias update (Eq. 5 in DNA).
    We keep this **outside** the gradient path and apply it to the identity bias per hop.

    Required stats per hop (pre-capacity, per batch element):
      - hop["id_topk_count"]: [B]  number of tokens that had identity in top-k
      - hop["total_routes"]:  [B]  total number of token→expert routes selected (= k * valid_tokens)

    Update (per hop s):
        b_s <- b_s + u * sign( r * k * cbar_s - id_count_s )
      where
        cbar_s = (sum_i c_i^{(s)}) / n_experts_total
        sum_i c_i^{(s)} == total_routes_s  (pre-capacity, across ALL experts incl. identity)

    Notes
    -----
    * Only identity biases are updated (b_s here).
    * This mirrors their “bias trick” applied to top-k selection only; the softmax probs are unchanged.
      (cf. “probabilities via softmax; routing decision made by sampling with hard top-k,” and then
       “modify the top-k selection via i ∈ top-k(ρ⋆ + b⋆); biases are non-zero only for identity.”) :contentReference[oaicite:1]{index=1}
    """
    if lr == 0.0:
        return biases_id

    b = biases_id
    for s, hop in enumerate(stats_tuple):
        # Aggregate across the batch.
        id_count = jnp.sum(hop["id_topk_count"].astype(jnp.int32)).astype(
            jnp.float32
        )  # ∑_{i∈Id} c_i^{(s)}
        total_routes = jnp.sum(hop["total_routes"].astype(jnp.int32)).astype(
            jnp.float32
        )  # ∑_i c_i^{(s)}

        # Average count per expert, including identity(s):
        cbar = total_routes / jnp.maximum(n_experts_total, 1)

        # Target = r * k * c̄^{(s)}  (Eq. 5)
        target = target_skip_frac * float(topk) * cbar

        step = lr * jnp.sign(target - id_count)
        b = b.at[s].add(step)

    if clip is not None and clip > 0:
        b = jnp.clip(b, -clip, clip)

    return b


# ───────────────────────────────────── Weight norms ───────────────────────────────────────── #


def l2_tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    sq = sum([jnp.sum(jnp.square(x)) for x in leaves]) if leaves else jnp.array(0.0)
    return float(jnp.sqrt(sq + 1e-12))


def router_l2_norm(model: DNA) -> float:
    # Only the router weights (proj matrices)
    leaves = jax.tree_util.tree_leaves(eqx.filter(model.routers, eqx.is_array))
    if not leaves:
        return 0.0
    sq = sum([jnp.sum(jnp.square(x)) for x in leaves])
    return float(jnp.sqrt(sq + 1e-12))


# ───────────────────────────────────── Logging utils ────────────────────────────────────── #


def log_metrics(
    step: int,
    metrics: Dict[str, float],
    stats_tuple,
    biases_id: jnp.ndarray,
    prefix: str,
):
    """
    stats_tuple is a tuple (len = n_hops) of dicts.
    Each value in a dict is a batched array over B (because we vmapped the model).
    We aggregate over batch here.
    """
    log = {f"{prefix}/{k}": v for k, v in metrics.items()}
    log["step"] = step

    if stats_tuple and isinstance(stats_tuple[0], dict):
        rho_means = []
        id_rates = []
        ent_means = []
        load_stds = []
        utils = []
        cap_drop_fracs = []

        def _mean_over_batch(x):
            return jnp.mean(x, axis=0)

        for hop in stats_tuple:
            load = _mean_over_batch(hop["load"])  # [E]
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

    # Bias + router hyperparams
    log["bias/id_mean"] = float(jnp.mean(biases_id))
    log["bias/id_min"] = float(jnp.min(biases_id))
    log["bias/id_max"] = float(jnp.max(biases_id))
    log["router/temp"] = float(cfg.router_temp)
    log["router/gumbel_tau"] = float(cfg.gumbel_tau)
    log["router/gumbel"] = float(cfg.gumbel)

    wandb.log(log)


# ───────────────────────────────────── Generation util ────────────────────────────────────── #


def generate_examples(
    model, tok, prompt="One day, ", n=5, gen_len=50, *, key, biases_id
):
    prompt_ids = jnp.array(tok.encode(prompt), dtype=jnp.int32)
    key, *subs = jax.random.split(key, n + 1)
    subs = jnp.stack(subs)

    @jax.vmap
    def _sample(k):
        return generate(
            model,
            prompt_ids,
            gen_len,
            1.0,
            key=k,
            biases=biases_id,
            gumbel=False,
            temp=cfg.router_temp,
        )

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

    # Identity bias buffer (trainer-controlled; outside grad). Shape [n_hops]
    biases_id = init_biases_id(cfg.n_hops, cfg.identity_bias_init)

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
            b2=0.999,
            eps=1e-8,
            weight_decay=cfg.wd,
        ),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # --- training loop ------------------------------------------------------
    for step in range(cfg.steps + 1):
        key, sk = jax.random.split(key)
        batch = sample_batch(train_it, cfg.batch_size)

        # One training step
        t0 = time.perf_counter()
        model, opt_state, loss, acc, stats, gnorm = train_step(
            model, opt_state, batch, biases_id, key=sk
        )
        dt_ms = (time.perf_counter() - t0) * 1000

        # Update identity biases outside JIT/grad, using routing stats
        stats_host = jax.tree_util.tree_map(lambda x: jax.device_get(x), stats)
        biases_id = update_biases_from_stats_id(
            biases_id,
            stats_host,
            topk=cfg.topk,
            target_skip_frac=cfg.id_target_frac,
            n_experts_total=cfg.n_modules + 1,  # include identity!
            lr=cfg.bias_lr,
            clip=cfg.bias_clip,
        )

        if step % cfg.log_every == 0:
            # Weight norms
            global_param_norm = l2_tree_norm(model)
            router_param_norm = router_l2_norm(model)

            metrics = {
                "loss": float(loss),
                "acc": float(acc),
                "lr": float(lr_schedule(jnp.array(step))),
                "grad_norm": float(gnorm),
                "step_ms": dt_ms,
                "tok_s": cfg.batch_size * cfg.seq_len / (dt_ms / 1000 + 1e-9),
                "w_norm/global": global_param_norm,
                "w_norm/routers": router_param_norm,
            }
            log_metrics(step, metrics, stats_host, biases_id, "train")

        if step % cfg.eval_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            val_batch = sample_batch(val_it, cfg.eval_samples // cfg.seq_len)
            val_loss, val_acc = eval_step(model, val_batch, biases_id, key=ek)
            wandb.log(
                {
                    "eval/loss": float(val_loss),
                    "eval/acc": float(val_acc),
                    "step": step,
                }
            )

        if step % cfg.example_every == 0 and step > 0:
            key, ek = jax.random.split(key)
            samples = generate_examples(
                model,
                tok,
                n=cfg.n_examples,
                gen_len=cfg.gen_len,
                key=ek,
                biases_id=biases_id,
            )
            print(f"\n========== SAMPLES @ step {step} ==========")
            for i, s in enumerate(samples):
                print(f"\n--- Example {i} ---\n{s}\n")
            wandb.log({"examples": "\n\n".join(samples), "step": step})


if __name__ == "__main__":
    main()
