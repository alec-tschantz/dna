# train.py
from __future__ import annotations

import time, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, NamedTuple
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, config as jax_config
import numpy as np
import equinox as eqx
import optax
import tyro
import wandb

from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from model import Transformer
from sample import sample_tokens
from dataloader import setup_tokenizer_and_streams, sample_batch

jax_config.update("jax_default_matmul_precision", "tensorfloat32")


# -------------------------
# Config / State
# -------------------------
@dataclass
class Config:
    vocab_size: int = 50_257
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    rope_base: float = 10_000.0
    batch_size: int = 16
    seq_len: int = 256
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    steps: int = 50_000
    warmup_steps: int = 2_000
    lr_init: float = 1e-6
    lr_peak: float = 3e-4
    lr_end: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 500
    log_every: int = 50
    save_every: int = 5_000
    eval_samples: int = 2048
    gen_max_new: int = 200
    gen_temperature: float = 0.8
    ckpt_dir: str = "checkpoints"
    wandb_project: str = "pjit-tests"


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    key: jax.Array


# -------------------------
# Mesh / Sharding helpers
# -------------------------
def create_mesh(n_devices: Optional[int] = None) -> Mesh:
    devs = jax.devices()
    if n_devices is not None:
        devs = devs[:n_devices]
    device_array = mesh_utils.create_device_mesh((len(devs),), devices=devs)
    return Mesh(device_array, ("data",))


def shard_batch(batch: Dict[str, np.ndarray], mesh: Mesh) -> Dict[str, jax.Array]:
    data_shard = NamedSharding(mesh, P("data", None))
    ids = jax.device_put(jnp.asarray(batch["input_ids"], dtype=jnp.int32), data_shard)
    mask = jax.device_put(jnp.asarray(batch["attention_mask"], dtype=jnp.bool_), data_shard)
    return {"input_ids": ids, "attention_mask": mask}


# -------------------------
# IO / Checkpoint
# -------------------------
def save_checkpoint(
    path: Path,
    step: int,
    params,
    static,
    opt_state,
    config: Config,
    metrics: Optional[Dict] = None,
):
    path.mkdir(parents=True, exist_ok=True)
    params_host = jax.tree_util.tree_map(jax.device_get, params)
    static_host = jax.tree_util.tree_map(jax.device_get, static)
    opt_state_host = jax.tree_util.tree_map(jax.device_get, opt_state)
    model = eqx.combine(params_host, static_host)
    eqx.tree_serialise_leaves(path / f"model_{step}.eqx", model)
    eqx.tree_serialise_leaves(path / f"opt_{step}.eqx", opt_state_host)
    meta = {**asdict(config), "step": step, "timestamp": datetime.now().isoformat()}
    (path / f"meta_{step}.json").write_text(json.dumps(meta, indent=2))


# -------------------------
# Losses (local sums) and forward
# -------------------------
def _per_shard_loss_sum_and_stats(
    params, static, ids: jnp.ndarray, mask_bool: jnp.ndarray, key: jax.Array, *, inference: bool
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Returns *unnormalized* loss sum on this shard and local stats.

    ids:  [B, T], int32
    mask: [B, T], bool
    """
    model = eqx.combine(params, static)

    B = ids.shape[0]
    row_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(B, dtype=jnp.uint32))

    def f(ids_row, mask_row, k):
        return model(ids_row, mask_row, key=k, inference=inference)

    logits = jax.vmap(f)(ids, mask_bool, row_keys)
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift = mask_bool[:, 1:]            # bool
    mask_f = mask_shift.astype(jnp.float32)  # 0/1 for weighting

    loss_tok = optax.softmax_cross_entropy_with_integer_labels(logits_shift, labels_shift)
    loss_sum = (loss_tok * mask_f).sum()

    preds = jnp.argmax(logits_shift, axis=-1)
    n_tokens = mask_f.sum()
    n_correct = ((preds == labels_shift) & mask_shift).sum()

    aux = {
        "n_tokens": n_tokens,
        "n_correct": n_correct,
    }
    return loss_sum, aux


# -------------------------
# Train step (differentiate the GLOBAL mean inside!)
# -------------------------
def build_train_step(optimizer, mesh: Mesh, static):
    def global_mean_loss(params, ids, mask_bool, key):
        # Local sums on each shard
        loss_sum_local, aux_local = _per_shard_loss_sum_and_stats(
            params, static, ids, mask_bool, key, inference=False
        )
        n_tok_local = aux_local["n_tokens"].astype(jnp.float32)

        # Global reductions (identical scalars on all shards)
        loss_sum_global = lax.psum(loss_sum_local, axis_name="data")
        n_tok_global = lax.psum(n_tok_local, axis_name="data")

        loss_global = loss_sum_global / jnp.maximum(n_tok_global, 1.0)

        # For metrics we’ll also want global n_correct
        n_correct_global = lax.psum(aux_local["n_correct"], axis_name="data")

        aux_out = {
            "n_tokens": n_tok_global,
            "n_correct": n_correct_global,
            "loss_global": loss_global,
        }
        return loss_global, aux_out

    # Autodiff sees psum and division by a constant -> gives correctly-scaled global gradient
    loss_and_grad = eqx.filter_value_and_grad(global_mean_loss, has_aux=True)

    aux_spec = {
        "n_tokens": P(),
        "n_correct": P(),
        "accuracy": P(),
        "perplexity": P(),
        "grad_norm": P(),
    }

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("data", None), P("data", None)),
        out_specs=(P(), P(), aux_spec),
    )
    def step_fn(state: TrainState, batch_ids, batch_mask_bool):
        # Per-replica RNG
        key, new_key = jax.random.split(state.key)
        key = jax.random.fold_in(key, lax.axis_index("data"))

        (loss_global, aux), grads = loss_and_grad(state.params, batch_ids, batch_mask_bool, key)
        # grads are already for the *global* mean loss; identical across replicas.

        grad_norm = optax.global_norm(grads)

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = eqx.apply_updates(state.params, updates)
        new_state = TrainState(params=new_params, opt_state=new_opt_state, key=new_key)

        n_tokens = aux["n_tokens"]
        n_correct = aux["n_correct"]
        aux_out = {
            "n_tokens": n_tokens,
            "n_correct": n_correct,
            "accuracy": n_correct / jnp.maximum(n_tokens, 1.0),
            "perplexity": jnp.exp(loss_global),
            "grad_norm": grad_norm,
        }
        return new_state, loss_global, aux_out

    return step_fn


# -------------------------
# Eval step (token-weighted global mean)
# -------------------------
def build_eval_step(mesh: Mesh, static):
    aux_spec = {"n_tokens": P(), "n_correct": P(), "accuracy": P(), "perplexity": P()}

    @partial(jax.jit)
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("data", None), P("data", None), P()),
        out_specs=(P(), aux_spec),
    )
    def step_fn(params, batch_ids, batch_mask_bool, key):
        key = jax.random.fold_in(key, lax.axis_index("data"))

        loss_sum_local, aux_local = _per_shard_loss_sum_and_stats(
            params, static, batch_ids, batch_mask_bool, key, inference=True
        )
        n_tok_local = aux_local["n_tokens"].astype(jnp.float32)

        loss_sum_global = lax.psum(loss_sum_local, axis_name="data")
        n_tok_global = lax.psum(n_tok_local, axis_name="data")
        loss_global = loss_sum_global / jnp.maximum(n_tok_global, 1.0)

        n_correct_global = lax.psum(aux_local["n_correct"], axis_name="data")
        aux_out = {
            "n_tokens": n_tok_global,
            "n_correct": n_correct_global,
            "accuracy": n_correct_global / jnp.maximum(n_tok_global, 1.0),
            "perplexity": jnp.exp(loss_global),
        }
        return loss_global, aux_out

    return step_fn


# -------------------------
# Text generation helper
# -------------------------

def generate_text(
    params, static, tokenizer, prompts: List[str], key: jax.Array,
    *, max_new: int = 200, temperature: float = 0.8, pad_id: int = 50256, eos_id: int = 50256,
) -> List[str]:
    if not prompts:
        return []
    pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else pad_id
    encs = [tokenizer.encode(p) for p in prompts]
    lens_np = np.array([len(e) for e in encs], dtype=np.int32)
    T0 = int(lens_np.max(initial=1))
    arr = np.full((len(prompts), T0), pad_tok, dtype=np.int32)
    for i, e in enumerate(encs):
        arr[i, : len(e)] = e

    ids  = jnp.asarray(arr, dtype=jnp.int32)
    lens = jnp.asarray(lens_np, dtype=jnp.int32)

    model = eqx.combine(params, static)
    out = sample_tokens(model, ids, lens, key=key, max_new=max_new,
                            temperature=temperature, pad_id=pad_id, eos_id=eos_id)
    out = jax.device_get(out)
    return [tokenizer.decode(out[i], skip_special_tokens=True) for i in range(len(prompts))]


# -------------------------
# Eval driver (token-weighted across batches)
# -------------------------
def evaluate(params, static, eval_fn, val_stream, tokenizer, config: Config, key: jax.Array, mesh: Mesh) -> Dict[str, float]:
    n_batches = max(1, config.eval_samples // config.batch_size)
    replicate = NamedSharding(mesh, P())

    total_loss_sum = 0.0
    total_tokens = 0.0
    total_correct = 0.0

    for _ in range(n_batches):
        batch_np = sample_batch(val_stream, config.batch_size)
        batch = shard_batch(batch_np, mesh)

        key, subkey = jax.random.split(key)
        subkey_dev = jax.device_put(subkey, replicate)
        loss, aux = eval_fn(params, batch["input_ids"], batch["attention_mask"], subkey_dev)
        loss, aux = jax.device_get((loss, aux))

        # Reconstruct sums so we average by *tokens*, not by batches.
        total_loss_sum += float(loss) * float(aux["n_tokens"])
        total_tokens   += float(aux["n_tokens"])
        total_correct  += float(aux["n_correct"])

    avg_loss = total_loss_sum / max(total_tokens, 1.0)
    avg_acc  = total_correct / max(total_tokens, 1.0)

    prompts = [
        "Once upon a time","The little robot","In a magical forest","The brave princess",
        "The king of France","My mother met a dog","Oh no!","Somebody help",
    ]
    key, subkey = jax.random.split(key)
    texts = generate_text(params, static, tokenizer, prompts, subkey,
                          max_new=config.gen_max_new, temperature=config.gen_temperature,
                          pad_id=50256, eos_id=50256)

    print("\n" + "=" * 50)
    print("Generated samples:")
    for p, t in zip(prompts, texts):
        print(f"\nPrompt: {p}\n→ {t}")
    print("=" * 50 + "\n")

    return {"loss": avg_loss, "accuracy": avg_acc, "perplexity": float(np.exp(avg_loss))}


# -------------------------
# Main
# -------------------------
def main():
    config = tyro.cli(Config)
    assert config.seq_len % 4 == 0
    host_key = jax.random.PRNGKey(config.seed)

    mesh = create_mesh()
    num_devices = mesh.devices.size
    assert config.batch_size % num_devices == 0
    print(f"Using {num_devices} device(s): {[d for d in mesh.devices.flat]}")

    tokenizer, train_stream, val_stream = setup_tokenizer_and_streams(
        dataset_name=config.dataset_name, dataset_config=config.dataset_config, seq_len=config.seq_len,
    )

    host_key, model_key = jax.random.split(host_key)
    model = Transformer(
        vocab_size=config.vocab_size, d_model=config.d_model, n_layers=config.n_layers,
        n_heads=config.n_heads, ff_mult=config.ff_mult, dropout=config.dropout,
        rope_base=config.rope_base, key=model_key,
    )
    params, static = eqx.partition(model, eqx.is_inexact_array)

    mesh_repl = NamedSharding(mesh, P())
    params = jax.device_put(params, mesh_repl)
    static = jax.device_put(static, mesh_repl)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.lr_init, peak_value=config.lr_peak,
        warmup_steps=config.warmup_steps, decay_steps=config.steps, end_value=config.lr_end,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=config.weight_decay,
        ),
    )
    opt_state = jax.device_put(optimizer.init(params), mesh_repl)

    host_key, train_key = jax.random.split(host_key)
    state = TrainState(params=params, opt_state=opt_state, key=jax.device_put(train_key, mesh_repl))

    train_fn = build_train_step(optimizer, mesh, static)
    eval_fn  = build_eval_step(mesh, static)

    run_name = f"transformer_L{config.n_layers}_D{config.d_model}_H{config.n_heads}"
    wandb.init(project=config.wandb_project, name=run_name, config=asdict(config))

    for step in range(1, config.steps + 1):
        t0 = time.perf_counter()
        batch_np = sample_batch(train_stream, config.batch_size)
        batch = shard_batch(batch_np, mesh)
        state, loss, aux = train_fn(state, batch["input_ids"], batch["attention_mask"])

        if step % config.log_every == 0:
            dt = time.perf_counter() - t0
            n_tokens = float(aux["n_tokens"])
            wandb.log(
                {
                    "train/loss": float(loss),
                    "train/accuracy": float(aux["accuracy"]),
                    "train/perplexity": float(aux["perplexity"]),
                    "train/grad_norm": float(aux["grad_norm"]),
                    "train/lr": float(schedule(step)),
                    "train/step_time": dt,
                    "train/tokens_per_sec": n_tokens / dt if dt > 0 else 0.0,
                },
                step=step,
            )
            print(
                f"Step {step:6d} | Loss: {float(loss):.4f} | Acc: {float(aux['accuracy']):.4f} | "
                f"PPL: {float(aux['perplexity']):.2f} | LR: {float(schedule(step)):.2e} | Time: {dt:.2f}s"
            )

        if step % config.eval_every == 0:
            host_key, eval_key = jax.random.split(host_key)
            eval_metrics = evaluate(state.params, static, eval_fn, val_stream, tokenizer, config, eval_key, mesh)
            wandb.log(
                {
                    "eval/loss": eval_metrics["loss"],
                    "eval/accuracy": eval_metrics["accuracy"],
                    "eval/perplexity": eval_metrics["perplexity"],
                },
                step=step,
            )
            print(
                f"[Eval] Loss: {eval_metrics['loss']:.4f} | "
                f"Acc: {eval_metrics['accuracy']:.4f} | "
                f"PPL: {eval_metrics['perplexity']:.2f}"
            )

        if step % config.save_every == 0:
            ckpt_path = Path(config.ckpt_dir) / run_name
            save_checkpoint(ckpt_path, step, state.params, static, state.opt_state, config)

    wandb.finish()


if __name__ == "__main__":
    main()
