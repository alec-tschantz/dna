# train.py
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from jaxtyping import Array, Bool, Float, Int
from jax import config as jax_config, lax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dataloader import sample_batch, setup_tokenizer_and_streams
from model import Transformer

jax_config.update("jax_default_matmul_precision", "tensorfloat32")


# --------------------------------------------------------------------------------------
# Config and state
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 50_257
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0
    batch_size: int = 256
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
    key: Array


# --------------------------------------------------------------------------------------
# Sampling (moved in from sample.py)
# --------------------------------------------------------------------------------------
@eqx.filter_jit
def sample_tokens(
    model: Any,
    prompt_ids: Int[Array, "B T0"],
    prompt_lens: Int[Array, "B"],
    *,
    key: Array,
    max_new: int = 200,
    temperature: float = 0.8,
    pad_id: int = 50256,
    eos_id: int = 50256,
) -> Int[Array, "B T_total"]:
    B, T0 = prompt_ids.shape
    total_len = T0 + max_new
    tokens: Int[Array, "B T_total"] = jnp.concatenate(
        [prompt_ids, jnp.full((B, max_new), pad_id, dtype=jnp.int32)], axis=1
    )
    current_pos: Int[Array, "B"] = prompt_lens.astype(jnp.int32)
    is_done: Bool[Array, "B"] = jnp.zeros_like(prompt_lens, dtype=bool)
    positions: Int[Array, "1 T_total"] = jnp.arange(total_len, dtype=jnp.int32)[None, :]

    def step(carry, _):
        tkns, pos, done, k = carry
        k, sub = jax.random.split(k)
        attn_mask: Bool[Array, "B T_total"] = positions < pos[:, None]
        keys = jax.random.split(sub, B)
        logits: Float[Array, "B T_total V"] = jax.vmap(
            lambda ts, m, kk: model(ts, m, key=kk, inference=True)
        )(tkns, attn_mask, keys)
        last_idx: Int[Array, "B"] = jnp.maximum(pos, 1) - 1
        last_logits: Float[Array, "B V"] = logits[jnp.arange(B), last_idx]

        def _sample(ops):
            lg, ks = ops
            scaled = lg / jnp.maximum(temperature, 1e-6)
            return jax.vmap(lambda _lg, _k: jax.random.categorical(_k, _lg))(scaled, ks)

        def _greedy(ops):
            lg, _ks = ops
            return jnp.argmax(lg, axis=-1)

        subs = jax.random.split(k, B + 1)
        k = subs[0]
        sample_keys = subs[1:]
        nxt: Int[Array, "B"] = lax.cond(
            temperature > 0.0, _sample, _greedy, operand=(last_logits, sample_keys)
        )
        nxt = jnp.where(done, pad_id, nxt)
        tkns = tkns.at[jnp.arange(B), pos].set(nxt)
        new_done = done | (nxt == eos_id)
        new_pos = jnp.where(new_done, pos, pos + 1)
        return (tkns, new_pos, new_done, k), None

    (tokens, _, _, _), _ = lax.scan(step, (tokens, current_pos, is_done, key), None, length=max_new)
    return tokens


# --------------------------------------------------------------------------------------
# Mesh / sharding
# --------------------------------------------------------------------------------------
def create_mesh(n_devices: Optional[int] = None) -> Mesh:
    devs = jax.devices()
    if n_devices is not None:
        devs = devs[:n_devices]
    device_array = mesh_utils.create_device_mesh((len(devs),), devices=devs)
    return Mesh(device_array, ("data",))


def shard_batch(batch: Dict[str, np.ndarray], mesh: Mesh) -> Dict[str, Array]:
    data_shard = NamedSharding(mesh, P("data", None))
    ids = jax.device_put(jnp.asarray(batch["input_ids"], dtype=jnp.int32), data_shard)
    mask = jax.device_put(jnp.asarray(batch["attention_mask"], dtype=jnp.bool_), data_shard)
    return {"input_ids": ids, "attention_mask": mask}


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    step: int,
    params: Any,
    static: Any,
    opt_state: Any,
    config: Config,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    params_host = jax.tree_util.tree_map(jax.device_get, params)
    static_host = jax.tree_util.tree_map(jax.device_get, static)
    opt_state_host = jax.tree_util.tree_map(jax.device_get, opt_state)
    model = eqx.combine(params_host, static_host)
    eqx.tree_serialise_leaves(path / f"model_{step}.eqx", model)
    eqx.tree_serialise_leaves(path / f"opt_{step}.eqx", opt_state_host)
    meta = {**asdict(config), "step": step, "timestamp": datetime.now().isoformat()}
    if metrics:
        meta["metrics"] = metrics
    (path / f"meta_{step}.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------------------
# Loss and per-shard forward
# --------------------------------------------------------------------------------------
def _per_shard_loss_sum_and_stats(
    params: Any,
    static: Any,
    ids: Int[Array, "B T"],
    mask_bool: Bool[Array, "B T"],
    key: Array,
    *,
    inference: bool,
) -> Tuple[Float[Array, ""], Dict[str, Array]]:
    model = eqx.combine(params, static)
    B = ids.shape[0]
    row_keys: Array = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(B, dtype=jnp.uint32))

    def f(ids_row: Int[Array, "T"], mask_row: Bool[Array, "T"], k: Array) -> Float[Array, "T V"]:
        return model(ids_row, mask_row, key=k, inference=inference)

    logits: Float[Array, "B T V"] = jax.vmap(f)(ids, mask_bool, row_keys)
    logits_shift = logits[:, :-1]
    labels_shift = ids[:, 1:]
    mask_shift: Bool[Array, "B T-1"] = mask_bool[:, 1:]
    mask_f: Float[Array, "B T-1"] = mask_shift.astype(jnp.float32)

    loss_tok: Float[Array, "B T-1"] = optax.softmax_cross_entropy_with_integer_labels(
        logits_shift, labels_shift
    )
    loss_sum: Float[Array, ""] = (loss_tok * mask_f).sum()

    preds = jnp.argmax(logits_shift, axis=-1)
    n_tokens: Float[Array, ""] = mask_f.sum()
    n_correct: Float[Array, ""] = ((preds == labels_shift) & mask_shift).sum()

    return loss_sum, {"n_tokens": n_tokens, "n_correct": n_correct}


# --------------------------------------------------------------------------------------
# Train / eval steps
# --------------------------------------------------------------------------------------
def build_train_step(optimizer: optax.GradientTransformation, mesh: Mesh, static: Any):
    def loss_fn(p: Any, ids: Int[Array, "B T"], m: Bool[Array, "B T"], k: Array):
        return _per_shard_loss_sum_and_stats(p, static, ids, m, k, inference=False)

    loss_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    aux_spec = {"n_tokens": P(), "n_correct": P(), "accuracy": P(), "perplexity": P(), "grad_norm": P()}

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("data", None), P("data", None)),
        out_specs=(P(), P(), aux_spec),
    )
    def step_fn(state: TrainState, batch_ids: Int[Array, "B T"], batch_mask: Bool[Array, "B T"]):
        key, new_key = jax.random.split(state.key)
        key = jax.random.fold_in(key, lax.axis_index("data"))

        (loss_sum_local, aux_local), grads_local = loss_and_grad(state.params, batch_ids, batch_mask, key)

        n_tok_local = aux_local["n_tokens"].astype(jnp.float32)
        n_tok_global = lax.psum(n_tok_local, "data")
        grads_sum = jax.tree_util.tree_map(lambda g: lax.psum(g, "data"), grads_local)
        loss_sum_global = lax.psum(loss_sum_local, "data")

        denom = jnp.maximum(n_tok_global, 1.0)
        grads_global_mean = jax.tree_util.tree_map(lambda g: g / denom, grads_sum)
        loss_global_mean = loss_sum_global / denom
        grad_norm = optax.global_norm(grads_global_mean)

        updates, new_opt_state = optimizer.update(grads_global_mean, state.opt_state, state.params)
        new_params = eqx.apply_updates(state.params, updates)
        new_state = TrainState(params=new_params, opt_state=new_opt_state, key=new_key)

        n_correct_global = lax.psum(aux_local["n_correct"], "data")
        aux_out = {
            "n_tokens": n_tok_global,
            "n_correct": n_correct_global,
            "accuracy": n_correct_global / jnp.maximum(n_tok_global, 1.0),
            "perplexity": jnp.exp(loss_global_mean),
            "grad_norm": grad_norm,
        }
        return new_state, loss_global_mean, aux_out

    return step_fn


def build_eval_step(mesh: Mesh, static: Any):
    aux_spec = {"n_tokens": P(), "n_correct": P(), "accuracy": P(), "perplexity": P()}

    @partial(jax.jit)
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("data", None), P("data", None), P()),
        out_specs=(P(), aux_spec),
    )
    def step_fn(params: Any, batch_ids: Int[Array, "B T"], batch_mask: Bool[Array, "B T"], key: Array):
        key = jax.random.fold_in(key, lax.axis_index("data"))

        loss_sum_local, aux_local = _per_shard_loss_sum_and_stats(
            params, static, batch_ids, batch_mask, key, inference=True
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


# --------------------------------------------------------------------------------------
# Text generation helper
# --------------------------------------------------------------------------------------
def generate_text(
    params: Any,
    static: Any,
    tokenizer: Any,
    prompts: List[str],
    key: Array,
    *,
    max_new: int,
    temperature: float,
    pad_id: int = 50256,
    eos_id: int = 50256,
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

    ids: Int[Array, "B T0"] = jnp.asarray(arr, dtype=jnp.int32)
    lens: Int[Array, "B"] = jnp.asarray(lens_np, dtype=jnp.int32)

    model = eqx.combine(params, static)
    out: Int[Array, "B T_total"] = sample_tokens(
        model,
        ids,
        lens,
        key=key,
        max_new=max_new,
        temperature=temperature,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    out_host = jax.device_get(out)
    return [tokenizer.decode(out_host[i], skip_special_tokens=True) for i in range(len(prompts))]


# --------------------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------------------
def evaluate(
    params: Any,
    static: Any,
    eval_fn,
    val_stream: Iterator[Dict[str, np.ndarray]],
    tokenizer: Any,
    config: Config,
    key: Array,
    mesh: Mesh,
) -> Dict[str, float]:
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

        total_loss_sum += float(loss) * float(aux["n_tokens"])
        total_tokens += float(aux["n_tokens"])
        total_correct += float(aux["n_correct"])

    avg_loss = total_loss_sum / max(total_tokens, 1.0)
    avg_acc = total_correct / max(total_tokens, 1.0)

    prompts = [
        "Once upon a time",
        "The little robot",
        "In a magical forest",
        "The brave princess",
        "The king of France",
        "My mother met a dog",
        "Oh no!",
        "Somebody help",
    ]
    key, subkey = jax.random.split(key)
    texts = generate_text(
        params,
        static,
        tokenizer,
        prompts,
        subkey,
        max_new=config.gen_max_new,
        temperature=config.gen_temperature,
        pad_id=50256,
        eos_id=50256,
    )

    print("\n" + "=" * 50)
    print("Generated samples:")
    for p, t in zip(prompts, texts):
        print(f"\nPrompt: {p}\n→ {t}")
    print("=" * 50 + "\n")

    return {"loss": avg_loss, "accuracy": avg_acc, "perplexity": float(np.exp(avg_loss))}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    config = tyro.cli(Config)
    assert config.seq_len % 4 == 0
    host_key = jax.random.PRNGKey(config.seed)

    mesh = create_mesh()
    num_devices = mesh.devices.size
    assert config.batch_size % num_devices == 0
    print(f"Using {num_devices} device(s): {[d for d in mesh.devices.flat]}")

    tokenizer, train_stream, val_stream = setup_tokenizer_and_streams(
        dataset_name=config.dataset_name, dataset_config=config.dataset_config, seq_len=config.seq_len
    )

    host_key, model_key = jax.random.split(host_key)
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
        rope_base=config.rope_base,
        key=model_key,
    )
    params, static = eqx.partition(model, eqx.is_inexact_array)

    mesh_repl = NamedSharding(mesh, P())
    params = jax.device_put(params, mesh_repl)
    static = jax.device_put(static, mesh_repl)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.lr_init, peak_value=config.lr_peak, warmup_steps=config.warmup_steps,
        decay_steps=config.steps, end_value=config.lr_end
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=config.weight_decay),
    )
    opt_state = jax.device_put(optimizer.init(params), mesh_repl)

    host_key, train_key = jax.random.split(host_key)
    state = TrainState(params=params, opt_state=opt_state, key=jax.device_put(train_key, mesh_repl))

    train_fn = build_train_step(optimizer, mesh, static)
    eval_fn = build_eval_step(mesh, static)

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
                f"PPL: {float(aux['perplexity']):.2f} | LR: {float(schedule(step)):.2e} | "
                f"Grad‖‖: {float(aux['grad_norm']):.3e} | Time: {dt:.2f}s"
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
