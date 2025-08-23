from __future__ import annotations
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from jaxtyping import Array, Bool, Float, Int
from jax import lax, random, config as jax_config
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dna import (
    Transformer,
    generate,
    load_checkpoint,
    sample_batch,
    save_checkpoint,
    setup_data_streams,
    setup_tokenizer,
)

jax_config.update("jax_default_matmul_precision", "tensorfloat32")


@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    ff_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10000.0
    batch_size: int = 512
    seq_len: int = 256
    steps: int = 100_000
    warmup_steps: int = 2000
    lr_init: float = 1e-6
    lr_peak: float = 3e-4
    lr_end: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    n_eval_batches: int = 16
    eval_every: int = 500
    log_every: int = 50
    save_every: int = 10000
    gen_max_new: int = 200
    gen_temperature: float = 0.8
    seed: int = 0
    ckpt_dir: str = "checkpoints"
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    resume_from: Optional[str] = None
    eval_prompts: List[str] = field(
        default_factory=lambda: [
            "Once upon a time",
            "The little robot",
            "In a magical forest",
            "The brave princess",
            "The king of France",
            "My mother met a dog",
            "Oh no!",
            "Somebody help",
        ]
    )


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    key: Array


def create_mesh(num_devices: Optional[int] = None) -> Mesh:
    devs = jax.devices()
    if num_devices is not None:
        devs = devs[:num_devices]
    mesh = Mesh(mesh_utils.create_device_mesh((len(devs),), devices=devs), ("data",))
    return mesh


def shard_batch(batch_np: Dict[str, np.ndarray], mesh: Mesh) -> Dict[str, Array]:
    data_shard = NamedSharding(mesh, P("data", None))
    ids = jax.device_put(jnp.array(batch_np["input_ids"], dtype=jnp.int32), data_shard)
    mask = jax.device_put(
        jnp.array(batch_np["attention_mask"], dtype=jnp.bool_), data_shard
    )
    return {"input_ids": ids, "attention_mask": mask}


def _loss_on_shard(
    params: Any,
    static: Any,
    ids: Int[Array, "B T"],
    mask: Bool[Array, "B T"],
    key: Array,
    *,
    inference: bool,
) -> tuple[Float[Array, ""], Dict[str, Array]]:
    model = eqx.combine(params, static)
    logits: Float[Array, "B T V"] = model(ids, mask, key=key, inference=inference)
    logits_next = logits[:, :-1]
    target_ids = ids[:, 1:]
    mask_next = mask[:, 1:].astype(jnp.float32)
    loss_per_token: Float[Array, "B T-1"] = (
        optax.softmax_cross_entropy_with_integer_labels(logits_next, target_ids)
    )
    token_loss_sum = (loss_per_token * mask_next).sum()
    predictions = jnp.argmax(logits_next, axis=-1)
    n_tokens = mask_next.sum()
    n_correct = ((predictions == target_ids) * mask_next).sum()
    return token_loss_sum, {"n_tokens": n_tokens, "n_correct": n_correct}


def build_train_step(optim: optax.GradientTransformation, mesh: Mesh, static: Any):
    def loss_fn(params, ids, mask, key):
        return _loss_on_shard(params, static, ids, mask, key, inference=False)

    loss_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    aux_pspec = {
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
        out_specs=(P(), P(), aux_pspec),
    )
    def train_step(state: TrainState, ids: Int[Array, "B T"], mask: Bool[Array, "B T"]):
        key, new_key = random.split(state.key)
        shard_key = random.fold_in(key, lax.axis_index("data"))
        (loss_sum_shard, aux_shard), grads_shard = loss_and_grad(
            state.params, ids, mask, shard_key
        )
        grads_sum = jax.tree.map(lambda g: lax.psum(g, "data"), grads_shard)
        loss_sum = lax.psum(loss_sum_shard, "data")
        n_tokens = lax.psum(aux_shard["n_tokens"], "data")
        n_correct = lax.psum(aux_shard["n_correct"], "data")
        loss_mean = loss_sum / jnp.maximum(n_tokens, 1.0)
        accuracy = n_correct / jnp.maximum(n_tokens, 1.0)
        grads_mean = jax.tree.map(lambda g: g / jnp.maximum(n_tokens, 1.0), grads_sum)
        updates, new_opt_state = optim.update(grads_mean, state.opt_state, state.params)
        new_params = eqx.apply_updates(state.params, updates)
        grad_norm = optax.global_norm(grads_mean)
        new_state = TrainState(params=new_params, opt_state=new_opt_state, key=new_key)
        aux_out = {
            "n_tokens": n_tokens,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "perplexity": jnp.exp(loss_mean),
            "grad_norm": grad_norm,
        }
        return new_state, loss_mean, aux_out

    return train_step


def build_eval_step(mesh: Mesh, static: Any):
    aux_pspec = {"n_tokens": P(), "n_correct": P(), "accuracy": P(), "perplexity": P()}

    @partial(jax.jit)
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("data", None), P("data", None), P()),
        out_specs=(P(), aux_pspec),
    )
    def eval_step(
        params: Any, ids: Int[Array, "B T"], mask: Bool[Array, "B T"], key: Array
    ):
        shard_key = random.fold_in(key, lax.axis_index("data"))
        loss_sum_shard, aux_shard = _loss_on_shard(
            params, static, ids, mask, shard_key, inference=True
        )
        n_tokens = lax.psum(aux_shard["n_tokens"], "data")
        n_correct = lax.psum(aux_shard["n_correct"], "data")
        loss_sum = lax.psum(loss_sum_shard, "data")
        loss_mean = loss_sum / jnp.maximum(n_tokens, 1.0)
        aux_out = {
            "n_tokens": n_tokens,
            "n_correct": n_correct,
            "accuracy": n_correct / jnp.maximum(n_tokens, 1.0),
            "perplexity": jnp.exp(loss_mean),
        }
        return loss_mean, aux_out

    return eval_step


def evaluate(
    params: Any,
    static: Any,
    eval_step,
    val_stream: Iterator[Dict[str, np.ndarray]],
    tokenizer: Any,
    config: Config,
    key: Array,
    mesh: Mesh,
) -> Dict[str, float]:
    batches = max(1, int(config.n_eval_batches))
    rep = NamedSharding(mesh, P())
    total_loss = 0.0
    total_tokens = 0.0
    total_correct = 0.0
    for _ in range(batches):
        batch_np = sample_batch(val_stream, config.batch_size)
        batch = shard_batch(batch_np, mesh)
        key, subkey = random.split(key)
        subkey_dev = jax.device_put(subkey, rep)
        loss, aux = eval_step(
            params, batch["input_ids"], batch["attention_mask"], subkey_dev
        )
        loss, aux = jax.device_get((loss, aux))
        total_loss += float(loss) * float(aux["n_tokens"])
        total_tokens += float(aux["n_tokens"])
        total_correct += float(aux["n_correct"])
    avg_loss = total_loss / max(total_tokens, 1.0)
    avg_acc = total_correct / max(total_tokens, 1.0)
    key, subkey = random.split(key)
    generated = generate(
        params,
        static,
        tokenizer,
        config.eval_prompts,
        subkey,
        max_new=config.gen_max_new,
        temperature=config.gen_temperature,
    )
    print("\n" + "=" * 80)
    print("Generated text samples:")
    for prompt, text in zip(config.eval_prompts, generated):
        print(f"Prompt: {prompt}\n→ {text}\n")
    print("=" * 80 + "\n")
    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "perplexity": float(np.exp(avg_loss)),
    }


def main():
    config = tyro.cli(Config)
    assert config.seq_len % 4 == 0
    host_key = random.PRNGKey(config.seed)
    mesh = create_mesh()
    num_devices = mesh.devices.size
    assert config.batch_size % num_devices == 0

    tokenizer = setup_tokenizer()
    train_stream, val_stream = setup_data_streams(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        seq_len=config.seq_len,
        tokenizer=tokenizer,
    )

    host_key, model_key = random.split(host_key)
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
    eqx.tree_pprint(model)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    n_params = sum(
        int(x.size) for x in jax.tree.leaves(params) if isinstance(x, jnp.ndarray)
    )

    wd_mask = jax.tree.map(
        lambda x: (
            isinstance(x, jnp.ndarray)
            and jnp.issubdtype(x.dtype, jnp.floating)
            and x.ndim >= 2
        ),
        params,
    )
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.lr_init,
        peak_value=config.lr_peak,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
        end_value=config.lr_end,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(lr_schedule, b1=0.9, b2=0.95, eps=1e-8),
        (
            optax.add_decayed_weights(config.weight_decay, mask=wd_mask)
            if config.weight_decay > 0
            else optax.identity()
        ),
    )

    mesh_sharding = NamedSharding(mesh, P())
    start_step = 1

    if config.run_name is None:
        ds_name = config.dataset_name.split("/")[-1]
        run_name = f"transformer_{config.n_layers}l_{config.d_model}d_{config.n_heads}h_seq{config.seq_len}_bs{config.batch_size}_{ds_name}"
    else:
        run_name = config.run_name

    if config.resume_from:
        (params, static), opt_state, loaded_step = load_checkpoint(
            Path(config.ckpt_dir), config.resume_from, model, optimizer
        )
        start_step = int(loaded_step) + 1
    else:
        opt_state = optimizer.init(params)

    params = jax.device_put(params, mesh_sharding)
    static = jax.device_put(static, mesh_sharding)
    opt_state = jax.device_put(opt_state, mesh_sharding)
    host_key, train_key = random.split(host_key)
    train_state = TrainState(
        params=params, opt_state=opt_state, key=jax.device_put(train_key, mesh_sharding)
    )

    train_step = build_train_step(optimizer, mesh, static)
    eval_step = build_eval_step(mesh, static)

    run = None
    if config.wandb_project:
        run = wandb.init(
            project=config.wandb_project, name=run_name, config=asdict(config)
        )
        wandb.log({"params/num_parameters": n_params}, step=0)

    for step in range(start_step, config.steps + 1):
        t_start = time.perf_counter()
        batch_np = sample_batch(train_stream, config.batch_size)
        batch = shard_batch(batch_np, mesh)
        train_state, loss, aux = train_step(
            train_state, batch["input_ids"], batch["attention_mask"]
        )

        if step % config.log_every == 0:
            elapsed = time.perf_counter() - t_start
            tokens = float(aux["n_tokens"])
            lr = float(lr_schedule(step))
            log_data = {
                "train/loss": float(loss),
                "train/accuracy": float(aux["accuracy"]),
                "train/perplexity": float(aux["perplexity"]),
                "train/grad_norm": float(aux["grad_norm"]),
                "train/lr": lr,
                "train/step_time_sec": elapsed,
                "train/tokens_per_sec": tokens / elapsed if elapsed > 0 else 0.0,
                "params/num_parameters": n_params,
            }
            if run:
                wandb.log(log_data, step=step)
            print(
                f"Step {step:>6d} | loss {float(loss):.4f} | acc {float(aux['accuracy']):.4f} | ppl {float(aux['perplexity']):.2f} | lr {lr:.3e} | grad_norm {float(aux['grad_norm']):.3e} | time {elapsed:.2f}s | tokens/sec {log_data['train/tokens_per_sec']:.2f}"
            )

        if step % config.eval_every == 0:
            host_key, eval_key = random.split(host_key)
            eval_metrics = evaluate(
                train_state.params,
                static,
                eval_step,
                val_stream,
                tokenizer,
                config,
                eval_key,
                mesh,
            )
            if run:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)
            print(
                f"[Eval @ step {step}] loss {eval_metrics['loss']:.4f} | acc {eval_metrics['accuracy']:.4f} | ppl {eval_metrics['perplexity']:.2f}"
            )

        if step % config.save_every == 0:
            ckpt_dir = Path(config.ckpt_dir)
            save_checkpoint(
                ckpt_dir,
                run_name,
                step,
                train_state.params,
                static,
                train_state.opt_state,
                config,
            )

    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
