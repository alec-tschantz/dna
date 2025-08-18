# logs/plots/temp_sweep.py
from __future__ import annotations
from typing import Any, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import wandb


def _heatmap(mat: np.ndarray, xlabels: List[str], ylabels: List[str], title: str):
    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=150)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=0)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_xlabel("select_temp")
    ax.set_ylabel("router_temp")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel("eval loss", rotation=90, va="center")
    fig.tight_layout()
    return fig


def log_temperature_sweep_grid(
    *,
    model,
    eval_step_fn,
    val_stream,
    key,
    tok,
    cfg,
    base_router_temp: float,
    base_select_temp: float,
    sample_batch_fn,
    step: int,
    r_scales: Tuple[float, ...] = (0.7, 1.0, 1.3),
    s_scales: Tuple[float, ...] = (0.7, 1.0, 1.3),
    batch_tokens: int = 8_192,  # small, fixed batch
):
    """Evaluate a tiny grid over router/select temps on a fixed mini-batch."""
    # Fix a small batch so grid is comparable & quick
    mini_batch = sample_batch_fn(val_stream, max(1, batch_tokens // cfg.seq_len))

    losses = np.zeros((len(r_scales), len(s_scales)), dtype=np.float32)

    for i, rs in enumerate(r_scales):
        for j, ss in enumerate(s_scales):
            eval_kwargs = {
                "gumbel_tau": jnp.array([0.0], dtype=jnp.float32),
                "router_temp": jnp.array([float(base_router_temp * rs)], dtype=jnp.float32),
                "select_temp": jnp.array([float(base_select_temp * ss)], dtype=jnp.float32),
            }
            key, sub = jax.random.split(key)
            val_loss, val_acc, _ = eval_step_fn(model, mini_batch, key=sub, model_kwargs=eval_kwargs)
            losses[i, j] = float(val_loss)

    # Visual
    xlabels = [f"{base_select_temp*ss:.2f}" for ss in s_scales]
    ylabels = [f"{base_router_temp*rs:.2f}" for rs in r_scales]
    fig = _heatmap(losses, xlabels, ylabels, f"Temp sweep (loss) • step {step}")

    best_idx = np.unravel_index(np.argmin(losses), losses.shape)
    best_router = float(base_router_temp * r_scales[best_idx[0]])
    best_select = float(base_select_temp * s_scales[best_idx[1]])
    best_loss = float(losses[best_idx])

    # Log everything
    table = wandb.Table(columns=["router_temp", "select_temp", "loss"])
    for i, rs in enumerate(r_scales):
        for j, ss in enumerate(s_scales):
            table.add_data(float(base_router_temp * rs), float(base_select_temp * ss), float(losses[i, j]))

    wandb.log(
        {
            "routing/temp_sweep_heatmap": wandb.Image(fig),
            # "router/eval/temp_sweep_table": table,
            # "routingtemp_sweep/best_loss": best_loss,
            # "routing/temp_sweep/best_router_temp": best_router,
            # "routing/temp_sweep/best_select_temp": best_select,
        }, step=step, commit=False
    )
    plt.close(fig)
    return key
