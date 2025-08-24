from __future__ import annotations
import json
import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import equinox as eqx
import jax
import optax


def save_checkpoint(
    ckpt_dir: Path,
    run_name: str,
    step: int,
    params: Any,
    static: Any,
    opt_state: Any,
    config: Any,
    metrics: Optional[Dict[str, float]] = None,
):
    run_dir = ckpt_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    params_cpu = jax.tree.map(jax.device_get, params)
    static_cpu = jax.tree.map(jax.device_get, static)
    opt_state_cpu = jax.tree.map(jax.device_get, opt_state)
    model = eqx.combine(params_cpu, static_cpu)
    eqx.tree_serialise_leaves(run_dir / "model.eqx", model)
    eqx.tree_serialise_leaves(run_dir / "opt.eqx", opt_state_cpu)
    metadata = {**asdict(config), "step": step, "timestamp": datetime.now().isoformat()}
    if metrics:
        metadata["metrics"] = metrics
    (run_dir / "meta.json").write_text(json.dumps(metadata, indent=2))


def load_checkpoint(
    ckpt_dir: Path,
    run_name: str,
    model_template: Any,
    optimizer: optax.GradientTransformation,
) -> tuple[Any, Any, int]:
    run_dir = ckpt_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"checkpoint directory {run_dir} not found")
    model_loaded = eqx.tree_deserialise_leaves(run_dir / "model.eqx", model_template)
    params_loaded, static_loaded = eqx.partition(model_loaded, eqx.is_inexact_array)
    opt_template = optimizer.init(params_loaded)
    opt_loaded = eqx.tree_deserialise_leaves(run_dir / "opt.eqx", opt_template)
    meta = json.loads((run_dir / "meta.json").read_text())
    step = meta["step"]
    return (params_loaded, static_loaded), opt_loaded, step


def print_prompt(prompt: str, output: str, width: int = 88):
    body_lines = [f"Prompt: {prompt}", "\n\n", f"Output: {output}"]
    print(_ansi_box("Generation", body_lines, width=width))


def _ansi_box(title: str, body_lines: list[str], width: int = 88) -> str:
    h, v = "─", "│"
    tl, tr, bl, br = "┌", "┐", "└", "┘"
    sep_left, sep_right = "├", "┤"

    def pad(s: str) -> str:
        s = s.replace("\t", " ")
        return s + " " * max(0, width - 4 - len(s))

    top = f"{tl}{h*(width-2)}{tr}"
    title_line = f"{v} {pad(title)} {v}"
    mid = f"{sep_left}{h*(width-2)}{sep_right}"
    bot = f"{bl}{h*(width-2)}{br}"

    lines = [top, title_line, mid]
    for ln in body_lines:
        for wrapped in textwrap.wrap(ln, width=width - 4, break_long_words=False):
            lines.append(f"{v} {pad(wrapped)} {v}")
    lines.append(bot)
    return "\n".join(lines)
