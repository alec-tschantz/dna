from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import equinox as eqx
import jax
from jax import random
import tyro

from dna import Transformer, generate, setup_tokenizer


@dataclass
class SampleConfig:
    ckpt_dir: str = "checkpoints"
    tokenizer_name: str = "gpt2"
    run_name: str = field(default="", help="Name of the run to load checkpoint from")
    prompts: List[str] = field(
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
    gen_max_new: int = 200
    gen_temperature: float = 0.8


def main():
    cfg = tyro.cli(SampleConfig)
    run_dir = Path(cfg.ckpt_dir) / cfg.run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"checkpoint directory {run_dir} not found")

    meta = json.loads((run_dir / "meta.json").read_text())

    key = random.PRNGKey(0)
    model_template = Transformer(
        vocab_size=meta["vocab_size"],
        d_model=meta["d_model"],
        n_layers=meta["n_layers"],
        n_heads=meta["n_heads"],
        ff_mult=meta["ff_mult"],
        dropout=meta["dropout"],
        rope_base=meta["rope_base"],
        key=key,
    )

    model_loaded = eqx.tree_deserialise_leaves(run_dir / "model.eqx", model_template)
    params, static = eqx.partition(model_loaded, eqx.is_inexact_array)

    tokenizer = setup_tokenizer(config.tokenizer_name)

    key = random.PRNGKey(0)
    outputs = generate(
        params,
        static,
        tokenizer,
        cfg.prompts,
        key,
        max_new=cfg.gen_max_new,
        temperature=cfg.gen_temperature,
    )

    print("\n" + "=" * 80)
    print("Generated text samples:")
    for prompt, text in zip(cfg.prompts, outputs):
        print(f"Prompt: {prompt}\n→ {text}\n")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
