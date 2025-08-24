from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import equinox as eqx
import jax
from jax import random
import tyro

from dna import Transformer, generate, setup_tokenizer, print_prompt


@dataclass
class SampleConfig:
    ckpt_dir: str = "checkpoints"
    run_name: str = field(default="", help="Name of the run to load checkpoint from")
    prompts: List[str] = field(
        default_factory=lambda: [
            "Explain how TLS certificates enable both authentication and encryption, in plain terms.",
            "Why are stars twinkly but planets not? Answer for a curious 10-year-old.",
            "You have 3 jugs of 3L, 5L, and 8L. How can you measure exactly 4L? Show steps.",
            "A farmer has goats and chickens. There are 22 heads and 56 legs. How many of each?",
            "Write a Python function that returns the top-k most frequent words in a list of strings.",
            "Describe and compare BFS and DFS. When does each shine?",
            "Derive the gradient of softmax cross-entropy loss with respect to the logits.",
            "What is the difference between variance and standard deviation? Give an intuitive example.",
            "In a small coastal village, a young cartographer discovers old maps that don't match the shoreline.",
            "Summarize the main arguments for and against universal basic income in ~5 bullet points.",
        ]
    )
    gen_max_new: int = 256
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

    tokenizer = setup_tokenizer()

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

    for prompt, text in zip(config.prompts, generated):
        print_prompt(prompt, text)


if __name__ == "__main__":
    main()
