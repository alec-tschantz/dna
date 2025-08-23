from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import jax
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _process_example(
    tokenizer: PreTrainedTokenizerBase, text: str, seq_len: int
) -> Dict[str, np.ndarray]:
    enc = tokenizer(
        text,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids = enc["input_ids"][0]
    mask = enc["attention_mask"][0]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        eos_positions = np.where(ids == eos_id)[0]
        if eos_positions.size > 0:
            first_eos = int(eos_positions[0])
            mask[first_eos + 1 :] = 0
    return {"input_ids": ids.astype(np.int32), "attention_mask": mask.astype(np.int32)}


def _prepare_local_snapshot(
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    allow_patterns: Optional[list[str]] = None,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    sentinel = local_dir / ".complete"
    if sentinel.exists():
        return local_dir
    if jax.process_index() == 0:
        err = None
        for attempt in range(max_retries):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                )
                (local_dir / ".complete").write_text("OK")
                err = None
                break
            except Exception as e:
                err = e
                time.sleep(backoff_factor * (2**attempt))
        if err is not None:
            raise RuntimeError(f"Failed to download {repo_type} '{repo_id}': {err}")
    else:
        while not sentinel.exists():
            time.sleep(1.0)
    return local_dir


def setup_tokenizer(cache_dir: Optional[str] = None) -> PreTrainedTokenizerBase:
    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
    cache_path = Path(cache_dir)
    tok_dir = cache_path / "snapshots" / "tokenizers" / "gpt2"
    _prepare_local_snapshot(
        "gpt2",
        "model",
        tok_dir,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "config.json",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_data_streams(
    dataset_name: str,
    dataset_config: Optional[str],
    seq_len: int,
    tokenizer: PreTrainedTokenizerBase,
    text_key: str = "text",
    cache_dir: Optional[str] = None,
) -> Tuple[Iterator[Dict[str, np.ndarray]], Iterator[Dict[str, np.ndarray]]]:
    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
    cache_path = Path(cache_dir)
    data_dir = cache_path / "snapshots" / "datasets" / dataset_name.replace("/", "__")
    allow_patterns = ["*train*.txt", "*valid*.txt", "*validation*.txt", "*test*.txt"]
    _prepare_local_snapshot(
        dataset_name, "dataset", data_dir, allow_patterns=allow_patterns
    )
    train_ds = load_dataset(
        str(data_dir), name=(dataset_config or ""), split="train", streaming=True
    )
    try:
        val_ds = load_dataset(
            str(data_dir),
            name=(dataset_config or ""),
            split="validation",
            streaming=True,
        )
    except Exception:
        val_ds = load_dataset(
            str(data_dir), name=(dataset_config or ""), split="train", streaming=True
        )

    def stream_iter(ds):
        for sample in ds:
            text = sample.get(text_key, "")
            if isinstance(text, str) and text:
                yield _process_example(tokenizer, text, seq_len)

    train_stream = stream_iter(train_ds)
    val_stream = stream_iter(val_ds)
    return train_stream, val_stream


def sample_batch(
    stream: Iterator[Dict[str, np.ndarray]], batch_size: int
) -> Dict[str, np.ndarray]:
    batch = [next(stream) for _ in range(batch_size)]
    ids = np.stack([ex["input_ids"] for ex in batch], axis=0)
    mask = np.stack([ex["attention_mask"] for ex in batch], axis=0)
    return {"input_ids": ids, "attention_mask": mask}
