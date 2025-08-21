# dataset.py
from __future__ import annotations
from typing import Dict, Iterator, Optional, Tuple
import os, time
from pathlib import Path

import numpy as np
import jax
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# --------------------------- tokenization utils ---------------------------


def _process_example(
    tok: PreTrainedTokenizerBase, text: str, seq_len: int
) -> Dict[str, np.ndarray]:
    enc = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="np",
    )
    ids = enc["input_ids"][0]  # (T,)
    msk = enc["attention_mask"][0]  # (T,)

    eos_id = tok.eos_token_id
    if eos_id is not None:
        pos = np.where(ids == eos_id)[0]
        if pos.size > 0:
            msk[int(pos[0]) + 1 :] = 0

    return {"input_ids": ids.astype(np.int32), "attention_mask": msk.astype(np.int32)}


# --------------------------- local snapshot helpers ---------------------------


def _prepare_local_snapshot(
    repo_id: str,
    repo_type: str,  # "model" or "dataset"
    local_dir: Path,
    allow_patterns: Optional[list[str]] = None,
    max_retries: int = 6,
    backoff_s: float = 2.0,
) -> Path:
    local_dir = Path(local_dir)
    sentinel = local_dir / ".ready"

    if sentinel.exists():
        return local_dir

    rank = jax.process_index()
    if rank == 0:
        local_dir.mkdir(parents=True, exist_ok=True)
        last_err = None
        for i in range(max_retries):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                )
                sentinel.write_text("ok")
                break
            except Exception as e:
                last_err = e
                sleep_for = backoff_s * (2**i)
                time.sleep(sleep_for)
        else:
            # failed after retries
            raise RuntimeError(
                f"Failed to snapshot_download {repo_type} '{repo_id}' after {max_retries} retries"
            ) from last_err
    else:
        # non-zero processes: wait until rank 0 is done
        while not sentinel.exists():
            time.sleep(1.0)

    return local_dir


# --------------------------- public API ---------------------------


def setup_tokenizer_and_streams(
    *,
    dataset_name: str = "roneneldan/TinyStories",
    dataset_config: Optional[str] = None,
    seq_len: int = 256,
    text_key: str = "text",
    cache_root: Optional[str] = None,
) -> Tuple[
    PreTrainedTokenizerBase,
    Iterator[Dict[str, np.ndarray]],
    Iterator[Dict[str, np.ndarray]],
]:
    if cache_root is None:
        cache_root = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
    cache_root = Path(cache_root)

    tok_local = _prepare_local_snapshot(
        repo_id="gpt2",
        repo_type="model",
        local_dir=cache_root / "snapshots" / "tokenizers" / "gpt2",
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "config.json",
        ],
    )
    tok = AutoTokenizer.from_pretrained(str(tok_local), local_files_only=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    ds_local = _prepare_local_snapshot(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=cache_root
        / "snapshots"
        / "datasets"
        / dataset_name.replace("/", "__"),
        allow_patterns=["*train.txt", "*valid.txt"],
    )

    train_ds = load_dataset(str(ds_local), split="train", streaming=True)
    try:
        val_ds = load_dataset(str(ds_local), split="validation", streaming=True)
    except Exception:
        val_ds = load_dataset(str(ds_local), split="train", streaming=True)

    def to_stream(hfds):
        for ex in hfds:
            txt = ex.get(text_key, "")
            if isinstance(txt, str) and txt:
                yield _process_example(tok, txt, seq_len)

    train_stream = to_stream(train_ds)
    val_stream = to_stream(val_ds)

    return tok, train_stream, val_stream


def sample_batch(
    stream: Iterator[Dict[str, np.ndarray]], batch_size: int
) -> Dict[str, np.ndarray]:
    ids, msk = [], []
    for _ in range(batch_size):
        ex = next(stream)
        ids.append(ex["input_ids"])
        msk.append(ex["attention_mask"])
    return {"input_ids": np.stack(ids, 0), "attention_mask": np.stack(msk, 0)}
