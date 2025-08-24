from __future__ import annotations
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import jax
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def _find_text_column(ds: IterableDataset) -> str:
    if getattr(ds, "features", None):
        preferred = [
            "text",
            "content",
            "document",
            "passage",
            "article",
            "story",
            "body",
        ]
        string_cols = [
            k for k, v in ds.features.items() if getattr(v, "dtype", None) == "string"
        ]
        for p in preferred:
            if p in string_cols:
                return p
        if string_cols:
            return string_cols[0]

    sample = next(iter(ds))
    preferred = ["text", "content", "document", "passage", "article", "story", "body"]
    for col in preferred:
        if col in sample and isinstance(sample[col], str):
            return col
    for k, v in sample.items():
        if isinstance(v, str):
            return k

    raise ValueError(
        f"Could not find a text column. Sample keys: {list(sample.keys())}"
    )


class PackedSequenceBuffer:
    """Accumulates token ids and emits fixed-length sequences without padding."""

    def __init__(self, seq_len: int, pad_id: int):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.token_buffer: List[int] = []
        self.mask_buffer: List[int] = []

    def add_tokens(self, tokens: List[int]) -> List[Dict[str, np.ndarray]]:
        self.token_buffer.extend(tokens)
        self.mask_buffer.extend([1] * len(tokens))
        out: List[Dict[str, np.ndarray]] = []
        while len(self.token_buffer) >= self.seq_len:
            ids = np.asarray(self.token_buffer[: self.seq_len], dtype=np.int32)
            attn = np.asarray(self.mask_buffer[: self.seq_len], dtype=np.int32)
            del self.token_buffer[: self.seq_len]
            del self.mask_buffer[: self.seq_len]
            out.append({"input_ids": ids, "attention_mask": attn})
        return out

    def flush(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.token_buffer:
            return None
        remaining = len(self.token_buffer)
        if remaining < self.seq_len:
            pad = self.seq_len - remaining
            self.token_buffer.extend([self.pad_id] * pad)
            self.mask_buffer.extend([0] * pad)
        ids = np.asarray(self.token_buffer[: self.seq_len], dtype=np.int32)
        attn = np.asarray(self.mask_buffer[: self.seq_len], dtype=np.int32)
        self.token_buffer.clear()
        self.mask_buffer.clear()
        return {"input_ids": ids, "attention_mask": attn}


class DataStream:
    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        text_column: str,
        pack_sequences: bool = False,
        min_text_length: int = 10,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.text_column = text_column
        self.pack_sequences = pack_sequences
        self.min_text_length = int(min_text_length)

        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id or self.pad_id

        if self.pad_id is None or self.eos_id is None:
            raise ValueError(
                "Tokenizer must define either pad_token_id or eos_token_id."
            )

        if pack_sequences:
            self.pack_buffer = PackedSequenceBuffer(self.seq_len, self.pad_id)
            self.sequence_queue: deque[Dict[str, np.ndarray]] = deque()

        self._reset_iter()

    def _reset_iter(self):
        self.data_iter = iter(self.dataset)

    def _tokenize_fixed(self, text: str) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        ids = enc["input_ids"][0].astype(np.int32)
        mask = enc["attention_mask"][0].astype(np.int32)

        if self.eos_id is not None:
            eos_pos = np.where(ids == self.eos_id)[0]
            if eos_pos.size > 0:
                first = int(eos_pos[0])
                if first + 1 < len(mask):
                    mask[first + 1 :] = 0
        return {"input_ids": ids, "attention_mask": mask}

    def _next_text(self) -> Optional[str]:
        for _ in range(1000):
            try:
                ex = next(self.data_iter)
                t = ex.get(self.text_column, "")
                if isinstance(t, str) and len(t) >= self.min_text_length:
                    return t
            except StopIteration:
                self._reset_iter()
            except Exception as e:
                print(f"[DataStream] read error: {e}")
                time.sleep(0.1)
        return None

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        if self.pack_sequences:
            if self.sequence_queue:
                return self.sequence_queue.popleft()

            while not self.sequence_queue:
                text = self._next_text()
                if text is None:
                    final_seq = self.pack_buffer.flush()
                    if final_seq is not None:
                        return final_seq
                    raise StopIteration("No more data available.")
                toks = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )
                if toks and (self.eos_id is not None) and (toks[-1] != self.eos_id):
                    toks.append(self.eos_id)
                self.sequence_queue.extend(self.pack_buffer.add_tokens(toks))

            return self.sequence_queue.popleft()

        text = self._next_text()
        if text is None:
            raise StopIteration("No more data available.")
        return self._tokenize_fixed(text)


def setup_tokenizer(
    tokenizer_name: str = "gpt2",
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )

    cache_path = Path(cache_dir)
    tok_dir = cache_path / "tokenizers" / tokenizer_name.replace("/", "__")
    tok_dir.mkdir(parents=True, exist_ok=True)
    sentinel = tok_dir / ".tokenizer_ready"

    if jax.process_index() == 0:
        if not sentinel.exists():
            tok = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir=str(cache_path),
                use_fast=True,
                trust_remote_code=False,
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            tok.padding_side = "right"
            tok.model_max_length = int(1e9)
            tok.clean_up_tokenization_spaces = False

            tok.save_pretrained(str(tok_dir))
            sentinel.write_text("OK")
    else:
        t0 = time.time()
        while not sentinel.exists():
            if time.time() - t0 > 600:
                raise TimeoutError(f"Timeout waiting for tokenizer: {tokenizer_name}")
            time.sleep(0.25)

    tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        str(tok_dir), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = int(1e9)
    tok.clean_up_tokenization_spaces = False
    return tok


def _load_streaming_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: Optional[str],
    seed: int,
    shuffle_buffer: int,
) -> IterableDataset:
    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )
    if jax.process_count() > 1:
        ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())

    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


def setup_data_streams(
    dataset_name: str,
    dataset_config: Optional[str],
    seq_len: int,
    tokenizer: PreTrainedTokenizerBase,
    pack_sequences: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[Iterator[Dict[str, np.ndarray]], Iterator[Dict[str, np.ndarray]]]:
    seed = int(os.environ.get("SEED", "0"))
    shuffle_buffer = int(os.environ.get("SHUFFLE_BUFFER", "50000"))

    train_ds = _load_streaming_split(
        dataset_name, dataset_config, "train", cache_dir, seed, shuffle_buffer
    )
    try:
        val_ds = _load_streaming_split(
            dataset_name,
            dataset_config,
            "validation",
            cache_dir,
            seed + 1,
            shuffle_buffer,
        )
    except Exception:
        try:
            val_ds = _load_streaming_split(
                dataset_name,
                dataset_config,
                "test",
                cache_dir,
                seed + 1,
                shuffle_buffer,
            )
        except Exception:
            print(
                f"[data] No validation/test split for {dataset_name}; using an independent train stream for eval."
            )
            val_ds = _load_streaming_split(
                dataset_name,
                dataset_config,
                "train",
                cache_dir,
                seed + 1,
                shuffle_buffer,
            )

    text_col = _find_text_column(train_ds)
    if jax.process_index() == 0:
        print(f"[data] Using text column '{text_col}' for dataset '{dataset_name}'")

    train_stream = DataStream(
        train_ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_col,
        pack_sequences=pack_sequences,
    )
    val_stream = DataStream(
        val_ds,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_col,
        pack_sequences=pack_sequences,
    )
    return train_stream, val_stream


def sample_batch(
    stream: Iterator[Dict[str, np.ndarray]],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Collect B sequences into a batch of shape [B, T]."""
    batch: List[Dict[str, np.ndarray]] = []
    for _ in range(batch_size):
        try:
            batch.append(next(stream))
        except StopIteration:
            raise RuntimeError("Data stream unexpectedly ended")
    ids = np.stack([ex["input_ids"] for ex in batch], axis=0)
    mask = np.stack([ex["attention_mask"] for ex in batch], axis=0)
    return {"input_ids": ids, "attention_mask": mask}
