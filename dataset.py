from __future__ import annotations
from typing import Dict, Iterator, Optional

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


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
    input_ids = enc["input_ids"][0]  # (T,)
    attn_mask = enc["attention_mask"][0]  # (T,)

    eos_id = tok.eos_token_id
    if eos_id is not None:
        pos = np.where(input_ids == eos_id)[0]
        if pos.size > 0:
            first_eos = int(pos[0])
            attn_mask[first_eos + 1 :] = 0

    return {
        "input_ids": input_ids.astype(np.int32),  # (T,)
        "attention_mask": attn_mask.astype(np.int32),  # (T,)
    }


def load_dataset_stream(
    name: str,
    tok: PreTrainedTokenizerBase,
    seq_len: int,
    split: str = "train",
    config: Optional[str] = None,
    text_key: str = "text",
    shuffle_buffer: int = 10_000,
    seed: int = 0,
) -> Iterator[Dict[str, np.ndarray]]:
    base = load_dataset(name, config, split=split, streaming=True)

    def epoch_iter(ds, s: int):
        ds_cur = (
            ds.shuffle(buffer_size=shuffle_buffer, seed=s)
            if shuffle_buffer and shuffle_buffer > 0
            else ds
        )
        for ex in ds_cur:
            txt = ex.get(text_key, "")
            if isinstance(txt, str) and txt:
                yield _process_example(tok, txt, seq_len)

    cur_seed = seed
    while True:
        for item in epoch_iter(base, cur_seed):
            yield item
        cur_seed += 1


def sample_batch(
    stream: Iterator[Dict[str, np.ndarray]], batch_size: int
) -> Dict[str, np.ndarray]:
    ids_list, mask_list = [], []
    for _ in range(batch_size):
        ex = next(stream)
        ids_list.append(ex["input_ids"])  # (T,)
        mask_list.append(ex["attention_mask"])  # (T,)
    input_ids = np.stack(ids_list, axis=0)  # (B, T)
    attention_mask = np.stack(mask_list, axis=0)  # (B, T)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
