#!/usr/bin/env python
import numpy as np
from transformers import AutoTokenizer
from datasets import disable_caching
from scripts import dataloader 

disable_caching()

# === Config ===
DATASETS = [
    ("roneneldan/TinyStories", "default", ["train", "validation"]),
    # ("wikitext", "wikitext-2-raw-v1", ["train", "validation"]),  # commented out
]
SEQ_LEN = 256
BATCH_SIZE = 2  # smaller batch for easier printing

# === Tokenizer ===
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def inspect_dataset(name, config, split):
    print(f"\n=== Inspecting {name} ({config}) split={split} ===")

    stream = dataloader.load_dataset_stream(
        name=name,
        tok=tok,
        seq_len=SEQ_LEN,
        split=split,
        config=config,
        text_key="text",
        shuffle_buffer=1000,
        seed=123,
    )

    batch = dataloader.sample_batch(stream, BATCH_SIZE)
    ids = batch["input_ids"]  # (B, T)
    mask = batch["attention_mask"]  # (B, T)

    print(f"input_ids shape: {ids.shape}, dtype: {ids.dtype}")
    print(f"mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask values: {set(np.unique(mask))}")

    for b in range(BATCH_SIZE):
        row_ids = ids[b]
        row_mask = mask[b]
        valid_len = int(row_mask.sum())

        raw_ids = row_ids.tolist()
        decoded_with_pads = tok.decode(raw_ids, skip_special_tokens=False)
        no_pad_ids = raw_ids[:valid_len]
        decoded_no_pads = tok.decode(no_pad_ids, skip_special_tokens=False)

        print(f"\n--- Example {b} ---")
        print(f"Token IDs (raw): {raw_ids}")
        print(f"\nDecoded (with pads): {repr(decoded_with_pads)}")
        print(f"\nDecoded (no pads): {repr(decoded_no_pads)}")


if __name__ == "__main__":
    for name, config, splits in DATASETS:
        for split in splits:
            inspect_dataset(name, config, split)
