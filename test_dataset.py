#!/usr/bin/env python3
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 256
BATCH_SIZE = 32
NUM_SHOW = 3  # how many examples to print from each split


def load_tinystories(tok: AutoTokenizer, seq_len: int, split: str = "train"):
    """
    Streaming TinyStories -> fixed length encodings.
    Also zero the attention mask *after* the first EOS in each row (include EOS in loss).
    """
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    def _proc(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"]          # (B, T) np.int64
        attn_mask = enc["attention_mask"]     # (B, T) np.int64 (1=keep, 0=ignore)

        # ensure mask is 0 AFTER the first EOS token in each row (keep EOS counted)
        eos_id = tok.eos_token_id
        for i in range(input_ids.shape[0]):
            row = input_ids[i]
            idx = np.where(row == eos_id)[0]
            if idx.size > 0:
                eos_pos = int(idx[0])
                attn_mask[i, eos_pos + 1:] = 0

        return {"input_ids": input_ids, "attention_mask": attn_mask}

    # Map in reasonably large batches; we’ll still robustly batch downstream.
    return ds.map(_proc, batched=True, batch_size=1024, remove_columns=["text"])


def _normalize_to_2d(arr) -> np.ndarray:
    """Turn (T,), (1, T), or already (B, T) into (B, T)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected array shape {arr.shape} (ndim={arr.ndim})")


def sample_batch(stream_it: Iterable[Dict[str, Any]], bsz: int) -> Dict[str, np.ndarray]:
    """
    Accumulate rows until we have exactly `bsz`. Works whether the stream yields
    single examples or big batches.
    """
    ids_buf, mask_buf, total = [], [], 0
    while total < bsz:
        ex = next(stream_it)
        ids  = _normalize_to_2d(ex["input_ids"])
        mask = _normalize_to_2d(ex["attention_mask"])
        ids_buf.append(ids)
        mask_buf.append(mask)
        total += ids.shape[0]

    ids  = np.concatenate(ids_buf, axis=0)[:bsz].astype(np.int32, copy=False)
    mask = np.concatenate(mask_buf, axis=0)[:bsz].astype(np.int32, copy=False)
    return {"input_ids": ids, "attention_mask": mask}


def _is_prefix_mask(mask: np.ndarray) -> Tuple[bool, int]:
    """
    Check mask is 1...10...0 (single transition). Return (ok, first_zero_idx or -1).
    """
    diffs = np.diff(mask.astype(np.int32))
    illegal = np.where(diffs == +1)[0]  # 0->1 jump after zeros = illegal
    first_zero = int(np.where(mask == 0)[0][0]) if (mask == 0).any() else -1
    return (illegal.size == 0, first_zero)


def _first_eos(ids: np.ndarray, eos_id: int) -> int:
    loc = np.where(ids == eos_id)[0]
    return int(loc[0]) if loc.size else -1


def _print_clipped_ids(ids: np.ndarray, mask: np.ndarray, width: int = 64) -> None:
    """
    Print first/last 'width' tokens with [id] for masked positions.
    """
    def render_slice(a_ids, a_mask):
        parts = []
        for tid, m in zip(a_ids.tolist(), a_mask.tolist()):
            s = str(tid)
            if m == 0:
                s = f"[{s}]"
            parts.append(s)
        return " ".join(parts)

    T = ids.shape[0]
    head = min(width, T)
    tail = min(width, T - head)
    if tail == 0:
        print("ids:  ", render_slice(ids[:head], mask[:head]))
    else:
        print("ids:  ", render_slice(ids[:head], mask[:head]), "...",
              render_slice(ids[-tail:], mask[-tail:]))

    print("mask: ", " ".join(map(str, mask[:head].tolist())),
          " ... " if tail else "",
          " ".join(map(str, mask[-tail:].tolist())) if tail else "")


def visualize_example(idx: int, ids: np.ndarray, mask: np.ndarray, tok: AutoTokenizer) -> None:
    """
    Detailed debug for one example.
    """
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    assert ids.shape == mask.shape == (SEQ_LEN,)
    assert ids.dtype == np.int32 and mask.dtype == np.int32

    valid_len = int(mask.sum())
    first_pad = int(np.where(mask == 0)[0][0]) if (mask == 0).any() else -1
    num_eos = int(np.sum(ids == eos_id))
    first_eos = _first_eos(ids, eos_id)
    prefix_ok, first_zero_idx = _is_prefix_mask(mask)

    print(f"Example {idx}:")
    print(f"- valid_len: {valid_len} / {SEQ_LEN} | first_pad_idx: {first_pad if first_pad>=0 else 'none'}")
    print(f"- eos_id: {eos_id} | pad_id: {pad_id} | num_eos_in_ids: {num_eos} | first_eos_idx: {first_eos if first_eos>=0 else 'none'}")
    print(f"- mask_prefix_ok: {prefix_ok} | first_zero_idx: {first_zero_idx if first_zero_idx>=0 else 'none'}")
    _print_clipped_ids(ids, mask, width=64)

    # Build marked decode (slow but fine for NUM_SHOW small)
    decoded_marked = []
    for t_id, m in zip(ids.tolist(), mask.tolist()):
        piece = tok.decode([t_id], skip_special_tokens=False)
        decoded_marked.append("▁<PAD>▁" if m == 0 else piece)
    decoded_marked = "".join(decoded_marked)

    print("decoded (pads marked):")
    print(decoded_marked)
    # quick invariant: all masked positions should be pad_id if pad==eos
    if (mask == 0).any():
        masked_ids = ids[mask == 0]
        ok = bool(np.all(masked_ids == pad_id))
        print("masked -> pad_id check:", "OK" if ok else f"FAIL (found {masked_ids[:10]})")
    else:
        print("no masked positions")
    print("-" * 80)


def batch_summary(ids_b: np.ndarray, mask_b: np.ndarray, tok: AutoTokenizer) -> None:
    """
    Print batch-level stats: length histogram, EOS alignment checks, etc.
    """
    eos_id = tok.eos_token_id
    valid_lens = mask_b.sum(axis=1).astype(int)
    first_eoses = np.array([_first_eos(ids_b[i], eos_id) for i in range(ids_b.shape[0])])
    num_eoses = (ids_b == eos_id).sum(axis=1)
    first_zeros = np.array([np.where(mask_b[i] == 0)[0][0] if (mask_b[i] == 0).any() else -1
                            for i in range(mask_b.shape[0])])

    print("=== Batch summary ===")
    print("shapes:", ids_b.shape, mask_b.shape, "| dtypes:", ids_b.dtype, mask_b.dtype)
    print("valid_len: min/med/max =", int(valid_lens.min()),
          int(np.median(valid_lens)), int(valid_lens.max()))
    # small histogram
    bins = [0, 32, 64, 96, 128, 160, 192, 224, 256]
    hist, edges = np.histogram(valid_lens, bins=bins)
    print("valid_len histogram:")
    for c, s, e in zip(hist.tolist(), edges[:-1], edges[1:]):
        print(f"  {s:3d}-{e:3d}: {c}")

    no_eos = int(np.sum(first_eoses < 0))
    eos_at_end = int(np.sum(first_eoses == (valid_lens - 1)))
    print(f"first_eos: none={no_eos}, at_end={eos_at_end}")

    # mask prefix violations
    prefix_ok = []
    for i in range(ids_b.shape[0]):
        ok, _ = _is_prefix_mask(mask_b[i])
        prefix_ok.append(ok)
    bad_prefix = np.where(np.logical_not(np.array(prefix_ok)))[0]
    if bad_prefix.size:
        print("WARNING: non-prefix masks at indices:", bad_prefix.tolist())
    else:
        print("mask prefix check: all OK")

    # verify policy: zeros start right AFTER first EOS (i.e., include EOS)
    policy_ok = []
    for i in range(mask_b.shape[0]):
        if first_eoses[i] >= 0:
            ok = (first_zeros[i] == -1) or (first_zeros[i] == first_eoses[i] + 1)
        else:
            ok = True  # no EOS in window -> no constraint
        policy_ok.append(ok)
    bad_policy = np.where(np.logical_not(np.array(policy_ok)))[0]
    if bad_policy.size:
        print("WARNING: EOS masking policy violated at indices:", bad_policy.tolist())

    print("======================")


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # pad==eos

    for split in ["train", "validation"]:
        print(f"\n===== SPLIT: {split} =====")
        ds = load_tinystories(tok, SEQ_LEN, split)
        it = iter(ds)

        batch = sample_batch(it, BATCH_SIZE)
        ids_b = batch["input_ids"]         # (B, T) int32
        mask_b = batch["attention_mask"]   # (B, T) int32

        batch_summary(ids_b, mask_b, tok)
        print()

        for i in range(min(NUM_SHOW, ids_b.shape[0])):
            visualize_example(i, ids_b[i], mask_b[i], tok)


if __name__ == "__main__":
    main()
