from typing import Any, Dict, List, Optional, Tuple
import textwrap
import plotly.graph_objects as go
from rich.pretty import pprint as rpprint
from collections import Counter
from pathlib import Path
from datetime import datetime
import json

import matplotlib as mpl

mpl.rcParams["text.usetex"] = False
from matplotlib.patches import Rectangle, PathPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dna import generate, Attention, FeedForward, Identity


def has_routing(model) -> bool:
    return hasattr(model, "routers") and getattr(model, "routers") is not None


def _mpl_safe(s: str) -> str:
    """Make a string safe for Matplotlib text rendering."""
    if s is None:
        return ""
    s = s.replace("\n", "\\n")
    s = s.replace("$", r"\$")
    return s


def _expert_meta(model) -> Optional[Dict[str, Any]]:
    """Collect labels/types for each expert from model.groups."""
    if not has_routing(model) or not hasattr(model, "groups") or len(model.groups) == 0:
        return None

    E = 0
    for g in model.groups:
        E = max(E, int(jnp.max(g["idx"]) + 1))

    labels: List[Optional[str]] = [None] * E
    types: List[Optional[str]] = [None] * E
    counters: Dict[str, int] = {}

    for g in model.groups:
        tname = type(g["static"]).__name__
        cnt = counters.get(tname, 0)
        idxs = list(map(int, jax.device_get(g["idx"]).tolist()))
        for i, e_idx in enumerate(idxs):
            labels[e_idx] = f"{tname}_{cnt + i}"
            types[e_idx] = tname
        counters[tname] = cnt + len(idxs)

    for i in range(E):
        if labels[i] is None:
            labels[i] = f"Expert_{i}"
        if types[i] is None:
            types[i] = "Unknown"

    uniq = sorted(set(types))
    type_to_id = {t: i for i, t in enumerate(uniq)}
    type_ids = jnp.array([type_to_id[t] for t in types], dtype=jnp.int32)

    return dict(
        n_experts=E,
        expert_labels=labels,
        expert_types=types,
        expert_type_ids=type_ids,
        type_to_id=type_to_id,
        id_to_type=uniq,
    )


def _forward_batched_for_stats(
    model,
    ids_bt: jnp.ndarray,
    mask_bt: jnp.ndarray,
    keys_b: jax.Array,
    *,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
):
    """Run model across batch to collect per-hop routing stats."""

    def fwd(x, m, k):
        _, stats = model(
            x,
            key=k,
            inference=True,
            mask=m,
            gumbel_tau=gumbel_tau,
            router_temp=router_temp,
            select_temp=select_temp,
            return_stats=True,
        )
        return stats

    return jax.vmap(fwd, in_axes=(0, 0, 0))(ids_bt, mask_bt, keys_b)


def _type_base_cmap(type_name: str) -> str:
    t = type_name.lower()
    if "ttent" in t:
        return "Blues"
    if "feed" in t or "mlp" in t:
        return "Oranges"
    if "ident" in t:
        return "Greens"
    return "Purples"


def _expert_color_table(meta) -> np.ndarray:
    """RGBA per expert, grouped by type with graded shades."""
    type_ids = np.asarray(meta["expert_type_ids"])
    id_to_type = list(meta["id_to_type"])
    E = type_ids.shape[0]
    colors = np.zeros((E, 4), dtype=np.float32)

    for t_id, t_name in enumerate(id_to_type):
        idxs = np.where(type_ids == t_id)[0]
        if idxs.size == 0:
            continue
        cmap = plt.get_cmap(_type_base_cmap(t_name))
        shades = [0.75] if idxs.size == 1 else np.linspace(0.35, 0.95, idxs.size)
        for j, e in enumerate(sorted(idxs.tolist())):
            rgba = np.array(cmap(float(shades[j])))
            rgba[3] = 1.0
            colors[e] = rgba
    return colors


def evaluate_for_visuals(
    model,
    batch: Dict[str, Any],
    key: jax.Array,
    *,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
    num_examples: int = 2,
):
    """Collect batch + per-example routing info for visuals."""
    if not has_routing(model):
        return None, None

    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"])
    B, _ = ids.shape
    keys = jax.random.split(key, B)

    stats = _forward_batched_for_stats(
        model,
        ids,
        mask,
        keys,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if not isinstance(stats, (tuple, list)) or len(stats) == 0:
        return None, None

    meta = _expert_meta(model)
    if meta is None:
        return None, None

    n_hops = len(stats)
    E = meta["n_experts"]

    # Average importance per hop over batch
    importance_matrix = []
    for hop_stats in stats:
        imp_be = hop_stats["importance"]
        avg = jnp.mean(imp_be, axis=0)
        if avg.shape[0] != E:
            pad = jnp.zeros((E,), dtype=avg.dtype).at[: avg.shape[0]].set(avg)
            avg = pad
        importance_matrix.append(avg)
    importance_matrix = jnp.stack(importance_matrix, axis=0)

    S = int(min(num_examples, B))
    sel_idx = jax.random.choice(
        jax.random.fold_in(key, 123), B, shape=(S,), replace=False
    )

    probs_by_hop, top1_idx, top1_prob, top1_type = [], [], [], []
    for hop_stats in stats:
        p_bte = hop_stats["routing_probs"]
        p_ste = jnp.take(p_bte, sel_idx, axis=0)

        sums = p_ste.sum(axis=-1)
        valid = sums > 1e-9
        p_norm = jnp.where(valid[..., None], p_ste / (sums[..., None] + 1e-9), 0.0)

        idx = jnp.argmax(p_norm, axis=-1)
        val = jnp.take_along_axis(p_norm, idx[..., None], axis=-1)[..., 0]

        type_ids = meta["expert_type_ids"]
        typ = jnp.take(type_ids, idx)

        probs_by_hop.append(p_ste)
        top1_idx.append(idx)
        top1_prob.append(val)
        top1_type.append(typ)

    batch_stats = dict(
        importance_matrix=importance_matrix,
        n_hops=n_hops,
        n_experts=E,
        expert_labels=meta["expert_labels"],
        expert_types=meta["expert_types"],
        expert_type_ids=meta["expert_type_ids"],
        type_to_id=meta["type_to_id"],
        id_to_type=meta["id_to_type"],
    )

    example_stats = dict(
        indices=sel_idx,
        ids=jnp.take(ids, sel_idx, axis=0),
        mask=jnp.take(mask, sel_idx, axis=0),
        probs=jnp.stack(probs_by_hop, axis=0),
        top1_expert=jnp.stack(top1_idx, axis=0),
        top1_prob=jnp.stack(top1_prob, axis=0),
        top1_type_id=jnp.stack(top1_type, axis=0),
    )

    return batch_stats, example_stats


def _collect_top1_indices_all(
    model,
    batch: Dict[str, Any],
    *,
    key: jax.Array,
    gumbel_tau: float,
    router_temp: float,
    select_temp: float,
):
    """Get top-1 expert per token per hop for an entire batch."""
    if not has_routing(model):
        return None, None, None

    ids = jnp.asarray(batch["input_ids"])
    mask = jnp.asarray(batch["attention_mask"]).astype(bool)
    B, T = ids.shape
    keys = jax.random.split(key, B)

    stats = _forward_batched_for_stats(
        model,
        ids,
        mask.astype(jnp.float32),
        keys,
        gumbel_tau=gumbel_tau,
        router_temp=router_temp,
        select_temp=select_temp,
    )
    if not isinstance(stats, (tuple, list)) or len(stats) == 0:
        return None, None, None

    top1_idx = []
    for hop in stats:
        p = jnp.asarray(hop["routing_probs"])
        idx = jnp.argmax(p, axis=-1)
        top1_idx.append(idx)
    top1 = jnp.stack(top1_idx, axis=0)

    meta = _expert_meta(model)
    if meta is None:
        return None, None, None

    return jax.device_get(top1), jax.device_get(mask), meta


def _rgba_to_plotly(rgba: np.ndarray, alpha: float = 0.8) -> str:
    r, g, b, _ = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(int)
    return f"rgba({r},{g},{b},{alpha})"
