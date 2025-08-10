# router.py
# -----------------------------------------------------------------------------
# Minimal, practical router for distributed neural architectures.
#
# Responsibilities:
#   • Project hidden states → expert logits
#   • Add optional per-hop bias (shape [E])
#   • Produce a HARD top‑k mask for selection + SOFT weights for combination
#   • If inference=False  → add Gumbel(0,1) to selection path (exploration)
#
# -----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx


# ---- utility ---------------------------------------------------------------


def _topk_mask(row_logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Row-wise hard top‑k along last dim.
    Args:
        row_logits: (..., E) scores
        k:          number of experts to select
    Returns:
        bool mask with the same trailing shape (..., E)
    """
    _, idx = jax.lax.top_k(row_logits, k)
    hard = jnn.one_hot(idx, row_logits.shape[-1]).sum(axis=-2)
    return hard.astype(bool)


# ---- public output ---------------------------------------------------------


@dataclass(frozen=True)
class RouterOutput:
    """Routing decisions for a single sequence (token-major).

    mask:   (T, E) bool   HARD top‑k selection mask
    weight: (T, E) float  SOFT weights used to combine expert outputs (softmax)
    logits: (T, E) float  Clean logits (pre‑Gumbel), useful for stats/analysis
    """

    mask: jnp.ndarray
    weight: jnp.ndarray
    logits: jnp.ndarray


# ---- concrete router -------------------------------------------------------


class TopKRouter(eqx.Module):
    """Linear(d_model → E) → [bias] → {softmax, top‑k}.

    If `inference` is False, we add Gumbel noise *only* to the selection path.
    The soft weights are always computed from the clean logits.
    """

    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_experts: int, k: int, *, key):
        self.k = int(k)
        self.proj = eqx.nn.Linear(d_model, n_experts, use_bias=False, key=key)

    def __call__(
        self,
        h_td: jnp.ndarray,  # (T, d)
        *,
        key: Optional[jax.Array],
        inference: bool,
        bias_e: Optional[jnp.ndarray] = None,  # (E,)
    ) -> RouterOutput:
        # Project tokens → expert logits (token-major)
        logits_te = jax.vmap(self.proj)(h_td)  # (T, E)

        # Optional per-hop expert bias (shape [E])
        if bias_e is not None:
            assert (
                bias_e.ndim == 1 and bias_e.shape[0] == logits_te.shape[-1]
            ), "bias_e must be shape (E,) and match router output size"
            logits_te = logits_te + bias_e[None, :]

        # Soft combination weights from clean logits
        weight_te = jnn.softmax(logits_te, axis=-1)  # (T, E)

        # Hard selection mask: add Gumbel noise only when training
        if inference:
            sel_scores = logits_te
        else:
            assert (
                key is not None
            ), "Training path requires a PRNG key for Gumbel noise."
            u = jax.random.uniform(key, logits_te.shape, minval=1e-6, maxval=1.0 - 1e-6)
            g = -jnp.log(-jnp.log(u))
            sel_scores = logits_te + g

        mask_te = _topk_mask(sel_scores, self.k)  # (T, E) bool
        return RouterOutput(mask=mask_te, weight=weight_te, logits=logits_te)
