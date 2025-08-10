# -----------------------------------------------------------------------------
# router.py
# -----------------------------------------------------------------------------
# Abstract router API + a concrete TopKRouter:
# - Produces *only* routing decisions: (T,E) hard mask and (T,E) soft weights.
# - No capacity logic here — that is handled by Modules.
# - Supports temperature and optional Gumbel exploration for training.
# - Optional "identity bias" convention: last column is identity expert.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import abc
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


# ---------- utilities ----------


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Row-wise hard top-k along last dim."""
    _, idx = jax.lax.top_k(logits, k)
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)
    return hard.astype(bool)


# ---------- public output container ----------


@dataclass(frozen=True)
class RouterOutput:
    """Routing decisions for a single sequence (token-major)."""

    mask: (
        jnp.ndarray
    )  # (T, E) bool   — hard top-k mask over *all* experts (incl. identity if present)
    weight: (
        jnp.ndarray
    )  # (T, E) float  — soft weights (e.g., softmax probs) for combination
    probs: jnp.ndarray  # (T, E) float  — same as weight (can differ for other routers)
    logits: (
        jnp.ndarray
    )  # (T, E) float  — raw router logits (pre temp/bias/gumbel for analysis)


# ---------- abstract base ----------


# ---------- base (NOT abstract on __call__) ----------
class RouterBase(eqx.Module):
    """Base router interface. Override __call__ in subclasses."""

    def __call__(
        self,
        h: jnp.ndarray,  # (T, d)
        *,
        key: Optional[jax.Array],
        temp: float = 1.0,
        sample_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        bias_row: Optional[jnp.ndarray] = None,  # (E,) or (T,E)
    ) -> RouterOutput:
        raise NotImplementedError("__call__ must be implemented by Router subclasses")


# ---------- concrete: Top-k router with linear projection ----------


class TopKRouter(RouterBase):
    """Linear(d_model→E) → (optional bias/Gumbel) → top-k (hard) + softmax (soft)."""

    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        self.k = k
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)


    def __call__(
        self,
        h: jnp.ndarray,
        *,
        key: Optional[jax.Array],
        temp: float = 1.0,
        sample_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        bias_row: Optional[jnp.ndarray] = None,
    ) -> RouterOutput:
        # --------- raw logits ---------
        logits_clean = jax.vmap(self.proj)(h)  # (T, E)
        E = logits_clean.shape[-1]

        # --------- optional identity bias (applies to selection path only) ---------
        if bias_row is not None:
            b_last = bias_row[-1] if bias_row.ndim == 2 else bias_row
            id_mask = (jnp.arange(E) == (E - 1))[None, :]  # (1, E)
            logits_sel = logits_clean + b_last * id_mask
        else:
            logits_sel = logits_clean

        # --------- temperatures ---------
        tau_soft = jnp.clip(temp, 1e-6, None)
        tau_sel = jnp.clip(gumbel_tau, 1e-6, None)

        # --------- soft weights from *clean* logits (for gradients/combination) ---------
        probs = jnn.softmax(logits_clean / tau_soft, axis=-1)  # (T, E)

        # --------- selection scores: (logits + gumbel) / tau_sel ---------
        if sample_gumbel:
            assert key is not None, "Gumbel sampling requires a PRNG key."
            u = jax.random.uniform(key, logits_sel.shape, minval=1e-6, maxval=1.0 - 1e-6)
            g = -jnp.log(-jnp.log(u))  # Gumbel(0,1)
            logits_for_topk = (logits_sel + g) / tau_sel
        else:
            logits_for_topk = logits_sel / tau_sel

        # --------- hard mask ---------
        mask = _topk_mask(logits_for_topk, self.k)  # (T, E) bool

        # For this router, `weight` equals `probs`; other routers could differ.
        return RouterOutput(mask=mask, weight=probs, probs=probs, logits=logits_clean)
