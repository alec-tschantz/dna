# dna/routing.py
from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


# -----------------------------------------------------------------------------
# Helper: hard top-k mask along the last axis
# -----------------------------------------------------------------------------


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Return a boolean mask marking the top-k entries per row.

    logits : (T, E)
    k      : int (must be ≤ E)
    """
    _, idx = jax.lax.top_k(logits, k)  # (T, k) int
    # Build a hard one-hot mask from indices and collapse over k
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)  # (T, E) in {0,1}
    return hard.astype(bool)


# -----------------------------------------------------------------------------
# Router: produces top-k selections over experts
# -----------------------------------------------------------------------------


class Router(eqx.Module):
    """Linear router over experts with training-time Gumbel exploration.

    n_exp is the number of experts you provide (include Identity() yourself if desired).
    """

    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        assert k <= n_exp, "Router topk must be ≤ number of experts"
        self.k = int(k)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)

    def __call__(
        self,
        h: jnp.ndarray,
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temperature: float = 1.0,
        select_temperature: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Route a sequence of token states.

        Parameters
        ----------
        h : jnp.ndarray
            Token states of shape ``(T, d_model)``.
        key : Optional[jax.Array]
            PRNG key for Gumbel when ``inference=False``.
        inference : bool
            If True: deterministic (no Gumbel).
        gumbel_tau : float
            Scale of Gumbel noise for exploration (selection only).
        router_temperature : float
            Temperature for *mixing* softmax (probabilities).
        select_temperature : Optional[float]
            Temperature for *selection* logits (top-k). If None, uses `router_temperature`.

        Returns
        -------
        mask_full : (T, E) bool
            Hard selection mask.
        probs : (T, E) float
            Soft routing probabilities from temperature-scaled softmax (mixing).
        logits_clean : (T, E) float
            Raw pre-temperature, pre-Gumbel logits (for diagnostics / ranking).
        """
        # Project tokens to logits over all experts.
        logits_clean = jax.vmap(self.proj)(h)  # (T, E)

        # --- Soft probabilities for mixing (no Gumbel) ---
        temp_mix = jnp.clip(router_temperature, 1e-6, None)
        probs = jnn.softmax(logits_clean / temp_mix, axis=-1)

        # --- Selection logits (may have its own temperature) ---
        temp_sel = jnp.clip(
            (
                select_temperature
                if select_temperature is not None
                else router_temperature
            ),
            1e-6,
            None,
        )
        logits_sel = logits_clean / temp_sel
        if not inference:
            assert key is not None, "Router: a PRNG key is required during training."
            # Standard Gumbel(0,1)
            u = jax.random.uniform(
                key, logits_sel.shape, minval=1e-6, maxval=1.0 - 1e-6
            )
            g = -jnp.log(-jnp.log(u))
            logits_sel = logits_sel + gumbel_tau * g

        # Hard top-k over (maybe) noisy/temperature-scaled logits
        mask_full = _topk_mask(logits_sel, self.k)  # (T, E)
        return mask_full, probs, logits_clean
