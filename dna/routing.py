# dna/routing.py
from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Return a boolean mask marking the top-k entries per row.

    Parameters
    ----------
    logits : jnp.ndarray
        Shape (T, E) - logits over experts for each token.
    k : int
        Number of top experts to select (must be ≤ E).

    Returns
    -------
    mask : jnp.ndarray
        Shape (T, E) boolean mask with k True values per row.
    """
    _, idx = jax.lax.top_k(logits, k)  # (T, k) int
    # Build hard one-hot mask from indices
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)  # (T, E) in {0,1}
    return hard.astype(bool)


class Router(eqx.Module):
    """Linear router for expert selection with training-time exploration.

    Projects token representations to logits over experts, then applies:
    1. Temperature scaling for mixing probabilities
    2. Optional separate temperature for selection
    3. Gumbel noise for exploration during training
    4. Hard top-k selection
    """

    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        """Initialize router.

        Parameters
        ----------
        d_model : int
            Input dimension (model hidden size).
        n_exp : int
            Number of experts to route between.
        k : int
            Number of experts to select per token (must be ≤ n_exp).
        key : PRNGKey
            Random key for weight initialization.
        """
        assert k <= n_exp, f"Router topk ({k}) must be ≤ number of experts ({n_exp})"
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
        """Route tokens to experts.

        Parameters
        ----------
        h : jnp.ndarray
            Token states of shape (T, d_model).
        key : Optional[jax.Array]
            PRNG key for Gumbel noise (required if inference=False).
        inference : bool
            If True, disables Gumbel noise for deterministic routing.
        gumbel_tau : float, default=1.0
            **Exploration noise scale** (training only):
            - Controls amount of Gumbel noise added to selection logits
            - Higher values (e.g., 1.5) = more exploration
            - Lower values (e.g., 0.5) = more exploitation
        router_temperature : float, default=1.0
            **Mixing weight temperature**: How to weight selected experts
            - Controls softmax sharpness for output combination weights
            - Higher values = more uniform mixing across experts
            - Lower values = concentrate weight on top experts
            - Typical range: 0.5 to 2.0
        select_temperature : Optional[float], default=None
            **Selection temperature** (optional): Which experts to select
            - Controls softmax sharpness for top-k selection
            - If None, uses router_temperature
            - Allows independent control of selection vs mixing
            - Higher values = more uniform selection probability
            - Lower values = more confident selection

        Returns
        -------
        mask_full : jnp.ndarray
            Shape (T, E) bool - hard top-k selection mask.
        probs : jnp.ndarray
            Shape (T, E) float - soft routing probabilities for mixing.
        logits_clean : jnp.ndarray
            Shape (T, E) float - raw logits before temperature/noise.
        """
        # Project tokens to expert logits
        logits_clean = jax.vmap(self.proj)(h)  # (T, E)

        # Compute soft probabilities for mixing (no Gumbel noise)
        temp_mix = jnp.clip(router_temperature, 1e-6, None)
        probs = jnn.softmax(logits_clean / temp_mix, axis=-1)

        # Prepare selection logits (may have different temperature)
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

        # Add Gumbel noise during training for exploration
        if not inference:
            assert key is not None, "Router requires a PRNG key during training"
            # Sample Gumbel(0,1) noise
            u = jax.random.uniform(
                key, logits_sel.shape, minval=1e-6, maxval=1.0 - 1e-6
            )
            g = -jnp.log(-jnp.log(u))
            # Scale noise by tau and add to logits
            logits_sel = logits_sel + gumbel_tau * g

        # Hard top-k selection
        mask_full = _topk_mask(logits_sel, self.k)  # (T, E)

        return mask_full, probs, logits_clean
