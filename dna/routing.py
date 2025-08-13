# dna/routing.py
from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Boolean mask of top-k per row. logits: (T, E) -> mask: (T, E)."""
    _, idx = jax.lax.top_k(logits, k)  # (T, k)
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)  # (T, E) in {0,1}
    return hard.astype(bool)


class Router(eqx.Module):
    """Linear router with temps + optional Gumbel for selection."""

    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        assert k <= n_exp, f"topk ({k}) must be ≤ n_exp ({n_exp})"
        self.k = int(k)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)

    def __call__(
        self,
        h: jnp.ndarray,  # (T, d)
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,  # for mixing
        select_temp: Optional[float] = None,  # for top-k selection
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns:
          mask_full:   (T, E) bool — hard top-k using *selection* scores
          probs:       (T, E) float — soft mixing probabilities (no noise)
          logits_clean:(T, E) float — raw linear logits (pre-temp)
          logits_sel:  (T, E) float — selection logits after temp (+gumbel if train)
        """
        logits_clean = jax.vmap(self.proj)(h)  # (T, E)

        # Mixing probs use router_temperature (no exploration)
        t_mix = jnp.clip(router_temp, 1e-6, None)
        probs = jnn.softmax(logits_clean / t_mix, axis=-1)  # (T, E)

        # Selection logits may use different temperature (+ gumbel if training)
        t_sel = jnp.clip(
            router_temp if select_temp is None else select_temp,
            1e-6,
            None,
        )
        logits_sel = logits_clean / t_sel
        if not inference:
            assert key is not None, "Router needs PRNG key during training"
            u = jax.random.uniform(key, logits_sel.shape, minval=1e-6, maxval=1 - 1e-6)
            g = -jnp.log(-jnp.log(u))
            logits_sel = logits_sel + gumbel_tau * g

        mask_full = _topk_mask(logits_sel, self.k)  # (T, E)
        return mask_full, probs, logits_clean, logits_sel
