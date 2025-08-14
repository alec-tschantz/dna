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


class CosineRouter(eqx.Module):
    prototypes: jnp.ndarray  # (E, P, d)
    scale: float = eqx.field(static=True)
    k: int = eqx.field(static=True)
    P: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        assert k <= n_exp
        self.k = int(k)
        self.P = 2
        self.scale = 10.0
        k_proto = key
        self.prototypes = jax.random.normal(k_proto, (n_exp, self.P, d_model)) * (
            1.0 / jnp.sqrt(d_model)
        )

    def __call__(
        self,
        h: jnp.ndarray,  # (T, d)
        *,
        key: jax.Array | None,
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: float | None = None,
    ):
        # Cosine sims to multiple prototypes per expert, then log-sum-exp across prototypes.
        eps = 1e-6
        h_norm = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + eps)  # (T, d)
        p_norm = self.prototypes / (
            jnp.linalg.norm(self.prototypes, axis=-1, keepdims=True) + eps
        )  # (E,P,d)
        sims = jnp.einsum("td,epd->tep", h_norm, p_norm)  # (T,E,P)
        logits_clean = self.scale * jax.nn.logsumexp(sims, axis=-1)  # (T,E)

        # Mixing
        t_mix = jnp.clip(router_temp, 1e-6, None)
        probs = jnn.softmax(logits_clean / t_mix, axis=-1)

        # Selection (+ optional gumbel)
        t_sel = jnp.clip(
            router_temp if select_temp is None else select_temp, 1e-6, None
        )
        logits_sel = logits_clean / t_sel
        if not inference:
            assert key is not None
            u = jax.random.uniform(key, logits_sel.shape, minval=1e-6, maxval=1 - 1e-6)
            g = -jnp.log(-jnp.log(u))
            logits_sel = logits_sel + gumbel_tau * g

        mask_full = _topk_mask(logits_sel, self.k)
        return mask_full, probs, logits_clean, logits_sel


class NormRouter(eqx.Module):
    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        assert k <= n_exp
        self.k = int(k)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)

    def __call__(
        self,
        h: jnp.ndarray,
        *,
        key: jax.Array | None,
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: float | None = None,
    ):
        logits_clean = jax.vmap(self.proj)(h)  # (T,E)

        # Selection
        t_sel = jnp.clip(
            router_temp if select_temp is None else select_temp, 1e-6, None
        )
        logits_sel = logits_clean / t_sel
        if not inference:
            assert key is not None
            u = jax.random.uniform(key, logits_sel.shape, minval=1e-6, maxval=1 - 1e-6)
            g = -jnp.log(-jnp.log(u))
            logits_sel = logits_sel + gumbel_tau * g
        mask_full = _topk_mask(logits_sel, self.k)  # (T,E)

        # Mixing: concentrate mass only on selected experts, then renormalize per token
        t_mix = jnp.clip(router_temp, 1e-6, None)
        dense = jnn.softmax(logits_clean / t_mix, axis=-1)  # (T,E)
        masked = jnp.where(mask_full, dense, 0.0)
        denom = masked.sum(axis=-1, keepdims=True) + 1e-9
        probs = masked / denom  # (T,E), sparse over top-k

        return mask_full, probs, logits_clean, logits_sel
