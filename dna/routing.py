from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx


# ------------------------------- utils -------------------------------- #


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Boolean mask of top-k per row. logits: (T, E) -> mask: (T, E)."""
    _, idx = jax.lax.top_k(logits, k)  # (T, k)
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)  # (T, E) in {0,1}
    return hard.astype(bool)


def _selection_logits(logits_clean: jnp.ndarray, select_temp: float) -> jnp.ndarray:
    t_sel = jnp.clip(select_temp, 1e-6, None)
    return logits_clean / t_sel


def _maybe_add_gumbel(
    logits_sel: jnp.ndarray,
    *,
    inference: bool,
    key: Optional[jax.Array],
    gumbel_tau: float,
) -> jnp.ndarray:
    if inference:
        return logits_sel
    assert key is not None, "Router needs PRNG key during training"
    u = jax.random.uniform(key, logits_sel.shape, minval=1e-6, maxval=1 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    return logits_sel + gumbel_tau * g


def _select_mask_and_logits(
    logits_clean: jnp.ndarray,
    *,
    router_temp: float,
    select_temp: Optional[float],
    inference: bool,
    key: Optional[jax.Array],
    gumbel_tau: float,
    k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t_sel = router_temp if select_temp is None else select_temp
    logits_sel = _selection_logits(logits_clean, t_sel)
    logits_sel = _maybe_add_gumbel(
        logits_sel, inference=inference, key=key, gumbel_tau=gumbel_tau
    )
    mask_full = _topk_mask(logits_sel, k)
    return logits_sel, mask_full


def _mixing_probs(
    logits_clean: jnp.ndarray,
    *,
    router_temp: float,
    mask: Optional[jnp.ndarray],
    norm_probs: bool,
) -> jnp.ndarray:
    """Return per-token mixing probabilities from clean logits.
    If norm_probs=True, concentrate mass on `mask` and renormalize per token.
    """
    t_mix = jnp.clip(router_temp, 1e-6, None)
    dense = jnn.softmax(logits_clean / t_mix, axis=-1)  # (T, E)
    if not norm_probs:
        return dense
    assert mask is not None, "mask is required when norm_probs=True"
    masked = jnp.where(mask, dense, 0.0)
    denom = masked.sum(axis=-1, keepdims=True) + 1e-9
    return masked / denom


# ------------------------------ routers ------------------------------- #


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
        norm_probs: bool = False,  # if True, renorm over selected experts
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        logits_clean = jax.vmap(self.proj)(h)  # (T, E)

        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
        )

        probs = _mixing_probs(
            logits_clean, router_temp=router_temp, mask=mask_full, norm_probs=norm_probs
        )
        return mask_full, probs, logits_clean, logits_sel


class CosineRouter(eqx.Module):
    """Router using cosine-similarity prototypes."""

    prototypes: jnp.ndarray  # (E, P, d)
    scale: float = eqx.field(static=True)
    k: int = eqx.field(static=True)
    P: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        assert k <= n_exp
        self.k = int(k)
        self.P = 2
        self.scale = 10.0
        self.prototypes = jax.random.normal(key, (n_exp, self.P, d_model)) * (
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
        norm_probs: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        eps = 1e-6
        h_norm = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + eps)  # (T, d)
        p_norm = self.prototypes / (
            jnp.linalg.norm(self.prototypes, axis=-1, keepdims=True) + eps
        )  # (E, P, d)
        sims = jnp.einsum("td,epd->tep", h_norm, p_norm)  # (T, E, P)
        logits_clean = self.scale * jax.nn.logsumexp(sims, axis=-1)  # (T, E)

        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
        )
        probs = _mixing_probs(
            logits_clean, router_temp=router_temp, mask=mask_full, norm_probs=norm_probs
        )
        return mask_full, probs, logits_clean, logits_sel


class LinearSequenceRouter(eqx.Module):
    """Recurrent router that accumulates sequence context before routing.

    Same API; supports normed probs via keyword.
    """

    w_in: eqx.nn.Linear  # (d -> d) input-to-hidden
    w_rec: eqx.nn.Linear  # (d -> d) hidden-to-hidden (no bias)
    proj: eqx.nn.Linear  # (d -> E) hidden-to-expert logits
    h0: jnp.ndarray  # (d,) learnable/initial hidden state
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        assert k <= n_exp, f"topk ({k}) must be ≤ n_exp ({n_exp})"
        self.k = int(k)
        k_in, k_rec, k_out, _ = jax.random.split(key, 4)
        self.w_in = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_in)
        self.w_rec = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_rec)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=k_out)
        self.h0 = jnp.zeros((d_model,), dtype=jnp.float32)

    def __call__(
        self,
        h: jnp.ndarray,  # (T, d)
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        norm_probs: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        def step(carry, x_t):
            s_prev = carry  # (d,)
            s_t = jnp.tanh(self.w_rec(s_prev) + self.w_in(x_t))  # (d,)
            logits_t = self.proj(s_t)  # (E,)
            return s_t, logits_t

        _, logits_seq = jax.lax.scan(step, self.h0, h)  # (T, E)
        logits_clean = logits_seq

        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
        )
        probs = _mixing_probs(
            logits_clean, router_temp=router_temp, mask=mask_full, norm_probs=norm_probs
        )
        return mask_full, probs, logits_clean, logits_sel


class SequenceRouter(eqx.Module):
    """Recurrent router with expressive MLP (SwiGLU) head + gated state update.

    Update rule (per token)
    -----------------------
      z_t   = W_rec s_{t-1} + W_in x_t
      c_t   = SiLU(z_t)                                # candidate state
      g_t   = sigmoid(W_g x_t)                         # elementwise gate in (0,1)
      s_t   = (1 - g_t) * s_{t-1} + g_t * c_t          # gated residual → stable grads

      # Expressive projection (SwiGLU with expansion factor r)
      u_t   = SiLU(W_up_a s_t) * (W_up_b s_t)          # (rd,)
      logits_t = W_down u_t                             # (E)

    Notes
    -----
    - SwiGLU improves expressivity while remaining cheap and well-behaved.
    - Elementwise gate lets the model decide how much to carry vs. refresh per dim.
    - All mixing/selection behavior is handled by shared helpers for consistency.
    """

    # Recurrent core
    w_in: eqx.nn.Linear  # d -> d
    w_rec: eqx.nn.Linear  # d -> d
    w_gate: eqx.nn.Linear  # d -> d  (elementwise gate)

    # Expressive projection (SwiGLU with expansion r)
    up_a: eqx.nn.Linear  # d -> r*d
    up_b: eqx.nn.Linear  # d -> r*d
    down: eqx.nn.Linear  # r*d -> E

    h0: jnp.ndarray  # (d,) learnable/initial hidden state

    # Statics
    k: int = eqx.field(static=True)
    r: int = eqx.field(static=True)  # expansion factor

    def __init__(self, d_model: int, n_exp: int, k: int, *, key, expansion: int = 4):
        assert k <= n_exp, f"topk ({k}) must be ≤ n_exp ({n_exp})"
        self.k = int(k)
        self.r = int(expansion)

        k_in, k_rec, k_gate, k_upa, k_upb, k_down, _ = jax.random.split(key, 7)
        # No biases to match your style elsewhere
        self.w_in = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_in)
        self.w_rec = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_rec)
        self.w_gate = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_gate)

        self.up_a = eqx.nn.Linear(d_model, d_model * self.r, use_bias=False, key=k_upa)
        self.up_b = eqx.nn.Linear(d_model, d_model * self.r, use_bias=False, key=k_upb)
        self.down = eqx.nn.Linear(d_model * self.r, n_exp, use_bias=False, key=k_down)

        self.h0 = jnp.zeros((d_model,), dtype=jnp.float32)

    def __call__(
        self,
        h: jnp.ndarray,  # (T, d)
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        norm_probs: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # Recurrent scan to produce contextual logits_clean
        def step(s_prev, x_t):
            # Candidate update and gate
            z_t = self.w_rec(s_prev) + self.w_in(x_t)  # (d,)
            c_t = jnn.silu(z_t)  # (d,)
            g_t = jnn.sigmoid(self.w_gate(x_t))  # (d,)
            s_t = (1.0 - g_t) * s_prev + g_t * c_t  # (d,)

            # Expressive projection to experts (SwiGLU)
            u = jnn.silu(self.up_a(s_t)) * self.up_b(s_t)  # (r*d,)
            logits_t = self.down(u)  # (E,)
            return s_t, logits_t

        _, logits_seq = jax.lax.scan(step, self.h0, h)  # (T, E)
        logits_clean = logits_seq

        # Selection + mask (shared helpers)
        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
        )

        # Mixing (shared helper; supports normed or dense probs)
        probs = _mixing_probs(
            logits_clean, router_temp=router_temp, mask=mask_full, norm_probs=norm_probs
        )

        return mask_full, probs, logits_clean, logits_sel
