from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from dna.modules import RMSNorm, Dropout


# ---------- utils ----------


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    _, idx = jax.lax.top_k(logits, k)  # (T, k)
    hard = jnn.one_hot(idx, logits.shape[-1])  # (T, k, E)
    return hard.sum(axis=-2).astype(bool)  # (T, E)


def _selection_logits(logits_clean: jnp.ndarray, select_temp: float) -> jnp.ndarray:
    t = jnp.clip(select_temp, 1e-6, None)
    return logits_clean / t


def _maybe_add_gumbel(
    logits_sel: jnp.ndarray,
    *,
    inference: bool,
    key: Optional[jax.Array],
    gumbel_tau: float,
) -> jnp.ndarray:
    if inference:
        return logits_sel
    u = jax.random.uniform(key, logits_sel.shape, minval=1e-6, maxval=1.0 - 1e-6)
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
    token_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t_sel = router_temp if select_temp is None else select_temp
    logits_sel = _selection_logits(logits_clean, t_sel)
    logits_sel = _maybe_add_gumbel(
        logits_sel, inference=inference, key=key, gumbel_tau=gumbel_tau
    )
    if token_mask is not None:
        logits_sel = jnp.where(token_mask[:, None], logits_sel, -jnp.inf)
    mask_full = _topk_mask(logits_sel, k)
    if token_mask is not None:
        mask_full = jnp.where(token_mask[:, None], mask_full, False)
    return logits_sel, mask_full


def _mixing_probs(
    logits_clean: jnp.ndarray,
    *,
    router_temp: float,
    mask: Optional[jnp.ndarray],
    norm_probs: bool,
    token_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    t_mix = jnp.clip(router_temp, 1e-6, None)
    dense = jnn.softmax(logits_clean / t_mix, axis=-1)  # (T, E)
    if token_mask is not None:
        dense = jnp.where(token_mask[:, None], dense, 0.0)  # zero out pads
    if not norm_probs:
        return dense
    masked = jnp.where(mask, dense, 0.0)
    denom = masked.sum(axis=-1, keepdims=True)
    out = masked / jnp.clip(denom, 1e-9, None)
    if token_mask is not None:
        out = jnp.where(token_mask[:, None], out, 0.0)
    return out


# ---------- routers ----------


class LinearRouter(eqx.Module):
    proj: eqx.nn.Linear
    dropout: Dropout
    # TODO: add layer norm
    k: int = eqx.field(static=True)
    norm_probs: bool = eqx.field(static=True)

    def __init__(
        self, d_model: int, n_exp: int, k: int, dropout: float, norm_probs: bool, *, key
    ):
        self.k = int(k)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)
        self.dropout = Dropout(dropout)
        self.norm_probs = norm_probs

    def __call__(
        self,
        h: jnp.ndarray,
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        token_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)
        logits_clean = jax.vmap(self.proj)(h)  # (T, E)

        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
            token_mask=token_mask,
        )
        probs = _mixing_probs(
            logits_clean,
            router_temp=router_temp,
            mask=mask_full,
            norm_probs=self.norm_probs,
            token_mask=token_mask,
        )
        return mask_full, probs, logits_clean, logits_sel


class CosineRouter(eqx.Module):
    prototypes: jnp.ndarray  # (E, P, d)
    dropout: Dropout
    scale: float = eqx.field(static=True)
    norm_probs: bool = eqx.field(static=True)
    k: int = eqx.field(static=True)
    P: int = eqx.field(static=True)

    def __init__(
        self, d_model: int, n_exp: int, k: int, dropout: float, norm_probs: bool, *, key
    ):
        self.k = int(k)
        self.P = 2
        self.scale = 10.0
        self.norm_probs = norm_probs
        self.prototypes = jax.random.normal(key, (n_exp, self.P, d_model)) * (
            1.0 / jnp.sqrt(d_model)
        )
        self.dropout = Dropout(dropout)

    def __call__(
        self,
        h: jnp.ndarray,
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        token_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        eps = 1e-6
        h_norm = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + eps)  # (T, d)
        p_norm = self.prototypes / (
            jnp.linalg.norm(self.prototypes, axis=-1, keepdims=True) + eps
        )  # (E,P,d)
        sims = jnp.einsum("td,epd->tep", h_norm, p_norm)  # (T,E,P)
        logits_clean = self.scale * jax.nn.logsumexp(sims, axis=-1)  # (T,E)

        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
            token_mask=token_mask,
        )
        probs = _mixing_probs(
            logits_clean,
            router_temp=router_temp,
            mask=mask_full,
            norm_probs=self.norm_probs,
            token_mask=token_mask,
        )
        return mask_full, probs, logits_clean, logits_sel


class SequenceRouter(eqx.Module):
    w_in: eqx.nn.Linear
    w_rec: eqx.nn.Linear
    proj: eqx.nn.Linear
    h0: jnp.ndarray
    h_norm: RMSNorm
    x_norm: RMSNorm
    dropout: Dropout
    k: int = eqx.field(static=True)
    norm_probs: bool = eqx.field(static=True)

    def __init__(
        self, d_model: int, n_exp: int, k: int, dropout: float, norm_probs: bool, *, key
    ):
        self.k = int(k)
        k_in, k_rec, k_out, _ = jax.random.split(key, 4)
        self.w_in = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_in)
        self.w_rec = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k_rec)
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=k_out)
        self.h_norm = RMSNorm(d_model)
        self.x_norm = RMSNorm(d_model)
        self.dropout = Dropout(dropout)

        self.h0 = jnp.zeros((d_model,), dtype=jnp.float32)
        self.norm_probs = norm_probs

    def __call__(
        self,
        h: jnp.ndarray,
        *,
        key: Optional[jax.Array],
        inference: bool,
        gumbel_tau: float = 1.0,
        router_temp: float = 1.0,
        select_temp: Optional[float] = None,
        token_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        T = h.shape[0]
        if token_mask is None:
            token_mask = jnp.ones((T,), dtype=bool)

        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        def step(s_prev, inputs):
            x_t, m_t = inputs
            s_prop = jnn.silu(
                self.w_rec(self.h_norm(s_prev)) + self.w_in(self.x_norm(x_t))
            )
            s_t = jnp.where(m_t, s_prop, s_prev)
            logits_t = self.proj(s_t)
            return s_t, logits_t

        _, logits_seq = jax.lax.scan(step, self.h0, (h, token_mask))  # (T,E)
        logits_clean = logits_seq

        logits_sel, mask_full = _select_mask_and_logits(
            logits_clean,
            router_temp=router_temp,
            select_temp=select_temp,
            inference=inference,
            key=key,
            gumbel_tau=gumbel_tau,
            k=self.k,
            token_mask=token_mask,
        )
        probs = _mixing_probs(
            logits_clean,
            router_temp=router_temp,
            mask=mask_full,
            norm_probs=self.norm_probs,
            token_mask=token_mask,
        )
        return mask_full, probs, logits_clean, logits_sel
