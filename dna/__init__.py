# -----------------------------------------------------------------------------
# generate.py
# -----------------------------------------------------------------------------
# Streaming autoregressive generation that is backend-agnostic:
# - Works with the new split Model (attention_mask_t kwarg)
# - Also works with older Dense/DNA models (attention_mask kwarg)
#
# The model is called over the full-length working buffer each step, but it
# must ignore positions >= cur via the attention mask we provide.
# -----------------------------------------------------------------------------

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any, Optional


DEFAULT_PROMPTS = [
    "One day",
    "Once upon a time",
    "In a faraway land",
    "The experiment began",
    "Suddenly, without warning",
]


# ---------- internal: forward wrapper to handle old/new mask kwarg ----------

def _forward_model(
    model: Any,
    toks_t: jnp.ndarray,           # (T_total,)
    *,
    key,
    attn_mask_t: jnp.ndarray,      # (T_total,) 1/0
    biases,
    router_temp: float,
    gumbel_tau: float,
):
    """Calls model with either attention_mask_t (new) or attention_mask (legacy)."""
    # Common kwargs for both codepaths
    base_kwargs = dict(
        key=key,
        inference=True,             # dropout off for generation
        biases=biases,
        gumbel=False,               # no gumbel at inference
        gumbel_tau=gumbel_tau,
        temp=router_temp,
    )

    # Try new API first
    try:
        return model(
            toks_t,
            attention_mask_t=attn_mask_t,
            **base_kwargs,
        )
    except TypeError:
        # Fallback to legacy name
        return model(
            toks_t,
            attention_mask=attn_mask_t,
            **base_kwargs,
        )


# ---------- public: generate tokens ----------

def generate(
    model: Any,                    # new Model or legacy Dense/DNA
    prompt_ids: jnp.ndarray,       # (T_prompt,)
    max_new_tokens: int,
    temperature: float = 0.8,
    *,
    key,
    biases=None,
    gumbel: bool = False,          # ignored at inference (kept for API compat)
    gumbel_tau: float = 1.0,
    router_temp: float = 1.0,
    greedy: bool = False,
    pad_id: int = 0,
    eos_id: Optional[int] = None,
) -> jnp.ndarray:
    """Autoregressive sampling.

    Shapes:
      prompt_ids: (T_prompt,) int32
      returns:    (T_prompt + max_new_tokens,) int32
    """
    if eos_id is None:
        eos_id = pad_id

    prompt_len = int(prompt_ids.shape[0])
    total_len = int(prompt_len + max_new_tokens)

    # Working token buffer (T_total,)
    tokens = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)

    # Greedy if requested OR temperature <= 0
    force_greedy = greedy or (temperature <= 0.0)

    # ---------- one decoding step (scanned) ----------
    def step(carry, _):
        toks, cur, done, k = carry
        k, subkey = jax.random.split(k)

        # Attention mask: positions < cur are valid; others masked out
        attn_mask = (jnp.arange(total_len, dtype=jnp.int32) < cur).astype(jnp.int32)  # (T_total,)

        # Full forward; model must ignore masked positions
        logits, _ = _forward_model(
            model,
            toks,
            key=subkey,
            attn_mask_t=attn_mask,
            biases=biases,
            router_temp=router_temp,
            gumbel_tau=gumbel_tau,
        )

        # Next-token distribution is at position cur-1 (the last visible token)
        vocab = logits.shape[-1]
        cur_idx = cur - jnp.asarray(1, jnp.int32)
        last_logits = jax.lax.dynamic_slice(logits, (cur_idx, 0), (1, vocab))[0]  # (V,)

        # Sample or take argmax
        def _greedy(_op):
            _k, lg = _op
            return jnp.argmax(lg, axis=-1).astype(jnp.int32)

        def _sample(op):
            k2, lg = op
            scaled = lg / jnp.clip(temperature, 1e-6, None)
            return jax.random.categorical(k2, scaled).astype(jnp.int32)

        operand = (subkey, last_logits)
        next_tok = jax.lax.cond(force_greedy, _greedy, _sample, operand)

        # After EOS, write PAD forever
        next_tok = jax.lax.select(done, jnp.asarray(pad_id, jnp.int32), next_tok)

        toks = toks.at[cur].set(next_tok)
        new_done = done | (next_tok == eos_id)
        return (toks, cur + 1, new_done, k), None

    # Scan for max_new_tokens steps
    (tokens, _, _, _), _ = jax.lax.scan(
        step,
        (tokens, jnp.asarray(prompt_len, jnp.int32), jnp.asarray(False), key),
        None,
        length=int(max_new_tokens),
    )
    return tokens
