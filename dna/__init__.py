import jax
from jax import numpy as jnp

from .dense import Dense
from .dna import DNA


def generate(
    model,
    prompt_ids: jnp.ndarray,
    max_new_tokens: int,
    temperature: float,               # kept for API parity; see note below
    *,
    key,                               # unused for greedy; kept for API parity
    biases=None,                       # e.g. [n_hops] identity biases (or None)
    gumbel: bool = False,              # router Gumbel flag (DNA only)
    gumbel_tau: float = 1.0,           # router Gumbel scale (DNA only)
    temp: float = 1.0,                 # router softmax temperature (DNA only)
    greedy: bool = True,               # new: if True do argmax; if False sample
):
    """
    Greedy (argmax) generation by default. Works for Dense and DNA.

    - For DNA, we pass router controls (biases/gumbel/temp) so routing at
      generation matches your training setup.
    - For Dense, those args are ignored.
    - `temperature` only affects sampling; for greedy it does nothing
      (argmax is invariant to positive scaling).
    """
    pad_id = 0
    prompt_len = prompt_ids.shape[0]
    total_len = prompt_len + max_new_tokens

    tokens = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)

    is_dna = isinstance(model, DNA)

    def step(carry, _):
        toks, cur, k = carry
        # Keep split for API compatibility; we don't actually use randomness in greedy mode.
        k, subkey = jax.random.split(k)

        if is_dna:
            logits, _ = model(
                toks,
                key=subkey,
                inference=True,
                biases=biases,
                gumbel=gumbel,
                gumbel_tau=gumbel_tau,
                temp=temp,
            )
        else:
            logits, _ = model(toks, key=subkey, inference=True)

        last_logits = jnp.take(logits, cur - 1, axis=0)

        if greedy:
            # Greedy decode: argmax over vocabulary.
            next_tok = jnp.argmax(last_logits, axis=-1).astype(jnp.int32)
        else:
            # Stochastic decode: categorical sample with output temperature.
            scaled = last_logits / jnp.clip(temperature, 1e-6, None)
            next_tok = jax.random.categorical(subkey, scaled).astype(jnp.int32)

        toks = toks.at[cur].set(next_tok)
        return (toks, cur + 1, k), None

    (tokens, _, _), _ = jax.lax.scan(
        step,
        (tokens, prompt_len, key),
        None,
        length=max_new_tokens,
    )
    return tokens
