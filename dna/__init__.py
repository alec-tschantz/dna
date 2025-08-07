import jax
from jax import numpy as jnp

from .dense import Dense
from .dna import DNA


def generate(
    model, prompt_ids: jnp.ndarray, max_new_tokens: int, temperature: float, *, key
):
    pad_id = 0
    prompt_len = prompt_ids.shape[0]
    total_len = prompt_len + max_new_tokens

    tokens = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)

    def step(carry, _):
        toks, cur, k = carry
        logits, _ = model(toks, key=k, inference=True)
        next_logits = jnp.take(logits, cur - 1, axis=0) / temperature
        k, subkey = jax.random.split(k)
        next_tok = jax.random.categorical(subkey, next_logits)
        toks = toks.at[cur].set(next_tok)
        return (toks, cur + 1, k), None

    (tokens, _, _), _ = jax.lax.scan(
        step,
        (tokens, prompt_len, key),
        None,
        length=max_new_tokens,
    )
    return tokens
