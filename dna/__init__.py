import jax
from jax import numpy as jnp

from .dense import Dense
from .dna import DNA


def generate(
    model,
    prompt_ids: jnp.ndarray,
    max_new_tokens: int,
    temperature: float,
    *,
    key,
    biases=None,
    gumbel: bool = False,
    gumbel_tau: float = 1.0,
    temp: float = 1.0,
    greedy: bool = True,
    pad_id: int = 0
):
    prompt_len = prompt_ids.shape[0]
    total_len = prompt_len + max_new_tokens

    tokens = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)


    def step(carry, _):
        toks, cur, k = carry
        k, subkey = jax.random.split(k)
        attn_mask = (jnp.arange(toks.shape[0]) < cur).astype(jnp.int32)

        logits, _ = model(
            toks,
            key=subkey,
            inference=True,
            attention_mask=attn_mask,
            biases=biases,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            temp=temp,
        )
       
        last_logits = jnp.take(logits, cur - 1, axis=0)

        if greedy:
            next_tok = jnp.argmax(last_logits, axis=-1).astype(jnp.int32)
        else:
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
