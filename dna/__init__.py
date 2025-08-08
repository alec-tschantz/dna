import jax
import jax.numpy as jnp

from .dense import Dense
from .dna import DNA

DEFAULT_PROMPTS = [
    "One day",
    "Once upon a time",
    "In a faraway land",
    "The experiment began",
    "Suddenly, without warning",
]

def generate(
    model: Dense | DNA,
    prompt_ids: jnp.ndarray,
    max_new_tokens: int,
    temperature: float = 0.8,
    *,
    key,
    biases=None,
    gumbel: bool = False,
    gumbel_tau: float = 1.0,
    router_temp: float = 1.0,
    greedy: bool = False,
    pad_id: int = 0,
    eos_id: int | None = None,
):
    if eos_id is None:
        eos_id = pad_id

    prompt_len = int(prompt_ids.shape[0])
    total_len = int(prompt_len + max_new_tokens)

    tokens = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)

    force_greedy = greedy or (temperature <= 0.0)

    def step(carry, _):
        toks, cur, done, k = carry
        k, subkey = jax.random.split(k)

        # Mask allows only tokens < cur to attend; rest are ignored
        attn_mask = (jnp.arange(total_len, dtype=jnp.int32) < cur).astype(jnp.int32)

        # Full-length forward; model should ignore masked positions
        logits, _ = model(
            toks,
            key=subkey,
            inference=True,         # dropout off for generation
            attention_mask=attn_mask,
            biases=biases,
            gumbel=False,           # no gumbel at inference
            gumbel_tau=gumbel_tau,
            temp=router_temp,
        )

        # Grab logits at position (cur-1) with dynamic slice
        vocab = logits.shape[-1]
        cur_idx = cur - jnp.asarray(1, jnp.int32)
        last_logits = jax.lax.dynamic_slice(logits, (cur_idx, 0), (1, vocab))[0]

        def true_fn(op):
            _, lg = op
            return jnp.argmax(lg, axis=-1).astype(jnp.int32)

        def false_fn(op):
            k2, lg = op
            scaled = lg / jnp.clip(temperature, 1e-6, None)
            return jax.random.categorical(k2, scaled).astype(jnp.int32)

        operand = (subkey, last_logits)
        next_tok = jax.lax.cond(force_greedy, true_fn, false_fn, operand)

        # After EOS, keep writing PAD
        next_tok = jax.lax.select(done, jnp.asarray(pad_id, jnp.int32), next_tok)

        toks = toks.at[cur].set(next_tok)
        new_done = done | (next_tok == eos_id)
        return (toks, cur + 1, new_done, k), None

    (tokens, _, _, _), _ = jax.lax.scan(
        step,
        (tokens, jnp.asarray(prompt_len, jnp.int32), jnp.asarray(False), key),
        None,
        length=int(max_new_tokens),
    )
    return tokens
