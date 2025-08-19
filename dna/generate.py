from __future__ import annotations
from typing import Optional, List, Dict, Any
from jaxtyping import Float, Int, Bool, Array

import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def sample(
    model: eqx.Module,
    prompt_ids: Int[Array, "T0"],
    max_new_tokens: int,
    pad_id: int,
    *,
    key: jax.Array,
    temperature: float = 0.8,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    router_temp: float = 1.0,
    select_temp: Optional[float] = None,
    gumbel_tau: float = 1.0,
) -> Int[Array, "T"]:
    """Optimized sampling with optional top-p and top-k."""
    if eos_id is None:
        eos_id = pad_id
    
    prompt_len = prompt_ids.shape[0]
    total_len = prompt_len + max_new_tokens
    
    # Initialize tokens
    tokens = jnp.full((total_len,), pad_id, dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_ids)
    
    # Sampling configuration
    use_sampling = temperature > 0.0
    
    def step(carry, _):
        toks, pos, done, k = carry
        k, subkey = jax.random.split(k)
        
        # Create attention mask
        attn_mask = jnp.arange(total_len) < pos
        
        # Get logits
        logits, _ = model(
            toks, key=subkey, inference=True, mask=attn_mask,
            gumbel_tau=gumbel_tau, router_temp=router_temp, select_temp=select_temp
        )
        
        # Get last token logits
        last_logits = logits[pos - 1]
        
        # Apply temperature
        if use_sampling:
            last_logits = last_logits / jnp.maximum(temperature, 1e-6)
        
        # Apply top-k if specified
        if top_k is not None and top_k > 0:
            top_k_val = jnp.minimum(top_k, last_logits.shape[-1])
            top_k_logits, top_k_indices = jax.lax.top_k(last_logits, top_k_val)
            last_logits = jnp.full_like(last_logits, -jnp.inf).at[top_k_indices].set(top_k_logits)
        
        # Apply top-p if specified
        if top_p is not None and top_p < 1.0 and use_sampling:
            sorted_logits = jnp.sort(last_logits)[::-1]
            sorted_probs = jnn.softmax(sorted_logits)
            cumsum_probs = jnp.cumsum(sorted_probs)
            cutoff_idx = jnp.searchsorted(cumsum_probs, top_p)
            cutoff_logit = sorted_logits[cutoff_idx]
            last_logits = jnp.where(last_logits >= cutoff_logit, last_logits, -jnp.inf)
        
        # Sample or take argmax
        if use_sampling:
            next_tok = jax.random.categorical(k, last_logits).astype(jnp.int32)
        else:
            next_tok = jnp.argmax(last_logits).astype(jnp.int32)
        
        # Update tokens if not done
        next_tok = jax.lax.select(done, pad_id, next_tok)
        toks = toks.at[pos].set(next_tok)
        
        # Check for EOS
        new_done = done | (next_tok == eos_id)
        
        return (toks, pos + 1, new_done, k), None
    
    # Run generation
    (tokens, _, _, _), _ = jax.lax.scan(
        step,
        (tokens, prompt_len, False, key),
        None,
        length=max_new_tokens
    )
    
    return tokens


def generate(
    model: eqx.Module,
    tok,
    *,
    key: jax.Array,
    gen_len: int = 100,
    per_prompt: int = 1,
    router_temp: float = 1.0,
    select_temp: Optional[float] = None,
    gumbel_tau: float = 1.0,
    prompts: Optional[List[str]] = None,
    n_examples: int = 5,
    temperature: float = 0.8,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate completions for multiple prompts."""
    if prompts is None:
        prompts = ["Once upon a time"]
    
    results = []
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else pad_id
    
    for prompt in prompts[:n_examples]:
        prompt_ids = jnp.array(tok.encode(prompt), dtype=jnp.int32)
        
        # Generate multiple completions per prompt
        key, *subkeys = jax.random.split(key, per_prompt + 1)
        subkeys = jnp.stack(subkeys)
        
        # Vectorized generation
        def gen_single(k):
            return sample(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=gen_len,
                pad_id=pad_id,
                temperature=temperature,
                key=k,
                router_temp=router_temp,
                select_temp=select_temp,
                gumbel_tau=gumbel_tau,
                top_p=top_p,
                top_k=top_k,
                eos_id=eos_id,
            )
        
        all_tokens = jax.vmap(gen_single)(subkeys)
        
        # Process completions
        prompt_result = {"prompt": prompt, "completions": []}
        
        for tokens in all_tokens:
            tokens = jax.device_get(tokens)
            
            # Find EOS position
            if eos_id in tokens:
                eos_pos = int(jnp.argmax(tokens == eos_id))
                tokens = tokens[:eos_pos + 1]
                stopped_eos = True
            else:
                stopped_eos = False
            
            # Decode
            text = tok.decode(tokens, skip_special_tokens=True)
            text_with_special = tok.decode(tokens, skip_special_tokens=False)
            
            prompt_result["completions"].append({
                "tokens": tokens.tolist(),
                "text": text,
                "text_with_special": text_with_special,
                "length": len(tokens),
                "stopped_eos": stopped_eos,
            })
        
        results.append(prompt_result)
    
    return results