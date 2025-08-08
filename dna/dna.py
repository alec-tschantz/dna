from typing import Tuple, Dict, Any, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from dna.nn import Embedding, Dropout, RMSNorm, Attention, FeedForward, rope_cos_sin


# routing utils


def _topk_mask(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    _, idx = jax.lax.top_k(logits, k)
    hard = jnn.one_hot(idx, logits.shape[-1]).sum(axis=-2)
    return hard.astype(bool)


def _capacity_select(mask_te: jnp.ndarray, gate_te: jnp.ndarray, capacity: int):
    m = mask_te.T
    g = jnp.where(m, gate_te.T, -jnp.inf)
    cap = int(min(capacity, g.shape[1]))
    _, top_idx = jax.lax.top_k(g, cap)
    E, C = top_idx.shape
    T = g.shape[1]
    slot = jnp.zeros((E, C, T), dtype=g.dtype)
    slot = slot.at[jnp.arange(E)[:, None], jnp.arange(C)[None, :], top_idx].set(1.0)
    slot = slot * m[:, None, :]
    kept = slot.sum(1).astype(bool)
    return slot, kept


def _sig(mod):
    st = jax.tree_util.tree_structure(eqx.filter(mod, eqx.is_array))
    static = str(eqx.filter(mod, lambda x: not eqx.is_array(x)))
    return type(mod).__name__, str(st), static


def _stack(mods):
    params = [eqx.filter(m, eqx.is_array) for m in mods]
    static = eqx.filter(mods[0], lambda x: not eqx.is_array(x))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *params), static


class Router(eqx.Module):
    proj: eqx.nn.Linear
    k: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_exp: int, k: int, *, key):
        self.k = k
        self.proj = eqx.nn.Linear(d_model, n_exp, use_bias=False, key=key)

    def __call__(
        self,
        h: jnp.ndarray,
        bias_row: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.Array] = None,
        sample_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        temp: float = 1.0,
    ):
        # clean logits for probability path
        logits_clean = jax.vmap(self.proj)(h)
        E = logits_clean.shape[-1]

        # bias touches only top-k selection path (identity = last col)
        if bias_row is not None:
            b_id = jax.lax.stop_gradient(bias_row[-1])
            id_mask = (jnp.arange(E) == (E - 1))[None, :]
            logits_topk = logits_clean + b_id * id_mask
        else:
            logits_topk = logits_clean

        # router probabilities from clean logits
        probs = jnn.softmax(logits_clean / jnp.clip(temp, 1e-6, None), axis=-1)

        # optional gumbel noise for discrete top-k
        if sample_gumbel:
            assert key is not None
            u = jax.random.uniform(key, logits_topk.shape, minval=1e-6, maxval=1 - 1e-6)
            g = -jnp.log(-jnp.log(u))
            logits_sel = logits_topk + gumbel_tau * g
        else:
            logits_sel = logits_topk

        mask_full = _topk_mask(logits_sel, self.k)
        return mask_full, probs, logits_clean


class DNA(eqx.Module):
    embed: Embedding
    dropout: Dropout
    backbone: Tuple[eqx.Module, ...]
    routers: Tuple[Router, ...]
    groups: Tuple[Dict[str, Any], ...]
    ln_out: RMSNorm

    n_modules: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(
        self,
        vocab: int,
        d_model: int,
        n_heads: int,
        n_modules: int,
        capacity: int,
        topk: int,
        n_hops: int,
        mlp_mult: int,
        dropout: float,
        rope_base: float,
        num_backbone: int,
        key,
    ):
        self.n_modules = n_modules
        self.capacity = capacity
        self.n_heads = n_heads
        self.rope_base = rope_base

        k_embed, k_modules, k_routers, k_backbone = jax.random.split(key, 4)
        self.embed = Embedding(vocab, d_model, key=k_embed)
        self.dropout = Dropout(dropout)
        self.ln_out = RMSNorm(d_model)

        # build alternating experts (Attn/FFN)
        module_keys = jax.random.split(k_modules, n_modules)
        experts = []
        for i, k_i in enumerate(module_keys):
            if i % 2 == 0:
                experts.append(Attention(d_model, n_heads, dropout, key=k_i))
            else:
                experts.append(FeedForward(d_model, mlp_mult, dropout, key=k_i))

        # pack experts with identical signatures
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for idx, mod in enumerate(experts):
            s = _sig(mod)
            entry = buckets.setdefault(s, {"idx": [], "mods": []})
            entry["idx"].append(idx)
            entry["mods"].append(mod)
        grouped = []
        for b in buckets.values():
            params, st = _stack(b["mods"])
            grouped.append(
                dict(idx=jnp.array(b["idx"], jnp.int32), params=params, static=st)
            )
        grouped.sort(key=lambda d: int(d["idx"][0]))
        self.groups = tuple(grouped)

        # optional dense backbone
        if num_backbone > 0:
            bb_keys = jax.random.split(k_backbone, num_backbone)
            bb = []
            for i in range(num_backbone):
                if i % 2 == 0:
                    bb.append(Attention(d_model, n_heads, dropout, key=bb_keys[i]))
                else:
                    bb.append(FeedForward(d_model, mlp_mult, dropout, key=bb_keys[i]))
            self.backbone = tuple(bb)
        else:
            self.backbone = tuple()

        # routers (identity is implicit last expert)
        router_keys = jax.random.split(k_routers, n_hops)
        total_experts = n_modules + 1
        self.routers = tuple(
            Router(d_model, total_experts, topk, key=k) for k in router_keys
        )

    # gather hop stats in a single place
    def _stats(
        self,
        *,
        kept: jnp.ndarray,
        probs_tr: jnp.ndarray,
        mask_tr: jnp.ndarray,
        rho: jnp.ndarray,
        logits_clean: jnp.ndarray,
        id_in_topk: jnp.ndarray,
        token_mask: jnp.ndarray,
        k_routes: int,
    ) -> Dict[str, Any]:
        load = kept.sum(axis=1).astype(jnp.int32)
        importance = probs_tr.sum(axis=0)
        selected_edges = mask_tr.sum()
        kept_edges = kept.sum()
        cap_drop_frac_edges = jnp.where(
            selected_edges > 0, (selected_edges - kept_edges) / selected_edges, 0.0
        )
        p_tr = probs_tr
        p_tr_sum = p_tr.sum(axis=1, keepdims=True) + 1e-9
        p_tr_norm = p_tr / p_tr_sum
        ent = -(p_tr_norm * jnp.log(p_tr_norm + 1e-9)).sum(axis=1)
        ent = ent / jnp.log(p_tr.shape[1] + 1e-9)
        util = (load > 0).mean()
        load_norm = load / (jnp.sum(load) + 1e-9)
        load_std = jnp.std(load_norm)
        rho_mean = rho.mean()
        rho_min = rho.min()
        rho_max = rho.max()
        valid_T = token_mask.sum().astype(jnp.int32)
        total_routes = jnp.asarray(k_routes * valid_T, jnp.int32)
        id_topk_count = jnp.sum(jnp.where(token_mask, id_in_topk, False)).astype(
            jnp.int32
        )
        id_rate = jnp.where(valid_T > 0, id_topk_count / valid_T, 0.0)
        return dict(
            load=load,
            importance=importance,
            rho_mean=rho_mean,
            rho_min=rho_min,
            rho_max=rho_max,
            id_topk_rate=id_rate,
            id_topk_count=id_topk_count,
            total_routes=total_routes,
            entropy_mean=ent.mean(),
            entropy_min=ent.min(),
            entropy_max=ent.max(),
            util=util,
            load_std=load_std,
            cap_drop_frac_edges=cap_drop_frac_edges,
            selected_edges=jnp.asarray(selected_edges, jnp.int32),
            kept_edges=jnp.asarray(kept_edges, jnp.int32),
            id_logit_mean=jnp.mean(logits_clean[:, -1]),
            nonid_logit_mean=jnp.mean(logits_clean[:, :-1]),
        )

    # single routing+expert hop
    def _hop(
        self,
        h: jnp.ndarray,
        router: Router,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        *,
        key,
        inference: bool,
        bias_row: Optional[jnp.ndarray],
        sample_gumbel: bool,
        gumbel_tau: float,
        temp: float,
        token_mask: jnp.ndarray,
    ):
        k_route, k_exec = jax.random.split(key)

        # route (identity included in routing)
        mask_full, probs_full, logits_clean = router(
            h,
            bias_row,
            key=k_route,
            sample_gumbel=sample_gumbel,
            gumbel_tau=gumbel_tau,
            temp=temp,
        )

        # drop identity column for dispatch
        probs_tr = probs_full[:, :-1]
        mask_tr = mask_full[:, :-1]
        id_in_topk = mask_full[:, -1]

        # ignore masked tokens entirely
        probs_tr = jnp.where(token_mask[:, None], probs_tr, 0.0)
        mask_tr = jnp.where(token_mask[:, None], mask_tr, False)
        id_in_topk = jnp.where(token_mask, id_in_topk, False)

        # capacity packing
        slot, kept = _capacity_select(mask_tr, probs_tr, self.capacity)

        # gather packed inputs
        xin = jnp.einsum("ect,td->ecd", slot, h)
        cos_r = jnp.einsum("ect,td->ecd", slot, cos)
        sin_r = jnp.einsum("ect,td->ecd", slot, sin)

        # execute experts (modules are residual)
        outs = []
        for gi, g in enumerate(self.groups):
            n_g = len(g["idx"])
            sub_keys = jax.random.split(jax.random.fold_in(k_exec, gi), n_g)
            inp = xin[g["idx"]]
            c = cos_r[g["idx"]]
            s = sin_r[g["idx"]]
            out_g = jax.vmap(
                lambda p, x, c1, s1, k1: eqx.combine(p, g["static"])(
                    x, c1, s1, key=k1, inference=inference
                )
            )(g["params"], inp, c, s, sub_keys)
            outs.append(out_g)
        expert_out = jnp.concatenate(outs, axis=0)

        # combine without renorm; identity not dispatched
        kept_t = kept.T
        combine_w = jnp.where(kept_t, probs_tr, 0.0)
        rho = combine_w.sum(axis=1, keepdims=True)
        combine = jnp.einsum("ecd,ect,et->td", expert_out, slot, combine_w.T)
        h_next = h + combine - rho * h
        h_next = jnp.where(token_mask[:, None], h_next, h)

        stats = self._stats(
            kept=kept,
            probs_tr=probs_tr,
            mask_tr=mask_tr,
            rho=rho,
            logits_clean=logits_clean,
            id_in_topk=id_in_topk,
            token_mask=token_mask,
            k_routes=router.k,
        )
        return h_next, stats

    @eqx.filter_jit
    def __call__(
        self,
        ids: jnp.ndarray,
        *,
        key,
        inference: bool,
        attention_mask: Optional[jnp.ndarray] = None,
        biases: Optional[jnp.ndarray] = None,
        gumbel: bool = False,
        gumbel_tau: float = 1.0,
        temp: float = 1.0,
    ):
        T = ids.shape[0]
        token_mask = (
            jnp.ones((T,), dtype=bool)
            if attention_mask is None
            else attention_mask.astype(bool)
        )

        # embed + dropout
        h = jax.vmap(self.embed)(ids)
        key, sub = jax.random.split(key)
        h = self.dropout(h, key=sub, inference=inference)

        # rotary caches
        d_h = self.embed.weight.shape[1] // self.n_heads
        cos, sin = rope_cos_sin(T, d_h, self.rope_base)

        # optional dense backbone
        for i, mod in enumerate(self.backbone):
            key, sub = jax.random.split(key)
            if isinstance(mod, Attention):
                out = mod(
                    h, cos, sin, key=sub, inference=inference, attention_mask=token_mask
                )
            else:
                out = mod(h, cos, sin, key=sub, inference=inference)
            out = jnp.where(token_mask[:, None], out - h, 0.0) + h
            h = out

        # router hops
        stats_all = []
        for hop_idx, R in enumerate(self.routers):
            key, sub = jax.random.split(key)
            if biases is None:
                bias_row = None
            elif biases.ndim == 1:
                E_total = self.n_modules + 1
                zero = jnp.zeros((E_total,), dtype=jnp.float32)
                bias_row = zero.at[-1].set(biases[hop_idx])
            else:
                bias_row = biases[hop_idx]
            h, st = self._hop(
                h,
                R,
                cos,
                sin,
                key=sub,
                inference=inference,
                bias_row=bias_row,
                sample_gumbel=gumbel and (not inference),
                gumbel_tau=gumbel_tau,
                temp=temp,
                token_mask=token_mask,
            )
            stats_all.append(st)

        # norm + logits
        h = jax.vmap(self.ln_out)(h)
        logits = jax.vmap(lambda t: t @ self.embed.weight.T)(h)
        return logits, tuple(stats_all)
