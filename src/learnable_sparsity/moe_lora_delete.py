"""Batched MoE dispatch for Qwen3MoeSparseMoeBlock across all training and inference methods.

Replaces HuggingFace Qwen3MoeSparseMoeBlock's Python loop over experts (which issues 3 * E
host-device syncs and E separate kernel launches per layer) with token-chunked padded torch.bmm
across all four methods:
  1. LoRA Delete (LoRADeleteSparseLinear): custom autograd with single batched E Triton kernel
     calls (build_masked_weight in forward, lowrank_grads in backward) and CPU fallback.
  2. Dense Delete (SparseLinear): batched autograd through diff_mask, preserving sparse_weight_norm
     strictly on routed experts (total_counts > 0).
  3. Full Fine-Tuning / MTH / Frozen Reference (nn.Linear): stacked weights with autograd.
  4. PEFT LoRA (lora.Linear): stacked base + active adapter deltas with autograd.

Touching all E experts in the stacked tensor / autograd Function also ensures unrouted experts
receive clean zero gradients rather than None, preventing FSDP2 post-backward reduce-scatter
deadlocks across ranks. Falls back to the original eager forward whenever any expert submodule
has forward or forward-pre hooks registered (e.g., Wanda calibration).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lora_deletion.lowrank import lowrank_logits

from learnable_sparsity import sparse_linear
from learnable_sparsity.lora_delete_linear import LoRADeleteSparseLinear
from learnable_sparsity.sparse_linear import SparseLinear

_MOE_TOKEN_CHUNK = 16384
_BATCHED_FWD_CALLS = 0
_BATCHED_BWD_CALLS = 0


def _pack_chunk_routing(selected_chunk: torch.Tensor, E: int):
    """Build padded [E, C] scatter/gather indices for one chunk of [T_c, K] selected experts."""
    K = selected_chunk.shape[1]
    flat_exp = selected_chunk.reshape(-1)
    M = flat_exp.numel()
    counts = torch.bincount(flat_exp, minlength=E)
    C = max(int(counts.max().item()), 1)
    perm = torch.argsort(flat_exp, stable=True)
    sorted_exp = flat_exp.index_select(0, perm)
    starts = torch.zeros(E, dtype=torch.long, device=flat_exp.device)
    starts[1:] = counts.cumsum(0)[:-1]
    pos_in_exp = torch.arange(M, device=flat_exp.device) - starts.index_select(0, sorted_exp)
    flat_pad_idx = sorted_exp * C + pos_in_exp
    token_idx = perm.div(K, rounding_mode="floor")
    slot_idx = perm.remainder(K)
    return C, counts, token_idx, slot_idx, flat_pad_idx


def _build_masked_weight_batched(W, B, A, col_norm, threshold, prior: float, alpha: float):
    """Batched [E, d_out, d_in] soft-masked weight W * sigmoid(alpha * (G0 + H)) on CUDA (Triton) or CPU."""
    if W.is_cuda:
        from lora_deletion.kernels import triton_ops

        return triton_ops.build_masked_weight(W, B, A, col_norm, threshold, prior, alpha)
    logits = lowrank_logits(W, B, A, col_norm, threshold, prior)
    m = torch.sigmoid(alpha * logits)
    w_tilde_f = W.to(B.dtype) * m
    return w_tilde_f.to(W.dtype), w_tilde_f.square().sum()


def _lowrank_grads_batched(W, dW_tilde, B, A, col_norm, threshold, prior: float, alpha: float, reg_coef: float):
    """Batched [E, d_out, r] / [E, r, d_in] low-rank mask gradients on CUDA (Triton) or CPU."""
    if W.is_cuda:
        from lora_deletion.kernels import triton_ops

        return triton_ops.lowrank_grads(W, dW_tilde, B, A, col_norm, threshold, prior, alpha, reg_coef)
    scale = alpha / math.sqrt(B.shape[-1])
    logits = lowrank_logits(W, B, A, col_norm, threshold, prior)
    m = torch.sigmoid(alpha * logits)
    w = W.to(m.dtype)
    dm = dW_tilde.to(m.dtype) * w + (2.0 * reg_coef) * w.square() * m
    dlogit = dm * m * (1.0 - m) * scale
    dB = torch.bmm(dlogit, A.transpose(1, 2)).to(B.dtype)
    dA = torch.bmm(B.transpose(1, 2), dlogit).to(A.dtype)
    return dB, dA


class _BatchedMoELoRADeleteFn(torch.autograd.Function):
    """Exact token-chunked autograd through all E experts with LoRADeleteSparseLinear projections."""

    @staticmethod
    def forward(
        ctx, hs_2d, routing_weights, selected_experts, E,
        flags, ld_specs, save_for_bwd,
        W_gate, W_up, W_down, cn_th_pairs, *ba_flat,
    ):
        T, D = hs_2d.shape
        dtype = W_gate.dtype
        Ws = (W_gate, W_up, W_down)
        W_effs = []
        B_list = []
        A_list = []
        ba_offset = 0
        ld_idx = 0
        for proj_idx, is_ld in enumerate(flags):
            W = Ws[proj_idx]
            if is_ld:
                B = torch.stack([t.float() for t in ba_flat[ba_offset : ba_offset + E]], dim=0)
                A = torch.stack([t.float() for t in ba_flat[ba_offset + E : ba_offset + 2 * E]], dim=0)
                ba_offset += 2 * E
                cn, th = cn_th_pairs[ld_idx]
                prior, alpha, _ = ld_specs[ld_idx]
                ld_idx += 1
                W_tilde, _ = _build_masked_weight_batched(W, B, A, cn, th, prior, alpha)
                W_effs.append(W_tilde)
                B_list.append(B)
                A_list.append(A)
            else:
                W_effs.append(W)

        W_gate_eff, W_up_eff, W_down_eff = W_effs
        W_gate_T = W_gate_eff.transpose(1, 2)
        W_up_T = W_up_eff.transpose(1, 2)
        W_down_T = W_down_eff.transpose(1, 2)

        final_hs = torch.zeros_like(hs_2d)
        total_counts = torch.zeros(E, dtype=torch.long, device=hs_2d.device)
        chunk_meta = []
        for t0 in range(0, T, _MOE_TOKEN_CHUNK):
            t1 = min(t0 + _MOE_TOKEN_CHUNK, T)
            hs_c = hs_2d[t0:t1]
            rw_c = routing_weights[t0:t1]
            sel_c = selected_experts[t0:t1]
            C, counts_c, token_idx, slot_idx, flat_pad_idx = _pack_chunk_routing(sel_c, E)

            x_perm = hs_c.index_select(0, token_idx).to(dtype)
            x_pad = hs_c.new_zeros((E * C, D), dtype=dtype)
            x_pad.index_copy_(0, flat_pad_idx, x_perm)
            x_pad = x_pad.view(E, C, D)

            g_pad = torch.bmm(x_pad, W_gate_T)
            u_pad = torch.bmm(x_pad, W_up_T)
            h_pad = (F.silu(g_pad) * u_pad).to(dtype)
            out_pad = torch.bmm(h_pad, W_down_T)

            out_perm = out_pad.view(E * C, D).index_select(0, flat_pad_idx)
            w_perm = rw_c[token_idx, slot_idx]
            out_weighted = (out_perm * w_perm.unsqueeze(-1)).to(hs_2d.dtype)
            final_hs[t0:t1].index_add_(0, token_idx, out_weighted)
            if save_for_bwd:
                total_counts.add_(counts_c)
                chunk_meta.append((t0, t1, C, token_idx, slot_idx, flat_pad_idx))

        if save_for_bwd:
            flat_cn_th = [t for pair in cn_th_pairs for t in pair]
            ctx.save_for_backward(
                hs_2d, routing_weights,
                W_gate, W_up, W_down,
                W_gate_eff, W_up_eff, W_down_eff,
                total_counts,
                *B_list, *A_list, *flat_cn_th,
            )
            ctx.chunk_meta = chunk_meta
            ctx.meta = (E, flags, ld_specs)
        return final_hs

    @staticmethod
    def backward(ctx, d_final_hs):
        global _BATCHED_BWD_CALLS
        _BATCHED_BWD_CALLS += 1
        E, flags, ld_specs = ctx.meta
        g_ld, u_ld, d_ld = flags
        num_ld = len(ld_specs)

        saved = ctx.saved_tensors
        (
            hs_2d, routing_weights,
            W_gate, W_up, W_down,
            W_gate_eff, W_up_eff, W_down_eff,
            total_counts,
        ) = saved[:9]
        rest = saved[9:]
        B_list = rest[:num_ld]
        A_list = rest[num_ld : 2 * num_ld]
        flat_cn_th = rest[2 * num_ld :]

        T, D = hs_2d.shape
        dtype = W_gate.dtype
        in_dtype = hs_2d.dtype
        rw_dtype = routing_weights.dtype

        need_dhs = ctx.needs_input_grad[0]
        need_drw = ctx.needs_input_grad[1]
        d_hs = d_final_hs.new_zeros((T, D), dtype=in_dtype) if need_dhs else None
        d_rw = d_final_hs.new_zeros(routing_weights.shape, dtype=rw_dtype) if need_drw else None
        dW_gate = d_final_hs.new_zeros(W_gate.shape, dtype=dtype) if g_ld else None
        dW_up = d_final_hs.new_zeros(W_up.shape, dtype=dtype) if u_ld else None
        dW_down = d_final_hs.new_zeros(W_down.shape, dtype=dtype) if d_ld else None

        W_gate_T = W_gate_eff.transpose(1, 2)
        W_up_T = W_up_eff.transpose(1, 2)
        W_down_T = W_down_eff.transpose(1, 2)

        for t0, t1, C, token_idx, slot_idx, flat_pad_idx in ctx.chunk_meta:
            hs_c = hs_2d[t0:t1]
            rw_c = routing_weights[t0:t1]
            df_c = d_final_hs[t0:t1]

            x_perm = hs_c.index_select(0, token_idx).to(dtype)
            x_pad = hs_c.new_zeros((E * C, D), dtype=dtype)
            x_pad.index_copy_(0, flat_pad_idx, x_perm)
            x_pad = x_pad.view(E, C, D)

            g_pad = torch.bmm(x_pad, W_gate_T)
            u_pad = torch.bmm(x_pad, W_up_T)

            gf = g_pad.float()
            sig_g = torch.sigmoid(gf)
            silu_g = (gf * sig_g).to(dtype)
            dsilu_g = (sig_g * (1.0 + gf * (1.0 - sig_g))).to(dtype)
            h_pad = (silu_g * u_pad).to(dtype) if (need_drw or d_ld) else None

            w_perm = rw_c[token_idx, slot_idx]
            df_perm = df_c.index_select(0, token_idx)

            if need_drw:
                out_pad = torch.bmm(h_pad, W_down_T)
                out_perm = out_pad.view(E * C, D).index_select(0, flat_pad_idx)
                d_w_perm = (df_perm.float() * out_perm.float()).sum(dim=-1).to(rw_dtype)
                d_rw[t0:t1][token_idx, slot_idx] = d_w_perm

            dout_perm = (df_perm * w_perm.unsqueeze(-1)).to(dtype)
            dout_pad = df_c.new_zeros((E * C, D), dtype=dtype)
            dout_pad.index_copy_(0, flat_pad_idx, dout_perm)
            dout_pad = dout_pad.view(E, C, D)

            if d_ld:
                dW_down.add_(torch.bmm(dout_pad.transpose(1, 2), h_pad))

            dh_pad = torch.bmm(dout_pad, W_down_eff)
            dg_pad = (dh_pad * u_pad * dsilu_g).to(dtype)
            du_pad = (dh_pad * silu_g).to(dtype)

            if need_dhs:
                dx_pad = torch.bmm(dg_pad, W_gate_eff) + torch.bmm(du_pad, W_up_eff)
                dx_perm = dx_pad.view(E * C, D).index_select(0, flat_pad_idx).to(in_dtype)
                d_hs[t0:t1].index_add_(0, token_idx, dx_perm)

            if g_ld:
                dW_gate.add_(torch.bmm(dg_pad.transpose(1, 2), x_pad))
            if u_ld:
                dW_up.add_(torch.bmm(du_pad.transpose(1, 2), x_pad))

        unrouted = total_counts == 0
        has_unrouted = bool(unrouted.any().item())
        Ws = (W_gate, W_up, W_down)
        dWs = (dW_gate, dW_up, dW_down)
        d_ba = []
        ld_idx = 0
        for proj_idx, is_ld in enumerate(flags):
            if not is_ld:
                continue
            cn = flat_cn_th[2 * ld_idx]
            th = flat_cn_th[2 * ld_idx + 1]
            prior, alpha, reg_coef = ld_specs[ld_idx]
            dB, dA = _lowrank_grads_batched(
                Ws[proj_idx], dWs[proj_idx], B_list[ld_idx], A_list[ld_idx],
                cn, th, prior, alpha, reg_coef,
            )
            if has_unrouted:
                dB = dB.masked_fill(unrouted[:, None, None], 0.0)
                dA = dA.masked_fill(unrouted[:, None, None], 0.0)
            d_ba.extend(dB.unbind(0))
            d_ba.extend(dA.unbind(0))
            ld_idx += 1

        return (d_hs, d_rw, None, None, None, None, None, None, None, None, None, *d_ba)


def _is_peft_lora_linear(p: nn.Module) -> bool:
    return hasattr(p, "base_layer") and hasattr(p, "lora_A") and hasattr(p, "lora_B")


def _proj_compute_dtype(p: nn.Module) -> torch.dtype:
    if _is_peft_lora_linear(p):
        return p.base_layer.weight.dtype
    return p.weight.dtype


def _has_expert_hooks(exps: nn.ModuleList) -> bool:
    """Return True if any expert submodule has forward or forward-pre hooks registered."""
    if exps._forward_hooks or exps._forward_pre_hooks:
        return True
    for e in exps:
        if e._forward_hooks or e._forward_pre_hooks:
            return True
        for p in (e.gate_proj, e.up_proj, e.down_proj):
            if p._forward_hooks or p._forward_pre_hooks:
                return True
            if isinstance(p, SparseLinear):
                dm = p.diff_mask
                if dm._forward_hooks or dm._forward_pre_hooks:
                    return True
            elif _is_peft_lora_linear(p):
                base = p.base_layer
                if base._forward_hooks or base._forward_pre_hooks:
                    return True
                for container in (p.lora_A, p.lora_B):
                    if isinstance(container, nn.ModuleDict):
                        for sub in container.values():
                            if sub._forward_hooks or sub._forward_pre_hooks:
                                return True
    return False


def _is_supported_proj(p: nn.Module) -> bool:
    if isinstance(p, nn.Linear):
        return p.bias is None and p.weight.dim() == 2
    if isinstance(p, (SparseLinear, LoRADeleteSparseLinear)):
        return p.weight.dim() == 2
    if _is_peft_lora_linear(p):
        base = p.base_layer
        if not isinstance(base, nn.Linear) or base.bias is not None or base.weight.dim() != 2:
            return False
        lora_variant = getattr(p, "lora_variant", {})
        lora_bias = getattr(p, "lora_bias", {})
        for adapter in p.active_adapters:
            if adapter not in p.lora_A:
                continue
            if adapter in lora_variant or lora_bias.get(adapter, False):
                return False
            if p.lora_A[adapter].weight.dim() != 2 or p.lora_B[adapter].weight.dim() != 2:
                return False
        return True
    return False


def _effective_expert_weights(
    experts: nn.ModuleList,
    attr: str,
    total_counts: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return stacked [E, d_out, d_in] effective weights in compute_dtype with PyTorch autograd preserved."""
    p0 = getattr(experts[0], attr)
    if compute_dtype is None:
        compute_dtype = _proj_compute_dtype(p0)

    # Eval-mode or unmasked LoRADeleteSparseLinear: compute all E hard masks in one batched lowrank_logits call.
    if isinstance(p0, LoRADeleteSparseLinear):
        W = torch.stack([getattr(e, attr).weight for e in experts], dim=0)
        if p0.sparse:
            B = torch.stack([getattr(e, attr).mask_B.float() for e in experts], dim=0)
            A = torch.stack([getattr(e, attr).mask_A.float() for e in experts], dim=0)
            cn = torch.stack([getattr(e, attr).col_norm.float() for e in experts], dim=0)
            th = torch.stack([getattr(e, attr).threshold.float() for e in experts], dim=0)
            mask = lowrank_logits(W, B, A, cn, th, p0.prior) > 0
            W = W * mask.to(W.dtype)
        return W if W.dtype == compute_dtype else W.to(compute_dtype)

    grad_enabled = torch.is_grad_enabled()
    active_token = sparse_linear._ACTIVE_FORWARD_TOKEN[0]
    routed_flags = (
        (total_counts > 0).tolist()
        if (isinstance(p0, SparseLinear) and p0.weight_norm and grad_enabled)
        else None
    )

    weights: list[torch.Tensor] = []
    for e_idx, exp in enumerate(experts):
        p = getattr(exp, attr)
        if isinstance(p, SparseLinear):
            w = p.weight
            if not p.sparse:
                w_eff = w
            elif p.step_weight is not None:
                w_eff = p.step_weight
            else:
                mask = p.diff_mask(target_shape=w.shape, dtype=w.dtype).to(w.device)
                if routed_flags is not None and p.weight_norm and routed_flags[e_idx]:
                    sparse_weight_norm = (w.detach() * mask).pow(2).sum()
                    if active_token is not None:
                        p.sparse_weight_norm = sparse_weight_norm
                        p._sparse_weight_norm_token = active_token
                w_eff = w * mask
            weights.append(w_eff if w_eff.dtype == compute_dtype else w_eff.to(compute_dtype))
        elif _is_peft_lora_linear(p):
            base_w = p.base_layer.weight
            w_eff = base_w if base_w.dtype == compute_dtype else base_w.to(compute_dtype)
            if not getattr(p, "disable_adapters", False) and not getattr(p, "merged", False):
                fan_in_fan_out = getattr(p, "fan_in_fan_out", False)
                for adapter in p.active_adapters:
                    if adapter not in p.lora_A:
                        continue
                    lora_A = p.lora_A[adapter].weight
                    lora_B = p.lora_B[adapter].weight
                    dropout = p.lora_dropout[adapter]
                    scaling = p.scaling[adapter]
                    a_eff = dropout(lora_A) if (p.training and not isinstance(dropout, nn.Identity)) else lora_A
                    delta = (lora_B @ a_eff) * scaling
                    if fan_in_fan_out:
                        delta = delta.T
                    if delta.dtype != compute_dtype:
                        w_eff = (w_eff.to(delta.dtype) + delta).to(compute_dtype)
                    else:
                        w_eff = w_eff + delta
            weights.append(w_eff)
        else:
            w = p.weight
            weights.append(w if w.dtype == compute_dtype else w.to(compute_dtype))

    return torch.stack(weights, dim=0)


def _prepare_peft_lora_specs(experts: nn.ModuleList, attr: str, compute_dtype: torch.dtype):
    """Stack base weights and per-adapter low-rank factors across all E experts for exact PEFT LoRA bmm."""
    p0 = getattr(experts[0], attr)
    W_base_T = torch.stack(
        [
            getattr(e, attr).base_layer.weight
            if getattr(e, attr).base_layer.weight.dtype == compute_dtype
            else getattr(e, attr).base_layer.weight.to(compute_dtype)
            for e in experts
        ],
        dim=0,
    ).transpose(1, 2)
    adapters = []
    if not getattr(p0, "disable_adapters", False) and not getattr(p0, "merged", False):
        fan_in_fan_out = getattr(p0, "fan_in_fan_out", False)
        for adapter in p0.active_adapters:
            if adapter not in p0.lora_A:
                continue
            A_s = torch.stack([getattr(e, attr).lora_A[adapter].weight for e in experts], dim=0)
            B_s = torch.stack([getattr(e, attr).lora_B[adapter].weight for e in experts], dim=0)
            if fan_in_fan_out:
                # transpose(B @ A) == A^T @ B^T; here x @ (delta)^T == x @ (B @ A)
                adapters.append((B_s, A_s, p0.lora_dropout[adapter], p0.scaling[adapter], p0.training, True))
            else:
                adapters.append(
                    (
                        A_s.transpose(1, 2),
                        B_s.transpose(1, 2),
                        p0.lora_dropout[adapter],
                        p0.scaling[adapter],
                        p0.training,
                        False,
                    )
                )
    return W_base_T, adapters


def _apply_proj_bmm(x_pad: torch.Tensor, proj_spec, compute_dtype: torch.dtype) -> torch.Tensor:
    """Apply either a [E, d_in, d_out] transposed weight tensor or a PEFT LoRA (W_base_T, adapters) spec."""
    if isinstance(proj_spec, torch.Tensor):
        return torch.bmm(x_pad, proj_spec)
    W_base_T, adapters = proj_spec
    out = torch.bmm(x_pad, W_base_T)
    for m1, m2, dropout, scaling, training, fan_in_fan_out in adapters:
        x_ad = x_pad if x_pad.dtype == m1.dtype else x_pad.to(m1.dtype)
        if training and not isinstance(dropout, nn.Identity):
            x_ad = dropout(x_ad)
        if fan_in_fan_out:
            # x @ (B @ A): first (x @ B) then (@ A)
            delta_out = torch.bmm(torch.bmm(x_ad, m1), m2) * scaling
        else:
            delta_out = torch.bmm(torch.bmm(x_ad, m1), m2) * scaling
        out = out + delta_out
    return out if out.dtype == compute_dtype else out.to(compute_dtype)


def _batched_qwen3_moe_forward(self, hidden_states: torch.Tensor):
    exps = self.experts
    if not isinstance(exps, nn.ModuleList) or len(exps) == 0:
        return self._eager_moe_forward(hidden_states)
    e0 = exps[0]
    g0, u0, d0 = e0.gate_proj, e0.up_proj, e0.down_proj
    if (
        getattr(getattr(e0, "config", None), "hidden_act", "silu") != "silu"
        or not (_is_supported_proj(g0) and _is_supported_proj(u0) and _is_supported_proj(d0))
        or _has_expert_hooks(exps)
    ):
        return self._eager_moe_forward(hidden_states)

    ld_flags = tuple(
        isinstance(p, LoRADeleteSparseLinear) and p.sparse and p.training
        for p in (g0, u0, d0)
    )
    if any(ld_flags) and not all(
        is_ld or isinstance(p, nn.Linear)
        for is_ld, p in zip(ld_flags, (g0, u0, d0))
    ):
        return self._eager_moe_forward(hidden_states)

    global _BATCHED_FWD_CALLS
    _BATCHED_FWD_CALLS += 1

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hs_2d = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hs_2d)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hs_2d.dtype)

    T = hs_2d.shape[0]
    if T == 0:
        return hidden_states, router_logits

    E = self.num_experts

    # Path 1: LoRA Delete in training mode (custom autograd with batched Triton / PyTorch ops).
    if any(ld_flags):
        W_gate = torch.stack([e.gate_proj.weight for e in exps], dim=0)
        W_up = torch.stack([e.up_proj.weight for e in exps], dim=0)
        W_down = torch.stack([e.down_proj.weight for e in exps], dim=0)
        ld_specs = []
        cn_th_pairs = []
        ba_flat = []
        for attr, is_ld in zip(("gate_proj", "up_proj", "down_proj"), ld_flags):
            if is_ld:
                p0 = getattr(e0, attr)
                ld_specs.append((float(p0.prior), float(p0.alpha), float(p0.reg_coef)))
                cn = torch.stack([getattr(e, attr).col_norm.float() for e in exps], dim=0)
                th = torch.stack([getattr(e, attr).threshold.float() for e in exps], dim=0)
                cn_th_pairs.append((cn, th))
                ba_flat.extend(getattr(e, attr).mask_B for e in exps)
                ba_flat.extend(getattr(e, attr).mask_A for e in exps)
        final_hs = _BatchedMoELoRADeleteFn.apply(
            hs_2d, routing_weights, selected_experts, E,
            ld_flags, tuple(ld_specs), torch.is_grad_enabled(),
            W_gate, W_up, W_down, tuple(cn_th_pairs), *ba_flat,
        )
        return final_hs.view(batch_size, sequence_length, hidden_dim), router_logits

    # Path 2: Dense Delete (SparseLinear), FFT / MTH / Ref (nn.Linear), PEFT LoRA, or eval LoRA Delete.
    chunks = []
    total_counts = torch.zeros(E, dtype=torch.long, device=hs_2d.device)
    for t0 in range(0, T, _MOE_TOKEN_CHUNK):
        t1 = min(t0 + _MOE_TOKEN_CHUNK, T)
        C, counts_c, token_idx, slot_idx, flat_pad_idx = _pack_chunk_routing(selected_experts[t0:t1], E)
        total_counts.add_(counts_c)
        chunks.append((t0, t1, C, token_idx, slot_idx, flat_pad_idx))

    dtype = _proj_compute_dtype(g0)
    proj_specs = []
    for attr, p0 in (("gate_proj", g0), ("up_proj", u0), ("down_proj", d0)):
        if _is_peft_lora_linear(p0):
            proj_specs.append(_prepare_peft_lora_specs(exps, attr, dtype))
        else:
            proj_specs.append(
                _effective_expert_weights(exps, attr, total_counts, compute_dtype=dtype).transpose(1, 2)
            )
    gate_spec, up_spec, down_spec = proj_specs

    final_hs = torch.zeros_like(hs_2d)
    for t0, t1, C, token_idx, slot_idx, flat_pad_idx in chunks:
        hs_c = hs_2d[t0:t1]
        rw_c = routing_weights[t0:t1]
        x_perm = hs_c.index_select(0, token_idx).to(dtype)
        x_pad = hs_c.new_zeros((E * C, hidden_dim), dtype=dtype)
        x_pad.index_copy_(0, flat_pad_idx, x_perm)
        x_pad = x_pad.view(E, C, hidden_dim)

        g_pad = _apply_proj_bmm(x_pad, gate_spec, dtype)
        u_pad = _apply_proj_bmm(x_pad, up_spec, dtype)
        out_pad = _apply_proj_bmm((F.silu(g_pad) * u_pad).to(dtype), down_spec, dtype)

        out_perm = out_pad.view(E * C, hidden_dim).index_select(0, flat_pad_idx)
        w_perm = rw_c[token_idx, slot_idx]
        out_weighted = (out_perm * w_perm.unsqueeze(-1)).to(hs_2d.dtype)
        final_hs[t0:t1].index_add_(0, token_idx, out_weighted)

    return final_hs.view(batch_size, sequence_length, hidden_dim), router_logits


def patch_qwen3_moe_sparse_block() -> None:
    """Idempotently patch Qwen3MoeSparseMoeBlock.forward with the batched implementation."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError:
        return
    if not hasattr(Qwen3MoeSparseMoeBlock, "_eager_moe_forward"):
        Qwen3MoeSparseMoeBlock._eager_moe_forward = Qwen3MoeSparseMoeBlock.forward
        Qwen3MoeSparseMoeBlock.forward = _batched_qwen3_moe_forward


def unpatch_qwen3_moe_sparse_block() -> None:
    """Restore Qwen3MoeSparseMoeBlock.forward to its original eager implementation (useful in tests)."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError:
        return
    if hasattr(Qwen3MoeSparseMoeBlock, "_eager_moe_forward"):
        Qwen3MoeSparseMoeBlock.forward = Qwen3MoeSparseMoeBlock._eager_moe_forward
        delattr(Qwen3MoeSparseMoeBlock, "_eager_moe_forward")
