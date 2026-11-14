"""Wanda importance scores and the Wanda prior on Deletion Tuning mask logits.

compute_wanda_scores scores every weight of the target linear layers as |W_ij| * a_j (Sun et al., 2023,
arXiv:2306.11695), where a_j sums the L2 norm of the inputs to column j over the calibration batches.
wanda_prior_deletions marks the lowest-scoring fraction of weights (below wanda_threshold), and apply_wanda_prior
shifts the mask logits so those weights start deleted.
"""

from typing import Iterable

import torch
import torch.nn as nn

from learnable_sparsity.lora_delete_linear import LoRADeleteSparseLinear
from learnable_sparsity.sparse_linear import SparseLinear

_MASKED_LAYERS = (SparseLinear, LoRADeleteSparseLinear)


def compute_wanda_scores(
    model: nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_samples: int,
    target_modules: list[str],
    return_activation_norms: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ``{module_name: [out_features, in_features] Wanda scores}`` for every target linear layer.

    A target is a layer whose name contains one of ``target_modules``. The dataloader yields dicts with ``input_ids``
    and ``attention_mask`` until ``num_samples`` examples are seen. Masked layers run their unmasked weights here.
    With ``return_activation_norms``, also return ``{module_name: [in_features] a}``, the column norms a_j.
    """
    model.eval()
    sparse_layers = [m for m in model.modules() if isinstance(m, _MASKED_LAYERS)]
    for m in sparse_layers:
        m.sparse = False
    activation_norms: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, *_MASKED_LAYERS)) and any(t in name for t in target_modules):
            def _hook(mod, inp, _out, name=name):
                """Add this batch's per-column input norm to the running total."""
                # [B, seq_len, in_features], or [routed tokens, in_features] for a mixture-of-experts expert.
                x = inp[0].detach()
                norm = x.float().pow(2).sum(dim=tuple(range(x.dim() - 1))).sqrt()  # [in_features]
                if name in activation_norms:
                    activation_norms[name] += norm
                else:
                    activation_norms[name] = norm
            hooks.append(module.register_forward_hook(_hook))

    samples_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            samples_seen += input_ids.shape[0]
            if samples_seen >= num_samples:
                break
    for h in hooks:
        h.remove()
    for m in sparse_layers:
        m.sparse = True

    scores: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, *_MASKED_LAYERS)) and name in activation_norms:
            W = module.weight.data.float().abs()  # [out, in]
            scores[name] = W * activation_norms[name].unsqueeze(0)
    return (scores, activation_norms) if return_activation_norms else scores


def wanda_threshold(scores: torch.Tensor, init_sparsity: float) -> float:
    """Return the k-th smallest score, k = max(1, int(numel * init_sparsity)); scores below it are deleted."""
    flat_scores = scores.view(-1)
    k_prune = max(1, int(flat_scores.numel() * init_sparsity))
    return flat_scores.kthvalue(k_prune).values.item()


def wanda_prior_deletions(
    model: nn.Module,
    wanda_scores: dict[str, torch.Tensor],
    init_sparsity: float,
) -> dict[str, torch.Tensor]:
    """Return ``{name: [out * in] bool}`` per SparseLinear, True below its ``init_sparsity`` Wanda score quantile."""
    deletions: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        # A mixture-of-experts expert that no calibration token reached has no scores and gets no prior.
        if not isinstance(module, SparseLinear) or name not in wanda_scores:
            continue
        flat_scores = wanda_scores[name].to(module.weight.device).view(-1)  # [out * in]
        deletions[name] = flat_scores < wanda_threshold(scores=flat_scores, init_sparsity=init_sparsity)
    return deletions


def apply_wanda_prior(model: nn.Module, deletions: dict[str, torch.Tensor], prior_strength: float) -> None:
    """Shift each mask logit by ``prior_strength`` standard deviations, down for ``deletions`` and up otherwise."""
    for name, module in model.named_modules():
        if name not in deletions:
            continue
        prior_sign = torch.where(deletions[name].to(module.weight.device), -1.0, 1.0)  # 1 keeps, -1 deletes
        with torch.no_grad():
            gate = module.diff_mask.gate  # [out * in]
            gate_std = gate.std().item()
            gate.data += (prior_sign * gate_std * prior_strength).to(gate.dtype)


def init_masks_from_wanda(
    model: nn.Module,
    wanda_scores: dict[str, torch.Tensor],
    init_sparsity: float,
    prior_strength: float,
) -> None:
    """Apply the Wanda prior that deletes the ``init_sparsity`` lowest-scoring weights of every SparseLinear."""
    apply_wanda_prior(
        model=model,
        deletions=wanda_prior_deletions(model=model, wanda_scores=wanda_scores, init_sparsity=init_sparsity),
        prior_strength=prior_strength,
    )


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList | None:
    """Return the main transformer decoder ModuleList (e.g. model.model.layers), if present."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers") and len(mod) >= 2:
            return mod
    return None


def compute_wanda_thresholds_and_norms(
    model: nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_samples: int,
    target_modules: list[str],
    init_sparsity: float,
    max_gpu_fraction: float = 0.72,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Stream-capable Wanda calibration returning ``(thresholds, activation_norms_cpu)`` with O(1) extra GPU memory.

    When ``model`` exceeds ``max_gpu_fraction`` of free memory on ``device`` (e.g. 235B bf16 = 470 GB on a single
    288 GB GPU), early decoder layers stay resident on ``device`` while overflow decoder layers stream CPU -> GPU -> CPU
    per forward pass, and per-layer kthvalue thresholds are computed one layer at a time on ``device`` without
    materializing a 300+ GB float32 ``scores`` dictionary. Passing ``max_gpu_fraction <= 0.0`` forces layer-by-layer
    streaming (useful for testing).
    """
    orig_device = next(model.parameters()).device
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    free_bytes = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 0
    force_stream = max_gpu_fraction <= 0.0
    offload_hooks = []

    if not force_stream and (device.type != "cuda" or param_bytes <= int(max_gpu_fraction * free_bytes)):
        model.to(device=device)
    else:
        layers = _find_decoder_layers(model)
        assert layers is not None, "Model exceeds single-GPU memory but has no decoder ModuleList to stream"
        # Move non-decoder-layer direct parameters/buffers (embed_tokens, rotary_emb, final norm, lm_head) to device.
        layer_param_ids = {id(p) for layer in layers for p in layer.parameters()}
        layer_buf_ids = {id(b) for layer in layers for b in layer.buffers()}
        for sub in model.modules():
            for p in sub.parameters(recurse=False):
                if id(p) not in layer_param_ids:
                    p.data = p.data.to(device)
            for b in sub.buffers(recurse=False):
                if id(b) not in layer_buf_ids:
                    b.data = b.data.to(device)
        budget = 0 if force_stream else int(max_gpu_fraction * torch.cuda.mem_get_info(device)[0])
        used = 0
        resident_count = 0
        for layer in layers:
            lbytes = sum(p.numel() * p.element_size() for p in layer.parameters()) + sum(
                b.numel() * b.element_size() for b in layer.buffers()
            )
            if not force_stream and used + lbytes <= budget:
                layer.to(device)
                used += lbytes
                resident_count += 1
            else:
                layer.to("cpu")
                offload_hooks.append(layer.register_forward_pre_hook(lambda mod, _inp: (mod.to(device), None)[1]))
                offload_hooks.append(layer.register_forward_hook(lambda mod, _inp, _out: (mod.to("cpu"), None)[1]))
        print(
            f"[Deletion Tuning] Streamed Wanda placement: {resident_count}/{len(layers)} decoder layers resident on "
            f"{device} ({used / 1e9:.1f} GB), {len(layers) - resident_count} layers streamed from CPU"
        )

    model.eval()
    sparse_layers = [m for m in model.modules() if isinstance(m, _MASKED_LAYERS)]
    for m in sparse_layers:
        m.sparse = False
    activation_norms: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, *_MASKED_LAYERS)) and any(t in name for t in target_modules):
            def _hook(mod, inp, _out, name=name):
                x = inp[0].detach()
                norm = x.float().pow(2).sum(dim=tuple(range(x.dim() - 1))).sqrt()
                if name in activation_norms:
                    activation_norms[name] += norm
                else:
                    activation_norms[name] = norm
            hooks.append(module.register_forward_hook(_hook))

    samples_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            samples_seen += input_ids.shape[0]
            if samples_seen >= num_samples:
                break

    for h in hooks + offload_hooks:
        h.remove()
    for m in sparse_layers:
        m.sparse = True

    thresholds: dict[str, float] = {}
    norms_cpu: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, *_MASKED_LAYERS)) and name in activation_norms:
                norm_dev = activation_norms[name].to(device=device, dtype=torch.float32)
                W = module.weight.data.to(device=device, dtype=torch.float32).abs()
                scores_layer = W * norm_dev.unsqueeze(0)
                thresholds[name] = wanda_threshold(scores=scores_layer, init_sparsity=init_sparsity)
                norms_cpu[name] = norm_dev.cpu()
                del W, scores_layer
    activation_norms.clear()
    model.to(device=orig_device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return thresholds, norms_cpu


def init_masks_from_wanda_thresholds(
    model: nn.Module,
    thresholds: dict[str, float],
    activation_norms: dict[str, torch.Tensor],
    prior_strength: float,
    score_device: torch.device | None = None,
    score_weight_dtype: torch.dtype | None = None,
) -> None:
    """Apply the Wanda prior to every SparseLinear one layer at a time without materializing full wanda_scores."""
    for name, module in model.named_modules():
        if not isinstance(module, SparseLinear) or name not in thresholds:
            continue
        with torch.no_grad():
            dev = score_device if score_device is not None else module.weight.device
            w_src = module.weight.data
            if score_weight_dtype is not None and w_src.dtype != score_weight_dtype:
                w_src = w_src.to(device=dev, dtype=score_weight_dtype)
            W = w_src.to(device=dev, dtype=torch.float32).abs()
            norm = activation_norms[name].to(device=dev, dtype=torch.float32)
            del_mask = (W * norm.unsqueeze(0)).view(-1) < thresholds[name]
            gate = module.diff_mask.gate
            prior_sign = torch.where(del_mask.to(gate.device), -1.0, 1.0)
            gate_std = gate.std().item()
            gate.data += (prior_sign * gate_std * prior_strength).to(gate.dtype)
            del W, norm, del_mask, prior_sign
