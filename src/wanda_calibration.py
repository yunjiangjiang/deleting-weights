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
