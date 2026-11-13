"""Random sparse fine-tuning baseline (MTH, Adewuyi et al., 2026), our re-implementation inside verl.

Full fine-tuning in which only a fixed random subset of the entries of every parameter trains. Each parameter W gets a
seeded mask M with entries drawn i.i.d. from Bernoulli(density), and before gradient clipping and every optimizer step
its gradient is replaced by ``grad * M``. With weight decay 0 (scripts/train_grpo.sh sets it), AdamW leaves the
entries outside M exactly unchanged: their gradients, and so their moment estimates and updates, stay 0.

Masks are drawn over each parameter's full shape, so they do not depend on the FSDP2 sharding; each rank keeps the
shard of its local gradient. Masking after the gradient reduction equals masking before it, since the reduction is a
sum over ranks.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor


def apply_random_sparse_ft(model: nn.Module, density: float, seed: int) -> None:
    """Draw every trainable parameter's mask (parameter i uses seed + i) and keep it on the model for mask_gradients."""
    assert 0 < density < 1, f"density must be in (0, 1), got {density}"
    masks = []
    trainable, kept = 0, 0
    for i, param in enumerate(p for p in model.parameters() if p.requires_grad):
        gen = torch.Generator(device=param.device).manual_seed(seed + i)
        full = torch.rand(param.shape, generator=gen, device=param.device) < density
        if isinstance(param, DTensor):
            local = distribute_tensor(full, param.device_mesh, param.placements, src_data_rank=None).to_local()
        else:
            assert not dist.is_initialized() or dist.get_world_size() == 1, "MTH needs FSDP2 (DTensor) parameters"
            local = full
        masks.append((param, local))
        trainable += param.numel()
        kept += int(full.sum())
    model.mth_grad_masks = masks
    print(f"[MTH] training {kept:,} of {trainable:,} parameter entries ({kept / trainable:.4%}, density {density})")


@torch.no_grad()
def mask_gradients(model: nn.Module) -> None:
    """Zero every gradient entry outside its parameter's mask."""
    for param, mask in model.mth_grad_masks:
        if param.grad is not None:
            grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            grad.mul_(mask)
