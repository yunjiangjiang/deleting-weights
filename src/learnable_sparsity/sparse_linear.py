"""Masked linear layers for Deletion Tuning, and saving their masks.

A SparseLinear keeps the pretrained weight frozen and multiplies it by the mask of its
DifferentiableMask. Saved masks hold one bit per weight, bit-packed and zstd-compressed.
"""

import functools
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zstandard as zstd
from safetensors.torch import save_file
from torch.nn.parameter import Parameter

from .differentiable_mask import DifferentiableMask

# Token of the micro-batch forward whose regularizer terms are stored for the GRPO actor. None stores nothing
# (backward recomputation after the actor consumed the terms); trainers that never open a window keep
# _STORE_ALWAYS, under which every forward stores its term.
_STORE_ALWAYS = object()
_ACTIVE_FORWARD_TOKEN: list = [_STORE_ALWAYS]


def open_sparse_weight_norm_window() -> None:
    """Start a micro-batch forward whose regularizer terms are stored for the actor."""
    _ACTIVE_FORWARD_TOKEN[0] = object()


def close_sparse_weight_norm_window() -> None:
    """Stop storing regularizer terms; later writes (backward recomputation) are ignored."""
    _ACTIVE_FORWARD_TOKEN[0] = None


class SparseLinear(nn.Module):
    """Bias-free linear layer whose frozen weight is multiplied by a learnable mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sharpness: list[float],
        total_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        target_device: torch.device,
        weight: torch.Tensor | None = None,
    ):
        """Create the weight (filled in by convert_to_sparse_model) and its mask logits."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparse = True  # False applies the unmasked weight (Wanda calibration)
        # Whether forwards with gradients compute the regularizer term ||W * mask||^2 (see set_weight_norm).
        self.weight_norm = True
        # Masked weight shared by every forward of one optimizer step (see build_step_weights).
        self.step_weight = None
        self._step_sharpness = None  # sharpness the step weight was built at
        self._gate_guard = None  # hook that rejects logit gradients bypassing the step weight
        if weight is not None:
            self.weight = Parameter(weight.data, requires_grad=False)
        else:
            self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
            if not self.weight.is_meta:
                nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.diff_mask = DifferentiableMask(
            target_param=self.weight,
            sharpness=sharpness,
            total_steps=total_steps,
            target_device=target_device,
        )
        self.weight.requires_grad_(False)

    def step(self) -> None:
        """Advance the mask sharpness schedule by one optimizer step."""
        self.diff_mask.step()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the masked weight and cache this layer's sparse-weight regularizer term."""
        weight = self.weight
        if not self.sparse:
            return nn.functional.linear(input, weight)
        if self.step_weight is not None:
            return nn.functional.linear(input, self.step_weight)
        mask = self.diff_mask(target_shape=weight.shape, dtype=weight.dtype).to(weight.device)
        # The regularizer term is computed in every forward with gradients, checkpoint recomputation included, and
        # stored unless the window is closed.
        if self.weight_norm and torch.is_grad_enabled():
            sparse_weight_norm = (weight.detach() * mask).pow(2).sum()
            if _ACTIVE_FORWARD_TOKEN[0] is not None:
                self.sparse_weight_norm = sparse_weight_norm
                self._sparse_weight_norm_token = _ACTIVE_FORWARD_TOKEN[0]
        return nn.functional.linear(input, weight * mask)

    def extra_repr(self) -> str:
        """Describe the layer's shape."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


def build_step_weights(model: nn.Module) -> None:
    """Build every SparseLinear's masked weight once for the coming optimizer step, shared by all its forwards.

    In training mode the step weight is a leaf whose gradient _replay_mask_backward passes to the mask logits after
    each backward, giving the same logit gradients as rebuilding the mask in every forward; in eval mode it is a
    constant. Call drop_step_weights after the optimizer step. The regularizer term is not supported.
    """
    for m in model.modules():
        if not isinstance(m, SparseLinear) or not m.sparse:
            continue
        assert not m.weight_norm, (
            "step weights carry no sparse-weight regularizer term; turn it off with set_weight_norm"
        )
        assert m.step_weight is None, "step weights were built twice without drop_step_weights"
        dm = m.diff_mask
        with torch.no_grad():
            step_weight = m.weight * dm(target_shape=m.weight.shape, dtype=m.weight.dtype).to(m.weight.device)
        if dm.training:
            m._step_sharpness = dm.current_sharpness  # set by the training-mode mask above
            step_weight.requires_grad_(True)
            step_weight.register_post_accumulate_grad_hook(functools.partial(_replay_mask_backward, layer=m))
            m._gate_guard = dm.gate.register_hook(_reject_direct_gate_grad)
        m.step_weight = step_weight


def _replay_mask_backward(step_weight: torch.Tensor, layer: "SparseLinear") -> None:
    """Accumulate one micro-batch's step-weight gradient into the mask logits, as autograd would, and clear it."""
    dm = layer.diff_mask
    with torch.no_grad():
        grad_mask = step_weight.grad * layer.weight
        grad_soft = grad_mask.to(dm.gate.dtype).reshape(-1)
        soft = torch.sigmoid(dm.gate * layer._step_sharpness)
        grad_logit = torch.ops.aten.sigmoid_backward(grad_soft, soft) * layer._step_sharpness
        if dm.gate.grad is None:
            dm.gate.grad = grad_logit
        else:
            dm.gate.grad.add_(grad_logit)
    step_weight.grad = None


def _reject_direct_gate_grad(grad: torch.Tensor) -> torch.Tensor:
    """Fail when autograd reaches a mask logit directly while its layer has a step weight."""
    raise RuntimeError("a forward pass bypassed the step weight: mask logits got a gradient directly")


def drop_step_weights(model: nn.Module) -> None:
    """Drop every SparseLinear's step weight; call after the optimizer step."""
    for m in model.modules():
        if isinstance(m, SparseLinear):
            if m._gate_guard is not None:
                m._gate_guard.remove()
            m.step_weight = None
            m._step_sharpness = None
            m._gate_guard = None


def set_weight_norm(model: nn.Module, enabled: bool) -> None:
    """Turn the sparse-weight regularizer term of every SparseLinear on or off."""
    for m in model.modules():
        if isinstance(m, SparseLinear):
            m.weight_norm = enabled


def convert_to_sparse_model(
    model: nn.Module,
    target_modules: list[str],
    sharpness: list[float],
    total_steps: int,
) -> nn.Module:
    """Replace each nn.Linear whose name contains one of ``target_modules`` with a SparseLinear of its frozen weight."""
    converted_count = 0
    target_device = model.device

    def _convert_module(module: nn.Module, name: str) -> None:
        """Convert matching children of ``module`` in place, recursing into the others."""
        nonlocal converted_count
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child_module, nn.Linear) and any(target in full_name for target in target_modules):
                if child_module.bias is not None:
                    warnings.warn(f"convert_to_sparse_model: {full_name} has a bias, which SparseLinear drops.")
                sparse_layer = SparseLinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    sharpness=sharpness,
                    total_steps=total_steps,
                    device=child_module.weight.device,
                    dtype=child_module.weight.dtype,
                    target_device=target_device,
                    weight=child_module.weight,
                )
                setattr(module, child_name, sparse_layer)
                converted_count += 1
            else:
                _convert_module(child_module, full_name)

    _convert_module(model, "")
    print(f"Converted {converted_count} linear layers to masked layers")
    return model


def local_hard_mask(module: SparseLinear) -> torch.Tensor:
    """Return the layer's hard mask over this rank's local logits as a flat bool tensor (True keeps)."""
    gate = module.diff_mask.gate
    return (gate.to_local() if hasattr(gate, "to_local") else gate.data) > 0


def extract_all_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return ``{"<layer>.weight": bool mask}`` on the CPU per SparseLinear, leaving the model in eval mode."""
    model.eval()
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, SparseLinear):
            out[f"{name}.weight"] = local_hard_mask(module=m).view(m.diff_mask.initial_mask_size).cpu()
    return out


def compress_mask(mask_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    """Return bit-packed, zstd-compressed masks (``<name>.shape`` and ``<name>.packed``) and size statistics."""
    compressed = {}
    stats = {
        "format": "packbits_zstd",
        "total_params": 0,
        "zero_params": 0,
        "original_size_mb": 0,
        "compressed_size_mb": 0,
    }
    cctx = zstd.ZstdCompressor(level=3)
    items = mask_dict.items() if isinstance(mask_dict, dict) else mask_dict
    for name, mask in items:
        mask_bool = mask.to(torch.bool).cpu()
        packed = np.packbits(mask_bool.numpy().ravel())
        zstd_bytes = cctx.compress(packed.tobytes())
        shape_cpu = torch.tensor(mask.shape, dtype=torch.int64, device="cpu")
        compressed[f"{name}.shape"] = shape_cpu
        compressed[f"{name}.packed"] = torch.frombuffer(bytearray(zstd_bytes), dtype=torch.uint8).clone()
        stats["total_params"] += mask_bool.numel()
        stats["zero_params"] += int((~mask_bool).sum().item())
        stats["original_size_mb"] += mask_bool.element_size() * mask_bool.numel() / 1024 / 1024
        stats["compressed_size_mb"] += (shape_cpu.element_size() * shape_cpu.numel() + len(zstd_bytes)) / 1024 / 1024
    stats["sparsity_ratio"] = stats["zero_params"] / stats["total_params"]
    stats["compression_ratio"] = stats["original_size_mb"] / stats["compressed_size_mb"]
    return compressed, stats


def save_mask(mask: dict[str, torch.Tensor], save_path: str) -> None:
    """Save boolean masks to ``save_path/model.safetensors`` with statistics in ``config.json``."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    compressed, stats = compress_mask(mask)
    with open(save_path / "config.json", "w") as f:
        json.dump(stats, f, indent=2)
    save_file(compressed, save_path / "model.safetensors")


def step_all_sparse_modules(model: nn.Module) -> None:
    """Advance the sharpness schedule of every mask by one optimizer step."""
    for module in model.modules():
        if isinstance(module, SparseLinear):
            module.step()

