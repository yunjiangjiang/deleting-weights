"""Learnable Deletion Tuning mask over one weight matrix."""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter


class DifferentiableMask(nn.Module):
    """One trainable logit per weight, relaxed to a soft mask during training.

    In training mode the mask is ``sigmoid(gate * sharpness)``, where the sharpness rises
    linearly from ``sharpness[0]`` to ``sharpness[1]`` over ``total_steps`` calls to ``step()``. In
    eval mode the mask is binary: a weight is kept exactly when its logit is positive.
    """

    def __init__(
        self,
        target_param: torch.Tensor,
        sharpness: list[float],
        total_steps: int,
        target_device: torch.device,
    ):
        """Create fp32 logits for ``target_param``, drawn from N(0, 0.01^2)."""
        super().__init__()
        if len(sharpness) != 2 or sharpness[0] <= 0 or sharpness[1] <= 0:
            raise ValueError(f"sharpness must be [start, end] with positive values, got {sharpness}")
        self.initial_mask_size = target_param.size()
        self.sharpness = sharpness
        self.total_steps = total_steps
        self.current_step = 0
        self.gate = Parameter(torch.empty(target_param.numel(), dtype=torch.float32, device=target_device))
        if not self.gate.is_meta:
            init.normal_(self.gate, std=0.01)
        # Current sharpness, updated by every training-mode forward.
        self.current_sharpness = sharpness[0]

    def step(self) -> None:
        """Advance the sharpness schedule by one optimizer step."""
        self.current_step += 1

    @property
    def progress(self) -> float:
        """Return the fraction of the schedule completed, in [0, 1]."""
        return min(self.current_step / self.total_steps, 1.0)

    def forward(self, target_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Return the soft (training) or hard (eval) mask, reshaped to ``target_shape`` and cast to ``dtype``."""
        if self.training:
            self.current_sharpness = self.sharpness[0] + (self.sharpness[1] - self.sharpness[0]) * self.progress
            return torch.sigmoid(self.gate * self.current_sharpness).reshape(target_shape).to(dtype)
        return (self.gate > 0).to(dtype).reshape(target_shape)
