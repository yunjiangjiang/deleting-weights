"""LoRA Deletion Tuning: a deletion mask whose logits are a Wanda prior plus a low-rank term.

For a frozen weight W [d_out, d_in] and rank r >= 1, with s = sqrt(3) (lora_deletion.lowrank):

    G0_ij  = +prior if |W_ij| * col_norm_j >= threshold else -prior     (Wanda prior, no stored logits)
    B [d_out, r] ~ Uniform(-init_std * s, +init_std * s)                (default init_std = 0.01; 0 gives B = 0)
    A [r, d_in]  ~ Uniform(-s, +s)                                      (unit variance)
    H      = (B @ A) / sqrt(r)
    m      = sigmoid(alpha * (G0 + H)) in training,  1[G0 + H > 0] in eval,     y = x @ (W * m)^T

with alpha on the dense mask's linear sharpness schedule. Default init_std = 0.01 matches Dense Deletion Tuning
(Paper Eq. 8, G ~ +-0.03 + noise with std = 0.01, lr = 3e-4, sharpness = [1000, 10000]), avoiding float32 sigmoid
saturation at alpha_start = 1000 while keeping Adam RMS(dH) ~ lr. Only B and A train, rank * (d_out + d_in) parameters
per layer instead of d_out * d_in logits. The regularizer c * ||W * m||_F^2 is not a loss term: its gradient is injected
in backward (lora_deletion.masked_delete) with c = ``reg_coef``, which the training loop sets before every forward that
is backpropagated.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from lora_deletion.lowrank import lowrank_logits
from lora_deletion.masked_delete import masked_delete
from torch.nn.parameter import Parameter


class LoRADeleteSparseLinear(nn.Module):
    """Bias-free linear layer whose frozen weight is multiplied by a mask with low-rank logits."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        prior: float,
        sharpness: list[float],
        total_steps: int,
        init_std: float = 0.01,
    ):
        """Take ``base``'s weight (frozen), with Uniform(+-sqrt(3)) A and init_std-scaled B in fp32.

        With default ``init_std = 0.01``, ``H = (B @ A) / sqrt(rank)`` has entrywise std 0.01 (matching Dense Deletion
        Tuning's logit noise so ``alpha_start = 1000`` has non-zero float32 gradients); ``init_std = 0.0`` gives exact
        ``B = 0`` (``H = 0``). Until col_norm and threshold are set (verl_adapters installs the Wanda prior),
        col_norm = 1 and threshold = 0 give G0 = +prior (keep every weight up to H).
        """
        super().__init__()
        d_out, d_in = base.out_features, base.in_features
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if not init_std >= 0.0:
            raise ValueError(f"init_std must be >= 0, got {init_std}")
        if len(sharpness) != 2 or sharpness[0] <= 0 or sharpness[1] <= 0:
            raise ValueError(f"sharpness must be [start, end] with positive values, got {sharpness}")
        if base.bias is not None:
            warnings.warn("LoRADeleteSparseLinear drops the bias of the layer it replaces.")
        self.in_features, self.out_features = d_in, d_out
        self.rank = rank
        self.prior = prior
        self.init_std = init_std
        self.sharpness = sharpness
        self.total_steps = total_steps
        self.current_step = 0
        self.sparse = True  # False applies the unmasked weight (Wanda calibration)
        # Coefficient c of the injected regularizer gradient d(c * ||W * m||^2) / d(B, A).
        self.reg_coef = 0.0
        device = base.weight.device
        s = math.sqrt(3.0)
        self.weight = Parameter(base.weight.data, requires_grad=False)
        if init_std == 0.0:
            self.mask_B = Parameter(torch.zeros(d_out, rank, dtype=torch.float32, device=device))
        else:
            self.mask_B = Parameter(torch.empty(d_out, rank, dtype=torch.float32, device=device))
        self.mask_A = Parameter(torch.empty(rank, d_in, dtype=torch.float32, device=device))
        if not self.mask_A.is_meta:
            init.uniform_(self.mask_A, -s, s)
        if init_std != 0.0 and not self.mask_B.is_meta:
            init.uniform_(self.mask_B, -init_std * s, init_std * s)
        self.register_buffer("col_norm", torch.ones(d_in, dtype=torch.float32, device=device))
        self.register_buffer("threshold", torch.zeros((), dtype=torch.float32, device=device))

    def step(self) -> None:
        """Advance the sharpness schedule by one optimizer step."""
        self.current_step += 1

    @property
    def progress(self) -> float:
        """Return the fraction of the schedule completed, in [0, 1]."""
        return min(self.current_step / self.total_steps, 1.0)

    @property
    def alpha(self) -> float:
        """Return the current sharpness, rising linearly from ``sharpness[0]`` to ``sharpness[1]``."""
        return self.sharpness[0] + (self.sharpness[1] - self.sharpness[0]) * self.progress

    @torch.no_grad()
    def full_hard_mask(self) -> torch.Tensor:
        """Return the [d_out, d_in] bool hard mask 1[G0 + H > 0] (True keeps), with H in fp32.

        Sharded (DTensor) parameters are gathered with full_tensor(), a collective every rank must join.
        """
        W, B, A = (p.full_tensor() if hasattr(p, "full_tensor") else p for p in (self.weight, self.mask_B, self.mask_A))
        if W.shape != (self.out_features, self.in_features):
            raise RuntimeError(f"weight has local shape {tuple(W.shape)}: lora_delete needs FSDP2 or unsharded weights")
        col_norm = self.col_norm.to(device=W.device, dtype=torch.float32)
        threshold = self.threshold.to(device=W.device, dtype=torch.float32)
        return lowrank_logits(
            W, B.to(device=W.device, dtype=torch.float32), A.to(device=W.device, dtype=torch.float32),
            col_norm, threshold, self.prior,
        ) > 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the soft-masked weight with the injected regularizer gradient (training) or the hard mask (eval)."""
        if not self.sparse:
            return F.linear(input, self.weight)
        if not self.training:
            return F.linear(input, self.weight * self.full_hard_mask().to(self.weight.dtype))
        # B and A arrive in bf16 under FSDP mixed precision; the kernels multiply them in their load dtype, so they are
        # upcast here (rank * (d_out + d_in) entries) to form H, and the logits, in fp32.
        return masked_delete(
            input, self.weight, self.mask_B.float(), self.mask_A.float(), self.col_norm, self.threshold, self.prior,
            self.alpha, self.reg_coef, backend="triton" if self.weight.is_cuda else "torch",
        )

    def extra_repr(self) -> str:
        """Describe the layer's shape and mask parameterization."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias=False, "
                f"rank={self.rank}, prior={self.prior}, init_std={self.init_std}")


def convert_to_lora_delete_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    prior: float,
    sharpness: list[float],
    total_steps: int,
    init_std: float = 0.01,
) -> nn.Module:
    """Replace each nn.Linear whose name contains one of ``target_modules`` with a LoRADeleteSparseLinear."""
    targets = [name for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and any(t in name for t in target_modules)]
    for name in targets:
        parent, _, child = name.rpartition(".")
        parent_module = model.get_submodule(parent)
        layer = LoRADeleteSparseLinear(
            base=getattr(parent_module, child), rank=rank, prior=prior, sharpness=sharpness, total_steps=total_steps,
            init_std=init_std,
        )
        setattr(parent_module, child, layer)
    print(f"Converted {len(targets)} linear layers to LoRA-deletion masked layers")
    return model
