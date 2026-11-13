"""Deletion Tuning inside verl's GRPO trainer.

Deletion Tuning learns which pretrained weights to set to zero. Every weight of each target
linear layer (the MLP gate projection by default) gets one trainable logit. During training
the mask is ``sigmoid(logit * sharpness)``, where the sharpness rises linearly over training so the
soft mask hardens; the deployed mask keeps a weight exactly when its logit is positive. The
logits start from a Wanda importance prior, so almost every weight starts kept. Only the
logits train; every pretrained weight stays frozen.

This module connects the method to verl:

    apply_deletion_tuning(model, model_config)       wrap target layers before FSDP
    set_mask_schedule(model, total_steps)            sharpness schedule over the run's optimizer steps
    mask_reg_loss(model)                             regularizer added to the GRPO loss
    collect_merged_params(fsdp_module)               dense ``weight * hard_mask`` for vLLM
    save_mask_checkpoint(fsdp_module, save_dir)      ``save_dir/mask/`` at every saved step
"""

import math
import os
from typing import Iterator

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer

import wanda_calibration
from learnable_sparsity import sparse_linear
from learnable_sparsity.sparse_linear import SparseLinear


def _wanda_scores(model: nn.Module, model_config: DictConfig) -> dict[str, torch.Tensor]:
    """Return Wanda scores (on the CPU) on seeded, chat-templated prompts from the calibration parquets.

    The scores are computed on the local GPU, with fp32 models run in bf16.
    """
    df = pd.concat([pd.read_parquet(f) for f in model_config.mask_calib_parquet_files], ignore_index=True)
    if len(df) > model_config.mask_calib_num_samples:
        df = df.sample(n=model_config.mask_calib_num_samples, random_state=42).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config.path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True) for m in df["prompt"]]
    max_length = model_config.mask_calib_max_prompt_length
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    batch_size = 32 if max_length <= 1024 else 8
    batches = [{k: enc[k][i:i + batch_size] for k in ("input_ids", "attention_mask")}
               for i in range(0, len(texts), batch_size)]

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    calib_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    model.to(device=calib_device, dtype=torch.bfloat16 if dtype == torch.float32 else dtype)
    print(f"[Deletion Tuning] Wanda calibration on {len(texts)} prompts, max_length={max_length}, "
          f"batch_size={batch_size}")
    scores = wanda_calibration.compute_wanda_scores(
        model=model, dataloader=batches, device=calib_device, num_samples=len(texts),
        target_modules=list(model_config.mask_target_modules),
    )
    model.to(device=device, dtype=dtype)
    return {k: v.cpu() for k, v in scores.items()}


def apply_deletion_tuning(model: nn.Module, model_config: DictConfig) -> nn.Module:
    """Mask the target linear layers, initialize the logits from the Wanda prior, and freeze everything else."""
    model.enable_input_require_grads()
    prior_strength = model_config.mask_prior_strength
    assert prior_strength == 0 or model_config.mask_calib_parquet_files is not None, (
        "mask_prior_strength > 0 needs mask_calib_parquet_files (the training parquets) to calibrate the Wanda prior"
    )
    # Only rank 0 holds the pretrained weights here; FSDP broadcasts its initialized logits to the other ranks.
    calibrate = prior_strength > 0 and dist.get_rank() == 0
    if calibrate:
        wanda_scores = _wanda_scores(model=model, model_config=model_config)

    sharpness = [model_config.mask_sharpness_start, model_config.mask_sharpness_end]
    # Optional run-level seed for the gate noise init; unset keeps the paper's default behavior.
    seed_env = os.environ.get("MASK_INIT_SEED")
    if seed_env is not None:
        torch.manual_seed(int(seed_env))
    # set_mask_schedule sets the schedule length from the trainer's step count.
    sparse_linear.convert_to_sparse_model(
        model=model, target_modules=list(model_config.mask_target_modules), sharpness=sharpness, total_steps=1,
    )
    if calibrate:
        wanda_calibration.init_masks_from_wanda(
            model=model, wanda_scores=wanda_scores, init_sparsity=model_config.mask_init_sparsity,
            prior_strength=prior_strength,
        )

    for name, param in model.named_parameters():
        if "diff_mask.gate" not in name:
            param.requires_grad_(False)
    # The actor reads this to weight the regularizer in the policy loss.
    model.mask_weight_reg = model_config.mask_weight_reg
    sparse_linear.set_weight_norm(model=model, enabled=model.mask_weight_reg > 0)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Deletion Tuning] target_modules={list(model_config.mask_target_modules)}, "
          f"init_sparsity={model_config.mask_init_sparsity}, prior_strength={prior_strength}, sharpness={sharpness}, "
          f"weight_reg={model.mask_weight_reg} (decays linearly to 0); "
          f"trainable mask logits: {trainable:,} of {total:,} parameters")
    return model


def set_mask_schedule(model: nn.Module, total_steps: int) -> None:
    """Set the number of optimizer steps over which every mask's sharpness rises."""
    assert total_steps > 0, f"the mask schedule needs a positive number of steps, got {total_steps}"
    for module in model.modules():
        if isinstance(module, SparseLinear):
            module.diff_mask.total_steps = total_steps


def mask_reg_loss(model: nn.Module) -> torch.Tensor:
    """Return lambda * sum ||W * mask||^2 over the terms stored in the open forward window, and close the window.

    lambda decays linearly from the model's mask_weight_reg to 0 over the mask schedule; the weights are detached.
    """
    active = sparse_linear._ACTIVE_FORWARD_TOKEN[0]
    progress = None
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if not isinstance(module, SparseLinear):
            continue
        if progress is None:
            progress = module.diff_mask.progress
        if not hasattr(module, "sparse_weight_norm"):
            continue
        term = module.sparse_weight_norm
        del module.sparse_weight_norm
        if module._sparse_weight_norm_token is active:
            total = total + term.float()
    sparse_linear.close_sparse_weight_norm_window()
    return model.mask_weight_reg * (1.0 - progress) * total


# Per-rank local mask shard sizes of each bucket, constant for a run.
_BUCKET_SHARD_SIZES: dict[tuple[str, ...], list[list[int]]] = {}
# Masked weights per all-gather (256 MiB of uint8 masks).
_MASK_BUCKET_NUMEL = 1 << 28


def _gather_bucket_masks(bucket: list[tuple[str, SparseLinear]]) -> list[torch.Tensor]:
    """Return the full 2D boolean hard masks of a bucket of layers, gathered across ranks with one all-gather."""
    shapes = [tuple(m.diff_mask.initial_mask_size) for _, m in bucket]
    local = [sparse_linear.local_hard_mask(module=m).to(torch.uint8) for _, m in bucket]
    if dist.get_world_size() == 1:
        return [t.view(*shape).bool() for t, shape in zip(local, shapes)]

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    local = [t.to(device) for t in local]
    world_size = dist.get_world_size()
    key = tuple(name for name, _ in bucket)
    if key not in _BUCKET_SHARD_SIZES:
        for name, m in bucket:
            placements = getattr(m.diff_mask.gate, "placements", ())
            if any(p.is_shard() and p.dim != 0 for p in placements):
                raise RuntimeError(f"mask logits of {name} are sharded as {placements}; only dim 0 is supported")
        sizes = torch.tensor([t.numel() for t in local], dtype=torch.long, device=device)
        gathered_sizes = [torch.empty_like(sizes) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, sizes)
        _BUCKET_SHARD_SIZES[key] = [g.tolist() for g in gathered_sizes]
    sizes = _BUCKET_SHARD_SIZES[key]  # sizes[rank][layer]
    if sizes[dist.get_rank()] != [t.numel() for t in local]:
        raise RuntimeError(f"mask shard sizes changed since they were cached for the bucket starting at {key[0]}")

    max_total = max(max(sum(rank_sizes) for rank_sizes in sizes), 1)
    send = torch.zeros(max_total, dtype=torch.uint8, device=device)
    n_local = sum(t.numel() for t in local)
    if n_local:
        send[:n_local] = torch.cat(local)
    recv = [torch.empty(max_total, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(recv, send)

    offsets = [0] * world_size
    masks = []
    for i, ((name, _), shape) in enumerate(zip(bucket, shapes)):
        expected = math.prod(shape)
        pieces = []
        for r in range(world_size):
            pieces.append(recv[r][offsets[r]:offsets[r] + sizes[r][i]])
            offsets[r] += sizes[r][i]
        # Each rank holds either the whole mask or a contiguous piece of it, in rank order.
        full_ranks = [r for r in range(world_size) if sizes[r][i] == expected]
        if full_ranks:
            full = pieces[full_ranks[0]]
        else:
            if sum(sizes[r][i] for r in range(world_size)) != expected:
                raise RuntimeError(f"mask shard sizes for {name} sum to {sum(sizes[r][i] for r in range(world_size))}, "
                                   f"expected {expected} (per-rank sizes {[sizes[r][i] for r in range(world_size)]})")
            full = torch.cat(pieces)
        masks.append(full.view(*shape).bool())
    return masks


def _iter_full_hard_masks(
    named_modules: list[tuple[str, SparseLinear]],
) -> Iterator[tuple[str, SparseLinear, torch.Tensor]]:
    """Yield (name, module, full boolean hard mask) per layer; every rank must consume all of it in the same order."""
    bucket: list[tuple[str, SparseLinear]] = []
    bucket_numel = 0
    for name, mod in named_modules:
        numel = math.prod(mod.diff_mask.initial_mask_size)
        if bucket and bucket_numel + numel > _MASK_BUCKET_NUMEL:
            yield from ((n, m, mask) for (n, m), mask in zip(bucket, _gather_bucket_masks(bucket)))
            bucket, bucket_numel = [], 0
        bucket.append((name, mod))
        bucket_numel += numel
    if bucket:
        yield from ((n, m, mask) for (n, m), mask in zip(bucket, _gather_bucket_masks(bucket)))


def collect_merged_params(fsdp_module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the Hugging Face state dict for vLLM, with each masked weight replaced by ``weight * hard_mask``.

    All ranks must consume the iterator together and in full, since its gathers are collective. Masked weights are
    built one at a time, as they are consumed.
    """
    model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    sparse_modules = {name.replace("_fsdp_wrapped_module.", ""): mod
                      for name, mod in model.named_modules() if isinstance(mod, SparseLinear)}
    raw_sd = fsdp_module.state_dict()

    for key, val in raw_sd.items():
        ckey = key.replace("_fsdp_wrapped_module.", "")
        parts = ckey.split(".")
        if not any(".".join(parts[:i]) in sparse_modules for i in range(1, len(parts) + 1)):
            yield ckey, val.detach()

    for name, _, full_mask in _iter_full_hard_masks(list(sparse_modules.items())):
        with torch.no_grad():
            weight = raw_sd[f"{name}.weight"]
            if hasattr(weight, "full_tensor"):
                weight = weight.full_tensor()
            yield f"{name}.weight", weight.detach() * full_mask.to(dtype=weight.dtype, device=weight.device)


def save_mask_checkpoint(fsdp_module: nn.Module, save_dir: str) -> None:
    """Save the hard mask to ``save_dir/mask/``; every rank must call this, and rank 0 writes the files."""
    rank = dist.get_rank()
    model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    named = [(name.replace("_fsdp_wrapped_module.", ""), mod) for name, mod in model.named_modules()
             if isinstance(mod, SparseLinear)]
    mask_dict: dict[str, torch.Tensor] = {}
    for clean, _, full_mask in _iter_full_hard_masks(named):
        if rank == 0:
            mask_dict[f"{clean}.weight"] = full_mask.cpu()
    if rank == 0:
        mask_save_dir = os.path.join(save_dir, "mask")
        sparse_linear.save_mask(mask=mask_dict, save_path=mask_save_dir)
        print(f"[Deletion Tuning] saved mask to {mask_save_dir}")

