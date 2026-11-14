"""Deletion Tuning inside verl's GRPO trainer.

Deletion Tuning learns which pretrained weights to set to zero. Every weight of each target
linear layer (the MLP gate projection by default) gets one trainable logit. During training
the mask is ``sigmoid(logit * sharpness)``, where the sharpness rises linearly over training so the
soft mask hardens; the deployed mask keeps a weight exactly when its logit is positive. The
logits start from a Wanda importance prior, so almost every weight starts kept. Only the
logits train; every pretrained weight stays frozen.

With ``mask_type: lora_delete`` (learnable_sparsity/lora_delete_linear.py) the logits are instead G0 + H: a fixed
+-0.03 Wanda prior G0 plus a trainable low-rank H, and the regularizer's gradient is injected in backward.

This module connects the method to verl:

    apply_deletion_tuning(model, model_config)       wrap target layers before FSDP
    set_mask_schedule(model, total_steps)            sharpness schedule over the run's optimizer steps
    open_micro_batch(model, loss_scale_factor)       before each micro-batch forward that is backpropagated
    mask_reg_loss(model)                             regularizer added to the GRPO loss
    step_mask_schedules(model)                       after each optimizer step
    collect_merged_params(fsdp_module)               dense ``weight * hard_mask`` for vLLM
    save_mask_checkpoint(fsdp_module, save_dir)      ``save_dir/mask/`` at every saved step
"""

import math
import os
import time
from typing import Iterator

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer

import wanda_calibration
from learnable_sparsity import sparse_linear
from learnable_sparsity.lora_delete_linear import LoRADeleteSparseLinear, convert_to_lora_delete_model
from learnable_sparsity.moe_lora_delete import patch_qwen3_moe_sparse_block
from learnable_sparsity.sparse_linear import SparseLinear
from lora_deletion.lowrank import lowrank_logits

MASKED_LAYERS = (SparseLinear, LoRADeleteSparseLinear)
# Magnitude of the lora_delete Wanda prior G0: the dense prior's shift of 3 logit standard deviations of 0.01.
LORA_DELETE_PRIOR = 0.03
# Names of the trainable mask parameters; everything else is frozen.
_MASK_PARAM_NAMES = ("diff_mask.gate", "mask_B", "mask_A")


def _wanda_scores(
    model: nn.Module, model_config: DictConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return Wanda scores and activation column norms (on the CPU) on seeded, chat-templated calibration prompts.

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
    scores, norms = wanda_calibration.compute_wanda_scores(
        model=model, dataloader=batches, device=calib_device, num_samples=len(texts),
        target_modules=list(model_config.mask_target_modules), return_activation_norms=True,
    )
    model.to(device=device, dtype=dtype)
    return {k: v.cpu() for k, v in scores.items()}, {k: v.cpu() for k, v in norms.items()}


def _wanda_thresholds(
    model: nn.Module, model_config: DictConfig,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Stream-capable Wanda calibration returning per-layer thresholds and CPU column norms."""
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

    orig_dtype = next(model.parameters()).dtype
    run_dtype = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype
    if orig_dtype != run_dtype:
        model.to(dtype=run_dtype)
    calib_device = (
        torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        if torch.cuda.is_available()
        else next(model.parameters()).device
    )
    print(f"[Deletion Tuning] Wanda calibration (stream-capable) on {len(texts)} prompts, max_length={max_length}, "
          f"batch_size={batch_size}")
    try:
        return wanda_calibration.compute_wanda_thresholds_and_norms(
            model=model, dataloader=batches, device=calib_device, num_samples=len(texts),
            target_modules=list(model_config.mask_target_modules), init_sparsity=model_config.mask_init_sparsity,
        )
    finally:
        if orig_dtype != run_dtype:
            model.to(dtype=orig_dtype)


def _init_lora_delete_from_wanda(
    model: nn.Module, wanda_scores: dict[str, torch.Tensor], activation_norms: dict[str, torch.Tensor],
    init_sparsity: float,
) -> None:
    """Install the Wanda prior G0 of every LoRADeleteSparseLinear: col_norm = a, threshold = the dense prior's.

    The threshold is the same k-th smallest score as wanda_prior_deletions, so G0 = +-LORA_DELETE_PRIOR shifts down
    the exact same weights as the dense prior while H = (B @ A) / sqrt(r) provides low-rank logit noise of entrywise
    std ``mask_init_std`` (default 0.01, matching Paper Eq. 8; ``mask_init_std = 0`` gives exact H = 0).
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRADeleteSparseLinear) and name in wanda_scores:
            with torch.no_grad():
                module.col_norm.copy_(activation_norms[name])
                module.threshold.fill_(wanda_calibration.wanda_threshold(
                    scores=wanda_scores[name], init_sparsity=init_sparsity))


def apply_deletion_tuning(model: nn.Module, model_config: DictConfig) -> nn.Module:
    """Mask the target linear layers, initialize the logits from the Wanda prior, and freeze everything else.

    ``model_config.mask_type`` is ``dense`` (one logit per weight) or ``lora_delete`` (G0 + H logits with
    ``mask_rank``).
    """
    patch_qwen3_moe_sparse_block()
    model.enable_input_require_grads()
    prior_strength = model_config.mask_prior_strength
    assert prior_strength == 0 or model_config.mask_calib_parquet_files is not None, (
        "mask_prior_strength > 0 needs mask_calib_parquet_files (the training parquets) to calibrate the Wanda prior"
    )
    orig_dtype = next(model.parameters()).dtype
    run_dtype = torch.bfloat16 if orig_dtype == torch.float32 else orig_dtype
    calib_device = (
        torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        if torch.cuda.is_available()
        else next(model.parameters()).device
    )
    # Only rank 0 holds the pretrained weights here; FSDP broadcasts its initialized logits to the other ranks.
    calibrate = prior_strength > 0 and dist.get_rank() == 0
    if calibrate:
        t_wanda = time.time()
        wcache = os.environ.get("WANDA_CACHE_PATH", "")
        loaded_cache = False
        if wcache and os.path.exists(wcache):
            obj = torch.load(wcache, map_location="cpu", weights_only=False)
            if obj.get("init_sparsity", model_config.mask_init_sparsity) == model_config.mask_init_sparsity:
                wanda_thresholds, activation_norms = obj["thresholds"], obj["norms"]
                loaded_cache = True
                print(f"[Deletion Tuning] Loaded cached Wanda thresholds from {wcache}")
        if not loaded_cache:
            wanda_thresholds, activation_norms = _wanda_thresholds(model=model, model_config=model_config)
            if wcache:
                os.makedirs(os.path.dirname(os.path.abspath(wcache)), exist_ok=True)
                torch.save(
                    {
                        "thresholds": wanda_thresholds,
                        "norms": activation_norms,
                        "init_sparsity": float(model_config.mask_init_sparsity),
                    },
                    wcache,
                )
                print(f"[Deletion Tuning] Saved Wanda thresholds cache to {wcache}")
        print(f"[Deletion Tuning] Wanda calibration finished in {time.time() - t_wanda:.2f}s "
              f"({len(activation_norms)} target layers calibrated)")

    sharpness = [model_config.mask_sharpness_start, model_config.mask_sharpness_end]
    # Optional run-level seed for the gate noise init; unset keeps the paper's default behavior.
    seed_env = os.environ.get("MASK_INIT_SEED")
    if seed_env is not None:
        torch.manual_seed(int(seed_env))
    target_modules = list(model_config.mask_target_modules)
    # set_mask_schedule sets the schedule length from the trainer's step count.
    if model_config.mask_type == "dense":
        sparse_linear.convert_to_sparse_model(
            model=model, target_modules=target_modules, sharpness=sharpness, total_steps=1,
        )
        if calibrate:
            wanda_calibration.init_masks_from_wanda_thresholds(
                model=model,
                thresholds=wanda_thresholds,
                activation_norms=activation_norms,
                prior_strength=prior_strength,
                score_device=calib_device,
                score_weight_dtype=run_dtype,
            )
    elif model_config.mask_type == "lora_delete":
        convert_to_lora_delete_model(
            model=model, target_modules=target_modules, rank=model_config.mask_rank,
            prior=LORA_DELETE_PRIOR, sharpness=sharpness, total_steps=1,
            init_std=model_config.mask_init_std,
        )
        if calibrate:
            for name, mod in model.named_modules():
                if isinstance(mod, LoRADeleteSparseLinear) and name in wanda_thresholds:
                    with torch.no_grad():
                        mod.col_norm.copy_(activation_norms[name])
                        mod.threshold.fill_(wanda_thresholds[name])
    else:
        raise ValueError(f"mask_type must be dense or lora_delete, got {model_config.mask_type!r}")

    for name, param in model.named_parameters():
        if not any(k in name for k in _MASK_PARAM_NAMES):
            param.requires_grad_(False)
    # The actor reads this to weight the regularizer in the policy loss.
    model.mask_weight_reg = model_config.mask_weight_reg
    sparse_linear.set_weight_norm(model=model, enabled=model.mask_weight_reg > 0)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank_info = (
        f"mask_rank={model_config.mask_rank}, mask_init_std={model_config.mask_init_std}, "
        if model_config.mask_type == "lora_delete" else ""
    )
    print(f"[Deletion Tuning] mask_type={model_config.mask_type}, {rank_info}target_modules={target_modules}, "
          f"init_sparsity={model_config.mask_init_sparsity}, prior_strength={prior_strength}, sharpness={sharpness}, "
          f"weight_reg={model.mask_weight_reg} (decays linearly to 0); "
          f"trainable mask logits: {trainable:,} of {total:,} parameters")
    return model


def _schedule(module: nn.Module) -> nn.Module:
    """Return the object holding a masked layer's sharpness schedule (step, progress, total_steps)."""
    return module.diff_mask if isinstance(module, SparseLinear) else module


def set_mask_schedule(model: nn.Module, total_steps: int) -> None:
    """Set the number of optimizer steps over which every mask's sharpness rises."""
    assert total_steps > 0, f"the mask schedule needs a positive number of steps, got {total_steps}"
    for module in model.modules():
        if isinstance(module, MASKED_LAYERS):
            _schedule(module).total_steps = total_steps


def step_mask_schedules(model: nn.Module) -> None:
    """Advance the sharpness schedule of every mask by one optimizer step."""
    for module in model.modules():
        if isinstance(module, MASKED_LAYERS):
            _schedule(module).step()


def open_micro_batch(model: nn.Module, loss_scale_factor: float) -> None:
    """Prepare the regularizer of a micro-batch forward that will be backpropagated with ``loss_scale_factor``.

    Opens the window in which dense layers store their ||W * mask||^2 terms for mask_reg_loss. Every
    LoRADeleteSparseLinear gets reg_coef = -lambda * (1 - progress) * loss_scale_factor, so the gradient its
    backward injects equals that of the term the actor's loss ``(policy_loss - mask_reg_loss) * loss_scale_factor``
    would hold for it; mask_reg_loss itself contributes 0 for these layers.
    """
    sparse_linear.open_sparse_weight_norm_window()
    for module in model.modules():
        if isinstance(module, LoRADeleteSparseLinear):
            module.reg_coef = -model.mask_weight_reg * (1.0 - module.progress) * loss_scale_factor


def mask_reg_loss(model: nn.Module) -> torch.Tensor:
    """Return lambda * sum ||W * mask||^2 over the terms stored in the open forward window, and close the window.

    lambda decays linearly from the model's mask_weight_reg to 0 over the mask schedule; the weights are detached.
    LoRADeleteSparseLinear layers add nothing here: their regularizer gradient is injected (see open_micro_batch).
    """
    active = sparse_linear._ACTIVE_FORWARD_TOKEN[0]
    progress = None
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if not isinstance(module, MASKED_LAYERS):
            continue
        if progress is None:
            progress = _schedule(module).progress
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


def _yield_bucket(bucket: list[tuple[str, SparseLinear]]) -> Iterator[tuple[str, SparseLinear, torch.Tensor]]:
    """Yield (name, module, full boolean hard mask) for a bucket of dense layers."""
    yield from ((n, m, mask) for (n, m), mask in zip(bucket, _gather_bucket_masks(bucket)))


@torch.no_grad()
def _lora_delete_hard_mask_from_sd(
    mod: LoRADeleteSparseLinear, raw_sd: dict[str, torch.Tensor], prefix: str,
) -> torch.Tensor:
    """Return ``mod``'s [d_out, d_in] bool hard mask from ``state_dict()`` entries (FSDP1, FSDP2 or unsharded)."""
    W, B, A = (
        t.full_tensor() if hasattr(t, "full_tensor") else t
        for t in (raw_sd[f"{prefix}.weight"], raw_sd[f"{prefix}.mask_B"], raw_sd[f"{prefix}.mask_A"])
    )
    col_norm = mod.col_norm.to(device=W.device, dtype=torch.float32)
    threshold = mod.threshold.to(device=W.device, dtype=torch.float32)
    return lowrank_logits(
        W, B.to(device=W.device, dtype=torch.float32), A.to(device=W.device, dtype=torch.float32),
        col_norm, threshold, mod.prior,
    ) > 0


def _iter_full_hard_masks(
    named_modules: list[tuple[str, nn.Module]],
    raw_sd: dict[str, torch.Tensor] | None = None,
) -> Iterator[tuple[str, nn.Module, torch.Tensor]]:
    """Yield (name, module, full boolean hard mask) per layer; every rank must consume all of it in the same order.

    Dense masks are all-gathered in buckets; a LoRADeleteSparseLinear's mask is built from its gathered W, B and A,
    one layer at a time (from ``raw_sd`` when provided so both FSDP1 SHARDED_STATE_DICT and FSDP2 work).
    """
    bucket: list[tuple[str, SparseLinear]] = []
    bucket_numel = 0
    for name, mod in named_modules:
        if isinstance(mod, LoRADeleteSparseLinear):
            if bucket:
                yield from _yield_bucket(bucket)
                bucket, bucket_numel = [], 0
            mask = (
                _lora_delete_hard_mask_from_sd(mod=mod, raw_sd=raw_sd, prefix=name)
                if raw_sd is not None
                else mod.full_hard_mask()
            )
            yield name, mod, mask
            continue
        numel = math.prod(mod.diff_mask.initial_mask_size)
        if bucket and bucket_numel + numel > _MASK_BUCKET_NUMEL:
            yield from _yield_bucket(bucket)
            bucket, bucket_numel = [], 0
        bucket.append((name, mod))
        bucket_numel += numel
    if bucket:
        yield from _yield_bucket(bucket)


def _clean_fsdp_name(name: str) -> str:
    return ".".join(p for p in name.split(".") if p and p != "_fsdp_wrapped_module")


def _top_level_fsdp_children(module: nn.Module, prefix: str = "") -> list[tuple[str, FSDP]]:
    out: list[tuple[str, FSDP]] = []
    unwrapped = getattr(module, "_fsdp_wrapped_module", module)
    for child_name, child in unwrapped.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, FSDP):
            out.append((child_prefix, child))
        else:
            out.extend(_top_level_fsdp_children(child, child_prefix))
    return out


def _direct_non_fsdp_named_params(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Parameter]]:
    out: list[tuple[str, nn.Parameter]] = []
    unwrapped = getattr(module, "_fsdp_wrapped_module", module)
    for pname, param in unwrapped.named_parameters(recurse=False):
        full = f"{prefix}.{pname}" if prefix else pname
        out.append((full, param))
    for child_name, child in unwrapped.named_children():
        if isinstance(child, FSDP):
            continue
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        out.extend(_direct_non_fsdp_named_params(child, child_prefix))
    return out


@torch.no_grad()
def _batched_lora_delete_masks(named_mods: list[tuple[str, LoRADeleteSparseLinear]]) -> dict[str, torch.Tensor]:
    """Compute boolean [d_out, d_in] hard masks for unsharded LoRADeleteSparseLinear modules, batched by shape."""
    groups: dict[tuple[int, int, int, float], list[tuple[str, LoRADeleteSparseLinear]]] = {}
    for name, mod in named_mods:
        key = (mod.out_features, mod.in_features, mod.rank, float(mod.prior))
        groups.setdefault(key, []).append((name, mod))
    masks_out: dict[str, torch.Tensor] = {}
    for (_, _, _, prior), grp in groups.items():
        dev = grp[0][1].weight.device
        W_s = torch.stack([m.weight.detach() for _, m in grp], dim=0)
        B_s = torch.stack([m.mask_B.detach().to(device=dev, dtype=torch.float32) for _, m in grp], dim=0)
        A_s = torch.stack([m.mask_A.detach().to(device=dev, dtype=torch.float32) for _, m in grp], dim=0)
        cn_s = torch.stack([m.col_norm.to(device=dev, dtype=torch.float32) for _, m in grp], dim=0)
        th_s = torch.stack([m.threshold.to(device=dev, dtype=torch.float32) for _, m in grp], dim=0)
        batch_masks = lowrank_logits(W_s, B_s, A_s, cn_s, th_s, prior) > 0
        for i, (name, _) in enumerate(grp):
            masks_out[name] = batch_masks[i]
    return masks_out


@torch.no_grad()
def _summoned_hard_masks(named_mods: list[tuple[str, nn.Module]]) -> dict[str, torch.Tensor]:
    """Compute boolean [d_out, d_in] hard masks for unsharded MASKED_LAYERS inside FSDP.summon_full_params."""
    masks_out: dict[str, torch.Tensor] = {}
    lora_mods: list[tuple[str, LoRADeleteSparseLinear]] = []
    for name, mod in named_mods:
        if isinstance(mod, LoRADeleteSparseLinear):
            lora_mods.append((name, mod))
        elif isinstance(mod, SparseLinear):
            gate = mod.diff_mask.gate
            local_gate = gate.to_local() if hasattr(gate, "to_local") else gate.detach()
            masks_out[name] = (local_gate > 0).view(mod.weight.shape)
    if lora_mods:
        masks_out.update(_batched_lora_delete_masks(lora_mods))
    return masks_out


def _has_masked_layers(fsdp_module: nn.Module) -> bool:
    return any(isinstance(mod, MASKED_LAYERS) for mod in fsdp_module.modules())


def _has_lora_delete_layers(fsdp_module: nn.Module) -> bool:
    return any(isinstance(mod, LoRADeleteSparseLinear) for mod in fsdp_module.modules())


def collect_merged_params(fsdp_module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the Hugging Face state dict for vLLM, with each masked weight replaced by ``weight * hard_mask``.

    All ranks must consume the iterator together and in full, since its gathers are collective. Masked weights are
    built one at a time (or one FSDP decoder layer at a time under FSDP1), as they are consumed.
    """
    if isinstance(fsdp_module, FSDP) and _has_masked_layers(fsdp_module):
        mp_dtype = getattr(getattr(fsdp_module, "mixed_precision", None), "param_dtype", None)
        # 1. Root non-FSDP parameters (e.g., embed_tokens, norm, lm_head)
        with FSDP.summon_full_params(fsdp_module, writeback=False, recurse=False):
            direct_params = dict(_direct_non_fsdp_named_params(fsdp_module))
            root_masked = [
                (_clean_fsdp_name(n), m)
                for n, m in fsdp_module._fsdp_wrapped_module.named_modules()
                if isinstance(m, MASKED_LAYERS) and f"{_clean_fsdp_name(n)}.weight" in direct_params
            ]
            root_masks = _summoned_hard_masks(root_masked) if root_masked else {}
            for pname, p in direct_params.items():
                if any(k in pname for k in _MASK_PARAM_NAMES):
                    continue
                mod_prefix = pname[:-7] if pname.endswith(".weight") else ""
                out_dt = mp_dtype or p.dtype
                if mod_prefix in root_masks:
                    w = p.detach() * root_masks[mod_prefix].to(dtype=p.dtype, device=p.device)
                    yield pname, w.to(dtype=out_dt).clone()
                else:
                    yield pname, p.detach().to(dtype=out_dt).clone()

        # 2. Each top-level FSDP child unit (e.g., each Qwen3MoeDecoderLayer with all 128 experts)
        for child_prefix, child_fsdp in _top_level_fsdp_children(fsdp_module):
            with FSDP.summon_full_params(child_fsdp, writeback=False, recurse=True):
                layer_masked: list[tuple[str, nn.Module]] = []
                for rel_name, mod in child_fsdp.named_modules():
                    if isinstance(mod, MASKED_LAYERS):
                        clean_rel = _clean_fsdp_name(rel_name)
                        full_mod_name = f"{child_prefix}.{clean_rel}" if clean_rel else child_prefix
                        layer_masked.append((full_mod_name, mod))
                layer_masks = _summoned_hard_masks(layer_masked) if layer_masked else {}
                for rel_pname, p in child_fsdp.named_parameters():
                    clean_rel_p = _clean_fsdp_name(rel_pname)
                    full_pname = f"{child_prefix}.{clean_rel_p}" if clean_rel_p else child_prefix
                    if any(k in full_pname for k in _MASK_PARAM_NAMES):
                        continue
                    mod_prefix = full_pname[:-7] if full_pname.endswith(".weight") else ""
                    out_dt = mp_dtype or p.dtype
                    if mod_prefix in layer_masks:
                        w = p.detach() * layer_masks[mod_prefix].to(dtype=p.dtype, device=p.device)
                        yield full_pname, w.to(dtype=out_dt).clone()
                    else:
                        yield full_pname, p.detach().to(dtype=out_dt).clone()
        return

    model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    sparse_modules = {name.replace("_fsdp_wrapped_module.", ""): mod
                      for name, mod in model.named_modules() if isinstance(mod, MASKED_LAYERS)}
    raw_sd = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in fsdp_module.state_dict().items()}

    for ckey, val in raw_sd.items():
        parts = ckey.split(".")
        if not any(".".join(parts[:i]) in sparse_modules for i in range(1, len(parts) + 1)):
            yield ckey, val.detach()

    for name, _, full_mask in _iter_full_hard_masks(list(sparse_modules.items()), raw_sd=raw_sd):
        with torch.no_grad():
            weight = raw_sd[f"{name}.weight"]
            if hasattr(weight, "full_tensor"):
                weight = weight.full_tensor()
            yield f"{name}.weight", weight.detach() * full_mask.to(dtype=weight.dtype, device=weight.device)


def save_mask_checkpoint(fsdp_module: nn.Module, save_dir: str) -> None:
    """Save the hard mask to ``save_dir/mask/``; every rank must call this, and rank 0 writes the files."""
    rank = dist.get_rank()
    if isinstance(fsdp_module, FSDP) and _has_lora_delete_layers(fsdp_module):
        mask_dict: dict[str, torch.Tensor] = {}
        with FSDP.summon_full_params(fsdp_module, writeback=False, recurse=False):
            direct_params = dict(_direct_non_fsdp_named_params(fsdp_module))
            root_masked = [
                (_clean_fsdp_name(n), m)
                for n, m in fsdp_module._fsdp_wrapped_module.named_modules()
                if isinstance(m, MASKED_LAYERS) and f"{_clean_fsdp_name(n)}.weight" in direct_params
            ]
            if root_masked:
                for mname, mval in _summoned_hard_masks(root_masked).items():
                    if rank == 0:
                        mask_dict[f"{mname}.weight"] = mval.cpu()
        for child_prefix, child_fsdp in _top_level_fsdp_children(fsdp_module):
            with FSDP.summon_full_params(child_fsdp, writeback=False, recurse=True):
                layer_masked = []
                for rel_name, mod in child_fsdp.named_modules():
                    if isinstance(mod, MASKED_LAYERS):
                        clean_rel = _clean_fsdp_name(rel_name)
                        full_mod_name = f"{child_prefix}.{clean_rel}" if clean_rel else child_prefix
                        layer_masked.append((full_mod_name, mod))
                if layer_masked:
                    for mname, mval in _summoned_hard_masks(layer_masked).items():
                        if rank == 0:
                            mask_dict[f"{mname}.weight"] = mval.cpu()
        if rank == 0:
            mask_save_dir = os.path.join(save_dir, "mask")
            sparse_linear.save_mask(mask=mask_dict, save_path=mask_save_dir)
            print(f"[Deletion Tuning] saved mask to {mask_save_dir}")
        return

    model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    named = [(name.replace("_fsdp_wrapped_module.", ""), mod) for name, mod in model.named_modules()
             if isinstance(mod, MASKED_LAYERS)]
    raw_sd = (
        {k.replace("_fsdp_wrapped_module.", ""): v for k, v in fsdp_module.state_dict().items()}
        if any(isinstance(mod, LoRADeleteSparseLinear) for _, mod in named)
        else None
    )
    mask_dict = {}
    for clean, _, full_mask in _iter_full_hard_masks(named, raw_sd=raw_sd):
        if rank == 0:
            mask_dict[f"{clean}.weight"] = full_mask.cpu()
    if rank == 0:
        mask_save_dir = os.path.join(save_dir, "mask")
        sparse_linear.save_mask(mask=mask_dict, save_path=mask_save_dir)
        print(f"[Deletion Tuning] saved mask to {mask_save_dir}")

