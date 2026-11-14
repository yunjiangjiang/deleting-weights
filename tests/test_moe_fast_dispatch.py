"""Unit tests for batched Qwen3MoeSparseMoeBlock dispatch across all 4 methods + FSDP1 merge/save.

Covers:
  1. lora_delete (LoRADeleteSparseLinear with reg_coef injection; gate_proj and all-3 expert projs)
  2. dense (SparseLinear with weight_norm=True and mask_reg_loss)
  3. fft (trainable nn.Linear across the full model)
  4. lora (PEFT LoraConfig across all 7 attention + MoE projections, including non-zero lora_B)

Verifies forward + backward numerical equivalence against unpatched eager Qwen3MoeSparseMoeBlock,
non-None zero gradients on unrouted experts (FSDP2 post-backward safety), multi-chunk token packing,
eval mode, forward-hook fallback, and (on CUDA) bf16 Triton/TensorCore execution + FSDP1 summon merge.
"""

from __future__ import annotations

import copy
import functools

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock

import apply_mask
import verl_adapters
import wanda_calibration
from learnable_sparsity import moe_lora_delete, sparse_linear
from learnable_sparsity.lora_delete_linear import LoRADeleteSparseLinear, convert_to_lora_delete_model
from learnable_sparsity.moe_lora_delete import patch_qwen3_moe_sparse_block, unpatch_qwen3_moe_sparse_block
from learnable_sparsity.sparse_linear import SparseLinear


@pytest.fixture(scope="module", autouse=True)
def single_process_group():
    if torch.cuda.is_available() and dist.is_initialized() and dist.get_backend() != "nccl":
        dist.destroy_process_group()
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, init_method="tcp://127.0.0.1:29537", rank=0, world_size=1)
    yield


@pytest.fixture(autouse=True)
def restore_moe_patch():
    unpatch_qwen3_moe_sparse_block()
    yield
    unpatch_qwen3_moe_sparse_block()


def _make_tiny_qwen3_moe(seed: int = 0) -> Qwen3MoeForCausalLM:
    """Build a 2-layer, 8-expert Qwen3MoeForCausalLM whose router leaves 2 experts unrouted per layer."""
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder_sparse_step=1,
    )
    model = Qwen3MoeForCausalLM(config)
    # Antithetic router rows guarantee >=2 positive logits per token while zeroed rows stay at 0.0 -> 0 routed tokens.
    with torch.no_grad():
        g0 = model.model.layers[0].mlp.gate.weight
        g0[0].normal_(std=5.0)
        g0[1].copy_(-g0[0])
        g0[2].normal_(std=5.0)
        g0[3].copy_(-g0[2])
        g0[6:].zero_()

        g1 = model.model.layers[1].mlp.gate.weight
        g1[0].normal_(std=5.0)
        g1[2].copy_(-g1[0])
        g1[3].normal_(std=5.0)
        g1[4].copy_(-g1[3])
        g1[[1, 5]].zero_()
    return model


def _prepare_method_model(
    base_model: Qwen3MoeForCausalLM,
    method: str,
    dtype: torch.dtype,
    device: torch.device,
    lora_delete_targets: list[str] | None = None,
) -> nn.Module:
    model = copy.deepcopy(base_model)
    torch.manual_seed(123)
    if method == "lora_delete":
        targets = lora_delete_targets or ["gate_proj"]
        convert_to_lora_delete_model(
            model=model,
            target_modules=targets,
            rank=8,
            prior=0.03,
            sharpness=[1000.0, 10000.0],
            total_steps=10,
            init_std=0.01,
        )
        for name, mod in model.named_modules():
            if isinstance(mod, LoRADeleteSparseLinear):
                mod.current_step = 3
                with torch.no_grad():
                    col_norm = torch.rand(mod.in_features, dtype=torch.float32) + 0.5
                    scores = mod.weight.data.float().abs() * col_norm.unsqueeze(0)
                    thr = wanda_calibration.wanda_threshold(scores=scores, init_sparsity=0.1)
                    mod.col_norm.copy_(col_norm)
                    mod.threshold.fill_(thr)
        for name, param in model.named_parameters():
            if not any(k in name for k in ("mask_B", "mask_A")):
                param.requires_grad_(False)
        model.mask_weight_reg = 0.05
        # Keep mask_B / mask_A in fp32 while casting base weights/buffers to target dtype.
        model.to(device=device, dtype=dtype)
        for mod in model.modules():
            if isinstance(mod, LoRADeleteSparseLinear):
                mod.mask_B.data = mod.mask_B.data.float()
                mod.mask_A.data = mod.mask_A.data.float()
                mod.col_norm.data = mod.col_norm.data.float()
                mod.threshold.data = mod.threshold.data.float()
        return model.train()

    if method == "dense":
        sparse_linear.convert_to_sparse_model(
            model=model,
            target_modules=["gate_proj"],
            sharpness=[1000.0, 10000.0],
            total_steps=10,
        )
        for mod in model.modules():
            if isinstance(mod, SparseLinear):
                mod.diff_mask.current_step = 3
        for name, param in model.named_parameters():
            if "diff_mask.gate" not in name:
                param.requires_grad_(False)
        model.mask_weight_reg = 0.05
        sparse_linear.set_weight_norm(model=model, enabled=True)
        model.to(device=device, dtype=dtype)
        for mod in model.modules():
            if isinstance(mod, SparseLinear):
                mod.diff_mask.gate.data = mod.diff_mask.gate.data.float()
        return model.train()

    if method == "fft":
        model.to(device=device, dtype=dtype)
        for param in model.parameters():
            param.requires_grad_(True)
        return model.train()

    if method == "lora":
        model.to(device=device, dtype=dtype)
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg).to(device=device)
        # Initialize lora_B non-zero so both lora_A and lora_B receive non-trivial gradients.
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.05)
        return peft_model.train()

    raise ValueError(f"Unknown method: {method}")


def _run_fwd_bwd(model: nn.Module, method: str, input_ids: torch.Tensor, scale: float = 0.75):
    if method in ("lora_delete", "dense"):
        verl_adapters.open_micro_batch(model=model, loss_scale_factor=scale)
    out = model(input_ids=input_ids, labels=input_ids)
    policy_loss = out.loss
    if method in ("lora_delete", "dense"):
        reg_loss = verl_adapters.mask_reg_loss(model=model)
        total_loss = (policy_loss - reg_loss) * scale
    else:
        reg_loss = torch.zeros((), device=input_ids.device)
        total_loss = policy_loss * scale
    total_loss.backward()
    return out.logits.detach(), reg_loss.detach()


def _compare_eager_vs_batched(
    method: str,
    dtype: torch.dtype,
    device: torch.device,
    rtol: float,
    atol: float,
    lora_delete_targets: list[str] | None = None,
    chunk_size: int | None = None,
) -> tuple[float, float, int]:
    base = _make_tiny_qwen3_moe(seed=7)
    eager_model = _prepare_method_model(base, method, dtype=dtype, device=device, lora_delete_targets=lora_delete_targets)
    batched_model = copy.deepcopy(eager_model)

    torch.manual_seed(99)
    input_ids = torch.randint(0, 128, (3, 7), device=device)

    unpatch_qwen3_moe_sparse_block()
    logits_eager, reg_eager = _run_fwd_bwd(eager_model, method, input_ids)

    orig_chunk = moe_lora_delete._MOE_TOKEN_CHUNK
    if chunk_size is not None:
        moe_lora_delete._MOE_TOKEN_CHUNK = chunk_size
    fwd_before = moe_lora_delete._BATCHED_FWD_CALLS
    try:
        patch_qwen3_moe_sparse_block()
        logits_batched, reg_batched = _run_fwd_bwd(batched_model, method, input_ids)
    finally:
        moe_lora_delete._MOE_TOKEN_CHUNK = orig_chunk
        unpatch_qwen3_moe_sparse_block()

    assert moe_lora_delete._BATCHED_FWD_CALLS == fwd_before + 2

    max_fwd_err = (logits_batched.float() - logits_eager.float()).abs().max().item()
    torch.testing.assert_close(logits_batched, logits_eager, rtol=rtol, atol=atol)
    if method == "dense":
        assert reg_eager.item() > 0.0
        torch.testing.assert_close(reg_batched, reg_eager, rtol=rtol, atol=atol)

    max_grad_err = 0.0
    unrouted_param_count = 0
    routed_param_count = 0
    for (n_e, p_e), (n_b, p_b) in zip(eager_model.named_parameters(), batched_model.named_parameters()):
        assert n_e == n_b
        if not p_e.requires_grad:
            assert p_b.grad is None
            continue
        # Batched dispatch MUST populate non-None gradients on ALL trainable parameters (including unrouted experts).
        assert p_b.grad is not None, f"Expected non-None gradient on {n_b} for FSDP2 safety"
        if p_e.grad is None:
            unrouted_param_count += 1
            assert torch.all(p_b.grad == 0), f"Unrouted expert param {n_b} got non-zero grad: {p_b.grad.abs().max()}"
        else:
            routed_param_count += 1
            err = (p_b.grad.float() - p_e.grad.float()).abs().max().item()
            max_grad_err = max(max_grad_err, err)
            torch.testing.assert_close(p_b.grad, p_e.grad, rtol=rtol, atol=atol, msg=f"Gradient mismatch on {n_b}")

    assert unrouted_param_count >= 2, f"Test setup must leave >=1 unrouted expert per layer, got {unrouted_param_count}"
    assert routed_param_count > 0
    return max_fwd_err, max_grad_err, unrouted_param_count


@pytest.mark.parametrize("method", ["lora_delete", "dense", "fft", "lora"])
def test_all_four_methods_cpu_fp32(method: str):
    max_fwd, max_grad, unrouted = _compare_eager_vs_batched(
        method=method,
        dtype=torch.float32,
        device=torch.device("cpu"),
        rtol=1e-4,
        atol=1e-5,
        chunk_size=5,  # 21 tokens -> 5 chunks to exercise multi-chunk packing + accumulation
    )
    print(f"[CPU fp32 {method}] max_fwd_err={max_fwd:.3e}, max_grad_err={max_grad:.3e}, unrouted_params={unrouted}")


def test_lora_delete_all_three_expert_projections_cpu_fp32():
    max_fwd, max_grad, unrouted = _compare_eager_vs_batched(
        method="lora_delete",
        dtype=torch.float32,
        device=torch.device("cpu"),
        rtol=1e-4,
        atol=1e-5,
        lora_delete_targets=["gate_proj", "up_proj", "down_proj"],
        chunk_size=6,
    )
    assert unrouted >= 4 * 3 * 2  # >=4 unrouted experts across 2 layers * 3 projs * (B, A)
    print(f"[CPU fp32 lora_delete 3-proj] max_fwd_err={max_fwd:.3e}, max_grad_err={max_grad:.3e}")


@pytest.mark.parametrize("method", ["lora_delete", "dense", "fft", "lora"])
def test_eval_mode_and_hook_fallback_cpu(method: str):
    base = _make_tiny_qwen3_moe(seed=11)
    model = _prepare_method_model(base, method, dtype=torch.float32, device=torch.device("cpu")).eval()
    input_ids = torch.randint(0, 128, (2, 6))

    unpatch_qwen3_moe_sparse_block()
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids).logits

    patch_qwen3_moe_sparse_block()
    fwd_before = moe_lora_delete._BATCHED_FWD_CALLS
    with torch.no_grad():
        batched_logits = model(input_ids=input_ids).logits
    assert moe_lora_delete._BATCHED_FWD_CALLS == fwd_before + 2
    torch.testing.assert_close(batched_logits, ref_logits, rtol=1e-5, atol=1e-6)

    #PEFT disable_adapter check (used by verl ref_compute_log_prob).
    if method == "lora":
        unpatch_qwen3_moe_sparse_block()
        with torch.no_grad(), model.disable_adapter():
            ref_disabled = model(input_ids=input_ids).logits
        patch_qwen3_moe_sparse_block()
        with torch.no_grad(), model.disable_adapter():
            batched_disabled = model(input_ids=input_ids).logits
        torch.testing.assert_close(batched_disabled, ref_disabled, rtol=1e-5, atol=1e-6)

    # Registering a forward hook on ANY expert projection must fall back to eager forward.
    hook_calls = []
    layers = model.base_model.model.model.layers if method == "lora" else model.model.layers
    # Attach hook to expert 0 and expert 1 (antithetic router pair -> at least one is guaranteed routed!).
    handles = [
        layers[0].mlp.experts[0].gate_proj.register_forward_hook(lambda _m, _i, _o: hook_calls.append(0)),
        layers[0].mlp.experts[1].gate_proj.register_forward_hook(lambda _m, _i, _o: hook_calls.append(1)),
    ]
    try:
        fwd_before_hook = moe_lora_delete._BATCHED_FWD_CALLS
        with torch.no_grad():
            hook_logits = model(input_ids=input_ids).logits
        # Layer 0 has hooks -> falls back to eager; Layer 1 has no hooks -> runs batched!
        assert moe_lora_delete._BATCHED_FWD_CALLS == fwd_before_hook + 1
        assert len(hook_calls) >= 1
        torch.testing.assert_close(hook_logits, ref_logits, rtol=1e-5, atol=1e-6)
    finally:
        for h in handles:
            h.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU/Triton tests")
@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float32, 1e-4, 1e-5), (torch.bfloat16, 2e-2, 1e-3)])
@pytest.mark.parametrize("method", ["lora_delete", "dense", "fft", "lora"])
def test_all_four_methods_gpu(method: str, dtype: torch.dtype, rtol: float, atol: float):
    max_fwd, max_grad, unrouted = _compare_eager_vs_batched(
        method=method,
        dtype=dtype,
        device=torch.device("cuda:0"),
        rtol=rtol,
        atol=atol,
        chunk_size=5,
    )
    print(f"[GPU {dtype} {method}] max_fwd_err={max_fwd:.3e}, max_grad_err={max_grad:.3e}, unrouted_params={unrouted}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU/Triton tests")
def test_lora_delete_all_three_expert_projections_gpu_bf16():
    max_fwd, max_grad, _ = _compare_eager_vs_batched(
        method="lora_delete",
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        rtol=2e-2,
        atol=1e-3,
        lora_delete_targets=["gate_proj", "up_proj", "down_proj"],
        chunk_size=6,
    )
    print(f"[GPU bf16 lora_delete 3-proj] max_fwd_err={max_fwd:.3e}, max_grad_err={max_grad:.3e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real FSDP1 summon tests")
@pytest.mark.parametrize("method", ["lora_delete", "dense"])
def test_fsdp1_summon_collect_merged_params_and_save_mask(method: str, tmp_path):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    dev = torch.device("cuda:0")
    base = _make_tiny_qwen3_moe(seed=21)
    # In FSDP1 + MixedPrecision (like qwen30b_a3b in verl), all orig_params start in uniform fp32 dtype while
    # MixedPrecision(param_dtype=torch.bfloat16) casts unsharded params and merged vLLM weights to bf16.
    unsharded = _prepare_method_model(base, method, dtype=torch.float32, device=dev)
    # Perturb mask logits so some weights are deleted and some are kept.
    with torch.no_grad():
        for mod in unsharded.modules():
            if isinstance(mod, LoRADeleteSparseLinear):
                mod.mask_B.normal_(std=0.05)
            elif isinstance(mod, SparseLinear):
                mod.diff_mask.gate.normal_(std=0.05)

    expected_merged = {k: v.to(dtype=torch.bfloat16).cpu() for k, v in verl_adapters.collect_merged_params(unsharded)}
    verl_adapters.save_mask_checkpoint(unsharded, str(tmp_path / f"ref_{method}"))
    expected_masks = apply_mask.load_mask(str(tmp_path / f"ref_{method}" / "mask"))

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock},
    )
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    fsdp_model = FSDP(
        copy.deepcopy(unsharded),
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=dev,
        use_orig_params=True,
    )
    assert isinstance(fsdp_model, FSDP)

    got_merged = {k: v.cpu() for k, v in verl_adapters.collect_merged_params(fsdp_model)}
    assert got_merged.keys() == expected_merged.keys()
    for k in expected_merged:
        torch.testing.assert_close(got_merged[k], expected_merged[k], rtol=0.0, atol=0.0, msg=f"Mismatch on {k}")

    verl_adapters.save_mask_checkpoint(fsdp_model, str(tmp_path / f"fsdp_{method}"))
    got_masks = apply_mask.load_mask(str(tmp_path / f"fsdp_{method}" / "mask"))
    assert got_masks.keys() == expected_masks.keys()
    for k in expected_masks:
        assert torch.equal(got_masks[k], expected_masks[k]), f"Mask mismatch on {k}"
