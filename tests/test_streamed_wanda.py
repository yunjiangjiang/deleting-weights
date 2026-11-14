"""Unit tests for streamed O(1)-VRAM Wanda calibration, dense streaming prior, and WANDA_CACHE_PATH.

Verifies bit-exact (0.0 diff) equivalence between ``compute_wanda_thresholds_and_norms`` (under
forced streaming ``max_gpu_fraction=0.0`` and partial resident budget) and ``compute_wanda_scores``
+ ``wanda_threshold``, both with and without ``patch_qwen3_moe_sparse_block()`` active, as well as
bit-exact ``init_masks_from_wanda_thresholds`` vs ``init_masks_from_wanda`` for Dense Delete.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import verl_adapters
import wanda_calibration
from learnable_sparsity import sparse_linear
from learnable_sparsity.lora_delete_linear import LoRADeleteSparseLinear, convert_to_lora_delete_model
from learnable_sparsity.moe_lora_delete import patch_qwen3_moe_sparse_block, unpatch_qwen3_moe_sparse_block
from learnable_sparsity.sparse_linear import SparseLinear


@pytest.fixture(scope="module", autouse=True)
def single_process_group():
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, init_method="tcp://127.0.0.1:29539", rank=0, world_size=1)
    yield


@pytest.fixture(autouse=True)
def restore_moe_patch():
    unpatch_qwen3_moe_sparse_block()
    yield
    unpatch_qwen3_moe_sparse_block()


def _make_moe_model(seed: int = 0, num_layers: int = 3) -> Qwen3MoeForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder_sparse_step=1,
    )
    model = Qwen3MoeForCausalLM(config)
    with torch.no_grad():
        g0 = model.model.layers[0].mlp.gate.weight
        g0[0].normal_(std=5.0)
        g0[1].copy_(-g0[0])
        g0[2].normal_(std=5.0)
        g0[3].copy_(-g0[2])
        g0[7].zero_()
    return model


def _calib_batches(seed: int = 42) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [
        {
            "input_ids": torch.randint(0, 128, (3, 9)),
            "attention_mask": torch.ones(3, 9, dtype=torch.long),
        }
        for _ in range(2)
    ]


def _assert_streamed_wanda_bit_exact(device: torch.device, dtype: torch.dtype, max_gpu_fraction: float):
    base = _make_moe_model(seed=5, num_layers=3).to(dtype=dtype)
    batches = _calib_batches(seed=42)
    target_modules = ["gate_proj", "q_proj"]
    init_sparsity = 0.15

    # 1. Reference: unpatched model on target device via standard compute_wanda_scores.
    ref_model = copy.deepcopy(base).to(device=device)
    scores_std, norms_std = wanda_calibration.compute_wanda_scores(
        model=ref_model,
        dataloader=batches,
        device=device,
        num_samples=6,
        target_modules=target_modules,
        return_activation_norms=True,
    )
    thresholds_std = {
        k: wanda_calibration.wanda_threshold(scores=v, init_sparsity=init_sparsity)
        for k, v in scores_std.items()
    }
    norms_std_cpu = {k: v.cpu() for k, v in norms_std.items()}

    # 2. Streamed: start model on CPU WITH patch_qwen3_moe_sparse_block() active (verifying hook fallback!).
    patch_qwen3_moe_sparse_block()
    stream_model = copy.deepcopy(base).to(device="cpu")
    thresholds_stream, norms_stream = wanda_calibration.compute_wanda_thresholds_and_norms(
        model=stream_model,
        dataloader=batches,
        device=device,
        num_samples=6,
        target_modules=target_modules,
        init_sparsity=init_sparsity,
        max_gpu_fraction=max_gpu_fraction,
    )

    # Model must be restored to its original device ("cpu") with no lingering hooks.
    assert next(stream_model.parameters()).device == torch.device("cpu")
    for mod in stream_model.modules():
        assert len(mod._forward_hooks) == 0 and len(mod._forward_pre_hooks) == 0

    assert thresholds_stream.keys() == thresholds_std.keys()
    assert norms_stream.keys() == norms_std_cpu.keys()
    # Unrouted expert 7 on layer 0 receives 0 tokens and must be absent from both dicts, while >15 target layers exist.
    assert "model.layers.0.mlp.experts.7.gate_proj" not in thresholds_stream
    assert len(thresholds_stream) >= 20

    for k in thresholds_std:
        thr_diff = abs(thresholds_stream[k] - thresholds_std[k])
        assert thr_diff == 0.0, f"Threshold diff on {k}: {thr_diff} ({thresholds_stream[k]} vs {thresholds_std[k]})"
        norm_diff = (norms_stream[k] - norms_std_cpu[k]).abs().max().item()
        assert norm_diff == 0.0, f"Activation norm diff on {k}: {norm_diff}"
        assert torch.equal(norms_stream[k], norms_std_cpu[k])

    # 3. Verify Dense Delete streamed prior init matches standard init_masks_from_wanda bit-for-bit (0.0 diff).
    torch.manual_seed(777)
    dense_ref = sparse_linear.convert_to_sparse_model(
        copy.deepcopy(base).to(device="cpu"),
        target_modules=target_modules,
        sharpness=[1000.0, 10000.0],
        total_steps=1,
    )
    wanda_calibration.init_masks_from_wanda(
        model=dense_ref,
        wanda_scores={k: v.cpu() for k, v in scores_std.items()},
        init_sparsity=init_sparsity,
        prior_strength=3.0,
    )

    torch.manual_seed(777)
    dense_stream = sparse_linear.convert_to_sparse_model(
        copy.deepcopy(base).to(device="cpu"),
        target_modules=target_modules,
        sharpness=[1000.0, 10000.0],
        total_steps=1,
    )
    wanda_calibration.init_masks_from_wanda_thresholds(
        model=dense_stream,
        thresholds=thresholds_stream,
        activation_norms=norms_stream,
        prior_strength=3.0,
        score_device=device,
    )

    for (n1, m1), (n2, m2) in zip(dense_ref.named_modules(), dense_stream.named_modules()):
        assert n1 == n2
        if isinstance(m1, SparseLinear):
            gate_diff = (m1.diff_mask.gate - m2.diff_mask.gate).abs().max().item()
            assert gate_diff == 0.0, f"Dense gate init diff on {n1}: {gate_diff}"
            assert torch.equal(m1.diff_mask.gate, m2.diff_mask.gate)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streamed_wanda_bit_exact_cpu(dtype: torch.dtype):
    _assert_streamed_wanda_bit_exact(device=torch.device("cpu"), dtype=dtype, max_gpu_fraction=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CPU->GPU->CPU streaming test")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streamed_wanda_bit_exact_gpu_forced_zero_budget(dtype: torch.dtype):
    # max_gpu_fraction=0.0 forces 0 resident layers so all 3 decoder layers stream CPU -> GPU -> CPU.
    _assert_streamed_wanda_bit_exact(device=torch.device("cuda:0"), dtype=dtype, max_gpu_fraction=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for partial GPU budget streaming test")
def test_streamed_wanda_bit_exact_gpu_partial_resident_budget():
    # Pick a tiny fraction of free GPU memory so exactly 1-2 decoder layers stay resident and the rest stream.
    model = _make_moe_model(seed=5, num_layers=3).to(dtype=torch.bfloat16)
    one_layer_bytes = sum(p.numel() * p.element_size() for p in model.model.layers[0].parameters())
    free_bytes = torch.cuda.mem_get_info(torch.device("cuda:0"))[0]
    frac = (1.5 * one_layer_bytes) / free_bytes
    _assert_streamed_wanda_bit_exact(device=torch.device("cuda:0"), dtype=torch.bfloat16, max_gpu_fraction=frac)


def test_wanda_cache_path_roundtrip_for_lora_delete_and_dense(tmp_path, monkeypatch):
    base = _make_moe_model(seed=13, num_layers=2).to(dtype=torch.bfloat16)
    batches = _calib_batches(seed=9)
    thresholds, norms = wanda_calibration.compute_wanda_thresholds_and_norms(
        model=copy.deepcopy(base),
        dataloader=batches,
        device=torch.device("cpu"),
        num_samples=6,
        target_modules=["gate_proj"],
        init_sparsity=0.2,
        max_gpu_fraction=0.0,
    )
    cache_file = tmp_path / "cache_dir" / "wanda_cache.pt"
    monkeypatch.setenv("WANDA_CACHE_PATH", str(cache_file))
    monkeypatch.setenv("MASK_INIT_SEED", "42")

    called = []

    def _fake_wanda_thresholds(model, model_config):
        called.append(1)
        return thresholds, norms

    monkeypatch.setattr(verl_adapters, "_wanda_thresholds", _fake_wanda_thresholds)

    cfg_lora = OmegaConf.create(
        {
            "mask_type": "lora_delete",
            "mask_rank": 8,
            "mask_init_std": 0.0,
            "mask_prior_strength": 3.0,
            "mask_init_sparsity": 0.2,
            "mask_sharpness_start": 1000.0,
            "mask_sharpness_end": 10000.0,
            "mask_target_modules": ["gate_proj"],
            "mask_weight_reg": 0.0,
            "mask_calib_parquet_files": ["dummy.parquet"],
        }
    )
    # 1st call populates cache_file.
    m_lora_1 = verl_adapters.apply_deletion_tuning(copy.deepcopy(base), cfg_lora)
    assert len(called) == 1 and cache_file.exists()

    # 2nd call loads from cache_file without invoking _wanda_thresholds!
    m_lora_2 = verl_adapters.apply_deletion_tuning(copy.deepcopy(base), cfg_lora)
    assert len(called) == 1
    for (_, mod1), (_, mod2) in zip(m_lora_1.named_modules(), m_lora_2.named_modules()):
        if isinstance(mod1, LoRADeleteSparseLinear):
            assert torch.equal(mod1.col_norm, mod2.col_norm)
            assert torch.equal(mod1.threshold, mod2.threshold)
            assert torch.equal(mod1.full_hard_mask(), mod2.full_hard_mask())

    # 3rd call with mask_type="dense" reuses the exact same WANDA_CACHE_PATH and produces matching Wanda priors!
    cfg_dense = OmegaConf.merge(cfg_lora, {"mask_type": "dense"})
    m_dense = verl_adapters.apply_deletion_tuning(copy.deepcopy(base), cfg_dense)
    assert len(called) == 1
    dense_mods = dict(m_dense.named_modules())
    for n, mod_l in m_lora_1.named_modules():
        if isinstance(mod_l, LoRADeleteSparseLinear):
            mod_d = dense_mods[n]
            assert isinstance(mod_d, SparseLinear)
            # With mask_init_std=0.0 on lora_delete and prior_strength=3.0 on dense (3 * std shift >> 1 * std noise),
            # check Wanda threshold sign matches on nearly all weights (~99.7% due to 3-sigma Gaussian shift).
            if n in thresholds:
                agree = (mod_l.full_hard_mask().view(-1) == (mod_d.diff_mask.gate > 0)).float().mean().item()
                assert agree > 0.99, f"Expected >99% agreement on {n}, got {agree}"
