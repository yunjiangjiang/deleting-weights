"""CPU tests of mask_type=lora_delete: layer vs the lora_deletion reference, meta init, schedule, regularizer, Wanda, saving.

Run from the repository root: python -m pytest tests/test_lora_delete_mask.py
"""

import copy
import math

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lora_deletion import LoRADeleteLinear
from lora_deletion.lowrank import lowrank_logits
from transformers import LlamaConfig, LlamaForCausalLM

import apply_mask
import verl_adapters
import wanda_calibration
from learnable_sparsity import sparse_linear
from learnable_sparsity.lora_delete_linear import LoRADeleteSparseLinear, convert_to_lora_delete_model


@pytest.fixture(scope="module", autouse=True)
def single_process_group():
    """A one-rank gloo process group, as verl_adapters' collectives expect."""
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
    yield


def _layer(d_out=30, d_in=18, rank=7, sharpness=(20.0, 40.0), seed=0) -> LoRADeleteSparseLinear:
    """A layer with 2D [d_out, rank] / [rank, d_in] factors and a threshold deleting ~1/4 of weights."""
    torch.manual_seed(seed)
    layer = LoRADeleteSparseLinear(
        base=nn.Linear(d_in, d_out, bias=False), rank=rank, prior=0.03,
        sharpness=list(sharpness), total_steps=10,
    )
    assert layer.mask_B.shape == (d_out, rank)
    assert layer.mask_A.shape == (rank, d_in)
    with torch.no_grad():
        layer.mask_B.normal_(std=0.02)
        layer.col_norm.copy_(torch.rand(d_in) + 0.5)
        score = layer.weight.abs() * layer.col_norm
        layer.threshold.fill_(score.flatten().kthvalue(score.numel() // 4).values.item())
    return layer


def _reference(layer: LoRADeleteSparseLinear) -> LoRADeleteLinear:
    """The library's LoRADeleteLinear with the same weight, 2D factors, prior and sharpness."""
    ref = LoRADeleteLinear(
        nn.Linear(layer.in_features, layer.out_features, bias=False), rank=layer.rank,
        prior=layer.prior, sharpness=tuple(layer.sharpness), backend="reference",
    )
    with torch.no_grad():
        ref.weight.copy_(layer.weight)
        ref.lora_B.copy_(layer.mask_B)
        ref.lora_A.copy_(layer.mask_A)
        ref.col_norm.copy_(layer.col_norm)
        ref.threshold.copy_(layer.threshold)
    ref.set_progress(layer.progress)
    return ref


def test_forward_and_hard_mask_match_library_reference():
    layer = _layer()
    layer.current_step = 3
    ref = _reference(layer)
    x = torch.randn(5, 7, layer.in_features)
    torch.testing.assert_close(layer(x), ref(x))
    assert torch.equal(layer.full_hard_mask(), ref.hard_mask())
    assert 0 < (~layer.full_hard_mask()).float().mean() < 1

    layer.sparse = False
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight))
    layer.sparse = True

    layer.eval()
    ref.eval()
    torch.testing.assert_close(layer(x), ref(x))


def test_rank_validation_and_meta_device_construction():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        LoRADeleteSparseLinear(
            base=nn.Linear(16, 32, bias=False), rank=0, prior=0.03, sharpness=[1.0, 2.0], total_steps=1,
        )
    with pytest.raises(ValueError, match="init_std must be >= 0"):
        LoRADeleteSparseLinear(
            base=nn.Linear(16, 32, bias=False), rank=4, prior=0.03, sharpness=[1.0, 2.0], total_steps=1, init_std=-0.01,
        )

    for init_std in (0.01, 0.0):
        with torch.device("meta"):
            model = nn.Sequential(nn.Linear(16, 32, bias=False))
            layer = LoRADeleteSparseLinear(
                base=model[0], rank=8, prior=0.03, sharpness=[1000.0, 10000.0], total_steps=10, init_std=init_std,
            )
        assert layer.weight.is_meta and layer.mask_B.is_meta and layer.mask_A.is_meta
        assert layer.col_norm.is_meta and layer.threshold.is_meta
        assert layer.mask_B.shape == (32, 8) and layer.mask_A.shape == (8, 16)
        materialized = layer.to_empty(device="cpu")
        assert not materialized.mask_A.is_meta and materialized.mask_A.shape == (8, 16)
        assert not materialized.mask_B.is_meta and materialized.mask_B.shape == (32, 8)


def test_default_init_gradient_flow_at_alpha_1000():
    torch.manual_seed(0)
    base = nn.Linear(256, 512, bias=False).float()
    col_norm = torch.rand(256, dtype=torch.float32) + 0.5
    scores = base.weight.data.abs() * col_norm.unsqueeze(0)
    threshold = wanda_calibration.wanda_threshold(scores=scores, init_sparsity=1.0 / 2048)

    layer = LoRADeleteSparseLinear(
        base=copy.deepcopy(base), rank=16, prior=0.03, sharpness=[1000.0, 10000.0], total_steps=10,
    ).train()
    assert layer.init_std == 0.01 and layer.alpha == 1000.0
    with torch.no_grad():
        layer.col_norm.copy_(col_norm)
        layer.threshold.fill_(threshold)
        H = (layer.mask_B @ layer.mask_A) / math.sqrt(layer.rank)
    assert 0.008 < H.std().item() < 0.012

    x = torch.randn(8, 256, dtype=torch.float32)
    layer(x).square().sum().backward()
    assert layer.mask_B.grad is not None and layer.mask_B.grad.norm().item() > 0.0
    assert layer.mask_A.grad is not None and layer.mask_A.grad.norm().item() > 0.0


def test_schedule_for_dense_and_lora_delete():
    model = nn.Sequential(nn.Linear(16, 32, bias=False), nn.Linear(16, 32, bias=False))
    model[0] = _layer(d_out=32, d_in=16, rank=8, sharpness=(1000.0, 10000.0))
    model[1] = sparse_linear.SparseLinear(16, 32, [1000.0, 10000.0], 1, "cpu", torch.float32, "cpu")
    verl_adapters.set_mask_schedule(model=model, total_steps=4)
    alphas = []
    for _ in range(6):
        model[1](torch.randn(2, 16))  # updates the dense mask's current sharpness
        alphas.append((model[0].alpha, model[1].diff_mask.current_sharpness))
        verl_adapters.step_mask_schedules(model=model)
    assert [a for a, _ in alphas] == [1000.0, 3250.0, 5500.0, 7750.0, 10000.0, 10000.0]
    assert all(a == d for a, d in alphas)


def test_injected_reg_gradient_equals_explicit_loss():
    """(task - lambda (1 - p) sum ||W * m||^2) * s, explicit vs injected through reg_coef and open_micro_batch."""
    lam, scale = 0.3, 0.25
    layers = [_layer(d_out=32, d_in=16, rank=8, seed=1), _layer(d_out=32, d_in=16, rank=8, seed=2)]
    for layer in layers:
        layer.total_steps, layer.current_step = 10, 4
    x = torch.randn(6, 16)
    target = torch.randn(6, 32)

    def task_loss(model):
        return (model[1](torch.tanh(model[0](x))[:, :16]) - target).pow(2).mean()

    explicit = nn.Sequential(*copy.deepcopy(layers))
    reg = sum(
        (m.weight * torch.sigmoid(m.alpha * lowrank_logits(
            m.weight, m.mask_B, m.mask_A, m.col_norm, m.threshold, m.prior,
        ))).pow(2).sum()
        for m in explicit
    )
    ((task_loss(explicit) - lam * (1 - 0.4) * reg) * scale).backward()

    injected = nn.Sequential(*copy.deepcopy(layers))
    injected.mask_weight_reg = lam
    verl_adapters.open_micro_batch(model=injected, loss_scale_factor=scale)
    assert all(m.reg_coef == pytest.approx(-lam * 0.6 * scale) for m in injected)
    loss = task_loss(injected)
    reg_loss = verl_adapters.mask_reg_loss(model=injected)
    assert reg_loss.item() == 0.0
    ((loss - reg_loss) * scale).backward()

    for e, i in zip(explicit, injected):
        torch.testing.assert_close(i.mask_B.grad, e.mask_B.grad)
        torch.testing.assert_close(i.mask_A.grad, e.mask_A.grad)
    # The regularizer changes the gradients, so the comparison above is not vacuous.
    plain = nn.Sequential(*copy.deepcopy(layers))
    (task_loss(plain) * scale).backward()
    assert not torch.allclose(plain[0].mask_B.grad, injected[0].mask_B.grad)


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
    )
    return LlamaForCausalLM(config)


def test_wanda_init_matches_dense_prior_signs():
    model = _tiny_llama()
    batches = [{"input_ids": torch.randint(0, 64, (4, 12)), "attention_mask": torch.ones(4, 12, dtype=torch.long)}
               for _ in range(2)]
    scores, norms = wanda_calibration.compute_wanda_scores(
        model=model, dataloader=batches, device=torch.device("cpu"), num_samples=8, target_modules=["gate_proj"],
        return_activation_norms=True,
    )
    assert scores.keys() == norms.keys() and len(scores) == 2
    init_sparsity = 0.05

    dense = sparse_linear.convert_to_sparse_model(
        copy.deepcopy(model), target_modules=["gate_proj"], sharpness=[1000.0, 10000.0], total_steps=1,
    )
    deletions = wanda_calibration.wanda_prior_deletions(model=dense, wanda_scores=scores, init_sparsity=init_sparsity)

    lora = convert_to_lora_delete_model(
        copy.deepcopy(model), target_modules=["gate_proj"], rank=8,
        prior=verl_adapters.LORA_DELETE_PRIOR, sharpness=[1000.0, 10000.0], total_steps=1, init_std=0.0,
    )
    verl_adapters._init_lora_delete_from_wanda(
        model=lora, wanda_scores=scores, activation_norms=norms, init_sparsity=init_sparsity,
    )
    # Re-running Wanda calibration on an already-converted lora_delete model disables masking via self.sparse=False.
    scores_after, norms_after = wanda_calibration.compute_wanda_scores(
        model=lora, dataloader=batches, device=torch.device("cpu"), num_samples=8, target_modules=["gate_proj"],
        return_activation_norms=True,
    )
    for k in scores:
        torch.testing.assert_close(scores_after[k], scores[k])
        torch.testing.assert_close(norms_after[k], norms[k])

    for name, m in lora.named_modules():
        if isinstance(m, LoRADeleteSparseLinear):
            keep = m.full_hard_mask()
            assert torch.equal(keep.view(-1), ~deletions[name])
            # Like the dense prior, scores strictly below the k-th smallest are deleted: k - 1 weights.
            assert (~keep).sum() == int(keep.numel() * init_sparsity) - 1
            logits = lowrank_logits(m.weight, m.mask_B, m.mask_A, m.col_norm, m.threshold, m.prior)
            assert torch.equal(logits.abs(), torch.full_like(logits, 0.03))  # no init noise


class _FakeDTensor:
    """Minimal stand-in for an FSDP1 SHARDED_STATE_DICT DTensor entry."""

    def __init__(self, full: torch.Tensor):
        self._full = full

    def full_tensor(self) -> torch.Tensor:
        return self._full

    def detach(self) -> torch.Tensor:
        return self._full.detach()


class _FakeFSDP1Wrapper(nn.Module):
    """Mimics FSDP1 where raw module weights are 1D flat shards while state_dict() yields DTensors."""

    def __init__(self, inner: nn.Module, full_sd: dict[str, torch.Tensor]):
        super().__init__()
        self._fsdp_wrapped_module = inner
        self._full_sd = full_sd

    def state_dict(self, *args, **kwargs):
        return {f"_fsdp_wrapped_module.{k}": _FakeDTensor(v) for k, v in self._full_sd.items()}


def test_full_masks_merge_and_save_for_unsharded_and_fsdp1(tmp_path):
    model = convert_to_lora_delete_model(
        _tiny_llama(), target_modules=["gate_proj"], rank=8,
        prior=0.03, sharpness=[1000.0, 10000.0], total_steps=1,
    )
    layers = {n: m for n, m in model.named_modules() if isinstance(m, LoRADeleteSparseLinear)}
    for m in layers.values():
        with torch.no_grad():
            m.mask_B.normal_(std=0.05)
    expected = {f"{n}.weight": m.full_hard_mask() for n, m in layers.items()}
    assert all(0 < (~k).sum() < k.numel() for k in expected.values())

    merged = dict(verl_adapters.collect_merged_params(model))
    assert not any("mask_" in k or "col_norm" in k or "threshold" in k for k in merged)
    base = model.state_dict()
    for key, keep in expected.items():
        torch.testing.assert_close(merged[key], base[key] * keep)

    verl_adapters.save_mask_checkpoint(model, str(tmp_path / "unsharded"))
    loaded = apply_mask.load_mask(str(tmp_path / "unsharded" / "mask"))
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[k], expected[k]) for k in expected)

    # Simulate FSDP1 SHARDED_STATE_DICT: flatten local module weights to 1D shards so mod.full_hard_mask() would fail.
    sharded_inner = copy.deepcopy(model)
    for m in sharded_inner.modules():
        if isinstance(m, LoRADeleteSparseLinear):
            m.weight = nn.Parameter(m.weight.data.view(-1)[:8], requires_grad=False)
            with pytest.raises(RuntimeError, match="local shape"):
                m.full_hard_mask()
    fsdp1 = _FakeFSDP1Wrapper(sharded_inner, base)
    merged_fsdp1 = dict(verl_adapters.collect_merged_params(fsdp1))
    for key, val in merged.items():
        torch.testing.assert_close(merged_fsdp1[key], val)
    verl_adapters.save_mask_checkpoint(fsdp1, str(tmp_path / "fsdp1"))
    loaded_fsdp1 = apply_mask.load_mask(str(tmp_path / "fsdp1" / "mask"))
    assert all(torch.equal(loaded_fsdp1[k], expected[k]) for k in expected)
