"""CPU tests of the MTH and LoRA baselines: masked AdamW updates and LoRA adapter checkpoint resolution."""

import os

import pytest
import torch
import torch.nn as nn

import apply_mask
import random_sparse_ft


def test_only_masked_entries_train():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8))
    before = [p.detach().clone() for p in model.parameters()]
    random_sparse_ft.apply_random_sparse_ft(model, density=0.1, seed=3)
    masks = [m for _, m in model.mth_grad_masks]
    assert 0.07 < sum(int(m.sum()) for m in masks) / sum(m.numel() for m in masks) < 0.13
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(16, 64)).pow(2).mean().backward()
        random_sparse_ft.mask_gradients(model)
        opt.step()
    moved = torch.cat([(p != p0)[m] for p, p0, m in zip(model.parameters(), before, masks)])
    assert moved.float().mean() > 0.9
    assert all(torch.equal(p[~m], p0[~m]) for p, p0, m in zip(model.parameters(), before, masks))

    again = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8))
    random_sparse_ft.apply_random_sparse_ft(again, density=0.1, seed=3)
    assert all(torch.equal(a, m) for (_, a), m in zip(again.mth_grad_masks, masks))


def test_lora_adapter_model_path_dispatch(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    assert apply_mask.is_lora_adapter(str(adapter_dir))
    assert not apply_mask.is_lora_adapter(str(tmp_path))

    called = {}

    def fake_save_merged(base_model: str, adapter_dir: str, output_dir: str) -> None:
        called["args"] = (base_model, adapter_dir)
        (os.path.join(output_dir, "merged.ok"))
        with open(os.path.join(output_dir, "merged.ok"), "w") as f:
            f.write("ok")

    monkeypatch.setattr(apply_mask, "save_merged_lora", fake_save_merged)
    with apply_mask.model_path(base_model="dummy/base", checkpoint=str(adapter_dir)) as resolved:
        assert os.path.isfile(os.path.join(resolved, "merged.ok"))
    assert called["args"] == ("dummy/base", str(adapter_dir))
    assert not os.path.exists(resolved)
