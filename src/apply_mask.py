"""Apply a Deletion Tuning mask to its base model and save the result as a standard model.

A mask directory holds ``model.safetensors`` and ``config.json``. For every masked weight
matrix, ``model.safetensors`` stores its shape (``<name>.shape``) and its keep bits, bit-packed
and zstd-compressed (``<name>.packed``); ``config.json`` records the format and the mask size.
Applying the mask sets every deleted weight of the base model to zero and changes nothing else,
so the output loads in Hugging Face Transformers or vLLM like the base model.

Usage:
    python src/apply_mask.py --base_model meta-llama/Llama-3.2-1B-Instruct \\
        --mask_dir outputs/grpo/llama1b/global_step_125/mask --output_dir outputs/masked/llama1b
"""

import argparse
import contextlib
import json
import os
import tempfile
from typing import Iterator

import numpy as np
import torch
import transformers
import zstandard
from safetensors.torch import load_file


def load_mask(mask_dir: str) -> dict[str, torch.Tensor]:
    """Return a map from parameter name to boolean keep mask (True keeps the weight)."""
    with open(os.path.join(mask_dir, "config.json")) as f:
        fmt = json.load(f)["format"]
    assert fmt == "packbits_zstd", f"unsupported mask format {fmt!r}"
    tensors = load_file(os.path.join(mask_dir, "model.safetensors"), device="cpu")
    dctx = zstandard.ZstdDecompressor()
    masks = {}
    for key in tensors:
        if not key.endswith(".shape"):
            continue
        name = key[: -len(".shape")]
        shape = tuple(tensors[key].tolist())
        numel = int(np.prod(shape))
        packed = np.frombuffer(dctx.decompress(tensors[f"{name}.packed"].numpy().tobytes()), dtype=np.uint8)
        masks[name] = torch.from_numpy(np.unpackbits(packed)[:numel].astype(bool)).reshape(shape)
    return masks


@torch.no_grad()
def apply_mask(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> int:
    """Zero the deleted weights of ``model`` in place and return how many weights were deleted."""
    params = dict(model.named_parameters())
    missing = sorted(set(masks) - set(params))
    assert not missing, f"mask entries not found in the model: {missing[:3]}"
    deleted = 0
    for name, keep in masks.items():
        weight = params[name]
        assert keep.shape == weight.shape, f"{name}: mask {tuple(keep.shape)} vs weight {tuple(weight.shape)}"
        weight.mul_(keep.to(device=weight.device, dtype=weight.dtype))
        deleted += int((~keep).sum())
    return deleted


def is_mask(path: str) -> bool:
    """Return whether a directory holds a mask rather than a model (a model's config.json names its model type)."""
    with open(os.path.join(path, "config.json")) as f:
        return "model_type" not in json.load(f)


def save_masked_model(base_model: str, mask_dir: str, output_dir: str) -> int:
    """Save the base model in bf16 with the deleted weights zeroed, plus its tokenizer; return the deleted count."""
    model = transformers.AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    deleted = apply_mask(model=model, masks=load_mask(mask_dir=mask_dir))
    model.save_pretrained(output_dir)
    transformers.AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    return deleted


def is_lora_adapter(path: str) -> bool:
    """Return whether a directory holds a PEFT LoRA adapter (adapter_config.json)."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def save_merged_lora(base_model: str, adapter_dir: str, output_dir: str) -> None:
    """Merge a PEFT LoRA adapter into the base model in bf16 and save the merged model with its tokenizer."""
    from peft import PeftModel

    model = transformers.AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(output_dir)
    transformers.AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)


@contextlib.contextmanager
def model_path(base_model: str, checkpoint: str) -> Iterator[str]:
    """Yield a loadable model path for a checkpoint: a mask directory, a LoRA adapter, a model directory, or "base".

    "base" yields the base model and a model directory yields itself; a mask or LoRA adapter yields the merged base
    model, saved in bf16 to a temporary directory that is removed afterwards.
    """
    if checkpoint == "base":
        yield base_model
    elif is_lora_adapter(path=checkpoint):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_merged_lora(base_model=base_model, adapter_dir=checkpoint, output_dir=tmp_dir)
            yield tmp_dir
    elif not is_mask(path=checkpoint):
        yield checkpoint
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_masked_model(base_model=base_model, mask_dir=checkpoint, output_dir=tmp_dir)
            yield tmp_dir


def main() -> None:
    """Load the base model, apply the mask, and save the masked model with its tokenizer."""
    parser = argparse.ArgumentParser(description="Apply a Deletion Tuning mask to its base model.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model (Hugging Face ID or path)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with model.safetensors and config.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the masked model")
    args = parser.parse_args()

    deleted = save_masked_model(base_model=args.base_model, mask_dir=args.mask_dir, output_dir=args.output_dir)
    print(f"deleted {deleted:,} weights; saved to {args.output_dir}")


if __name__ == "__main__":
    main()
