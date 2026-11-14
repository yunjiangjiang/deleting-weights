"""Build Countdown-5 train/test parquets and Countdown-4 OOD test parquet for verl GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def make_messages(nums: list[int], target: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                f"Using the numbers {nums}, create an equation that equals {target}. "
                "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                "Reason step by step, and return ONLY the final arithmetic expression (without '= target') "
                "in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
            ),
        },
    ]


def _to_rows(df: pd.DataFrame, data_source: str) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        nums = [int(x) for x in list(row["nums"])]
        target = int(row["target"])
        rows.append({
            "data_source": data_source,
            "prompt": make_messages(nums, target),
            "ability": "math",
            "reward_model": {"ground_truth": json.dumps({"nums": nums, "target": target}), "style": "rule"},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, default=Path("data/countdown"))
    ap.add_argument("--train_size", type=int, default=32768)
    ap.add_argument("--test5_size", type=int, default=512)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    fp5_train = hf_hub_download("yzhuang/tinyzero-Countdown-Tasks-5", "data/train-00000-of-00001.parquet", repo_type="dataset")
    fp5_test = hf_hub_download("yzhuang/tinyzero-Countdown-Tasks-5", "data/test-00000-of-00001.parquet", repo_type="dataset")
    fp34 = hf_hub_download("Jiayi-Pan/Countdown-Tasks-3to4", "data/train-00000-of-00001.parquet", repo_type="dataset")

    df5_train = pd.read_parquet(fp5_train).iloc[: args.train_size].reset_index(drop=True)
    df5_test = pd.read_parquet(fp5_test).iloc[: args.test5_size].reset_index(drop=True)
    df34_test = pd.read_parquet(fp34).iloc[327680 : 327680 + 300]
    df4_test = df34_test[df34_test["nums"].apply(len) == 4].reset_index(drop=True)

    for name, rows in [
        ("train_5.parquet", _to_rows(df5_train, "countdown_5")),
        ("test_5.parquet", _to_rows(df5_test, "countdown_5")),
        ("test_4.parquet", _to_rows(df4_test, "countdown_4")),
    ]:
        out = args.dst / name
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
