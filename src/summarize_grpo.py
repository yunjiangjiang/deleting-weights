"""Print a GRPO run's best validation result.

GSM8K runs report the best validation accuracy over the evaluations during training. SimpleRL-Zoo
runs report, at each evaluation, the mean accuracy over the six benchmarks, and take the best
step by that mean (the earliest step on ties).

Usage: python src/summarize_grpo.py <run_dir> [<run_dir> ...]
"""

import json
import os
import sys

DATA_SOURCES = {
    "gsm8k": ["openai/gsm8k"],
    "countdown_5": ["countdown_5"],
    "simplerl_zoo": [f"simplerl_zoo_{b}" for b in ["gsm8k", "math500", "minerva", "olympiadbench", "aime24", "amc23"]],
}


def summarize(run_dir: str) -> dict:
    """Return the best step, its score in percent, and the per-benchmark scores at that step."""
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        history = [json.loads(line) for line in f if line.strip()]
    best = None
    for entry in history:
        row = entry["data"]
        for sources in DATA_SOURCES.values():
            if entry["step"] > 0 and f"val-core/{sources[0]}/acc/mean@1" in row:
                per_bench = {s: row[f"val-core/{s}/acc/mean@1"] for s in sources}
                score = sum(per_bench.values()) / len(per_bench)
                if best is None or score > best["score"]:
                    best = {"step": entry["step"], "score": score, "per_benchmark": per_bench}
    assert best is not None, f"no validation results in {run_dir}"
    return {
        "run": run_dir,
        "best_step": best["step"],
        "score": round(100 * best["score"], 2),
        "per_benchmark": {k: round(100 * v, 2) for k, v in best["per_benchmark"].items()},
    }


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(summarize(run)))
