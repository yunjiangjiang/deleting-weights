import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "countdown_reward",
    os.path.join(os.path.dirname(__file__), "..", "verl", "verl", "utils", "reward_score", "countdown_reward.py"),
)
countdown_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(countdown_reward)


def test_countdown_exact_rational_and_multiset():
    gt = json.dumps({"nums": [3, 8, 3, 2, 10], "target": 25})
    assert countdown_reward.compute_score("think... <answer>(8 - 3) * (3 + 2) * 10 / 10</answer>", gt) == 0.0
    assert countdown_reward.compute_score("step... <answer>(8 - 3) * (3 + 2) + 10 - 10</answer>", gt) == 0.0
    assert countdown_reward.compute_score("step... <answer>(10 + 3 + 2) + (8 + 3) - 1</answer>", gt) == 0.0
    assert countdown_reward.compute_score("step... <answer>10 * 2 + 8 - 3 / 3</answer>", json.dumps({"nums": [3, 8, 3, 2, 10], "target": 27})) == 1.0
    gt_rat = json.dumps({"nums": [1, 2, 3, 3, 9], "target": 9})
    assert countdown_reward.compute_score("<answer>(1 / 3 + 2 / 3) * 9</answer>", gt_rat) == 1.0
    assert countdown_reward.compute_score("<answer>9 / (3 - 3)</answer>", gt_rat) == 0.0
