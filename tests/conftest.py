"""Put src/ on the import path, as scripts/train_grpo.sh does with PYTHONPATH."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
