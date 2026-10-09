"""Re-score an existing JSONL file from run_eval.py, without re-running
anything -- for re-checking a past run, or scoring a baseline file committed
under evals/baselines/."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import print_score_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score an existing eval JSONL file.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.path.read_text().splitlines() if line.strip()]
    print(f"scoring {len(records)} records from {args.path}")
    print()
    print_score_table(records)
