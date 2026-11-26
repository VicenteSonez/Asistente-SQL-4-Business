#!/usr/bin/env python3
"""Deliverable 4: build the Planner's fine-tuning data (no GPU, no model, deterministic).

    python scripts/build_planner_data.py            # writes data/finetune/{train,dev}.jsonl,
                                                    # revision_manual.csv and reporte.json

The output is versioned; running the script again with the same seed rebuilds the same files.
See src/sql4business/finetune_data.py for how each correct plan is produced and checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sql4business.finetune_data import SEED, build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=ROOT / "data" / "business.db", type=Path)
    parser.add_argument("--out", default=ROOT / "data" / "finetune", type=Path)
    parser.add_argument("--plans", type=int, default=900)
    parser.add_argument("--repairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = build_dataset(
        args.db, [ROOT / "data" / "questions.json", ROOT / "data" / "paraphrases.json"], args.out,
        plan_examples=args.plans, repair_examples=args.repairs, seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
