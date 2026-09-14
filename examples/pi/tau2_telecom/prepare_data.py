#!/usr/bin/env python3
"""Export real Tau2 task IDs; the canonical Pi extension supplies their prompts."""

import argparse
import json
from pathlib import Path

import pandas as pd
from tau2.runner import get_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--train-split", choices=["train", "small"], default="train")
    parser.add_argument("--train-task", help="Optional real task ID from the selected source split")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"domain": "telecom", "splits": {}, "source_splits": {}}
    for split in ("train", "test"):
        source_split = args.train_split if split == "train" else "test"
        tasks = list(get_tasks("telecom", task_split_name=source_split))
        if split == "train" and args.train_task:
            tasks = [task for task in tasks if str(task.id) == args.train_task]
        tasks = tasks[: args.max_tasks]
        if not tasks:
            raise ValueError(f"No matching tasks in Tau2 {split}")
        rows = [
            {
                "agent_name": "pi_agent",
                "data_source": f"tau2_telecom_{source_split}",
                "prompt": [{"role": "user", "content": f"Run Telecom task {task.id}."}],
                "reward_model": {"ground_truth": str(task.id), "style": "rule"},
                "extra_info": {"index": i, "task_id": str(task.id), "split": split, "source_split": source_split},
            }
            for i, task in enumerate(tasks)
        ]
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet", index=False)
        manifest["splits"][split] = [str(task.id) for task in tasks]
        manifest["source_splits"][split] = source_split
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
