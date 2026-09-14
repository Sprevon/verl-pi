#!/usr/bin/env python3
"""Audit a real RL run's Pi traces, optimizer metrics and checkpoint files."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-learning-signal", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir
    failures, sessions = [], []
    groups = defaultdict(list)
    for path in sorted((run_dir / "pi-traces").glob("*.jsonl")):
        events = read_jsonl(path)
        sample = events[0]
        requests = [event for event in events if event["type"] == "generation_request"]
        tokens = [event for event in events if event["type"] == "generation_tokens"]
        turns = [event for event in events if event["type"] == "step_complete"]
        evaluations = [event for event in events if event["type"] == "evaluation_result"]
        completions = [event for event in events if event["type"] == "session_complete"]
        if any(event["type"] == "rollout_error" for event in events):
            failures.append(f"{path.name}: rollout_error")
        if not requests or len(requests) != len(tokens) or len(tokens) != len(turns):
            failures.append(f"{path.name}: missing generation/turn events")
        if len(evaluations) != 1 or len(completions) != 1:
            failures.append(f"{path.name}: missing/duplicate evaluation or completion")
            continue
        for token_record in tokens:
            count = len(token_record["response_ids"])
            logprobs, mask = token_record["response_logprobs"], token_record["response_mask"]
            if not count or len(mask) != count or not all(value == 1 for value in mask):
                failures.append(f"{path.name}: invalid training mask")
            if logprobs is None or len(logprobs) != count or not all(math.isfinite(value) for value in logprobs):
                failures.append(f"{path.name}: invalid rollout logprobs")
        reward = evaluations[0]["result"]["reward"]
        summary = {
            "session_id": sample["session_id"],
            "uid": sample["uid"],
            "split": sample.get("split"),
            "task_id": sample["task_id"],
            "turns": len(turns),
            "generated_tokens": sum(len(event["response_ids"]) for event in tokens),
            "tool_results": sum(len(event["tool_results"]) for event in turns),
            "reward": reward,
            "truncated": completions[0].get("truncated", False),
        }
        sessions.append(summary)
        if sample.get("split") == "train":
            groups[sample["uid"]].append(reward)
    if not sessions or not groups:
        failures.append("No completed training sessions")
    if any(len(rewards) < 2 for rewards in groups.values()):
        failures.append("A GRPO training group has fewer than two samples")
    if not sum(session["tool_results"] for session in sessions):
        failures.append("Student rollouts did not execute any Pi tools")
    records = read_jsonl(run_dir / "metrics.jsonl")
    updates = [record for record in records if "actor/grad_norm" in record["data"]]
    if not updates:
        failures.append("No actor update metrics")
    for record in updates:
        for key in ("actor/grad_norm", "actor/pg_loss"):
            value = record["data"].get(key)
            if value is None or not math.isfinite(value):
                failures.append(f"Nonfinite/missing {key} at step {record['step']}")
    checkpoints = sorted((run_dir / "checkpoints").glob("global_step_*"))
    checkpoint_files = [
        str(path.relative_to(run_dir)) for checkpoint in checkpoints for path in checkpoint.rglob("*.pt")
    ]
    if not any("model" in Path(path).name for path in checkpoint_files):
        failures.append("Missing actor model checkpoint")
    if not any("optim" in Path(path).name for path in checkpoint_files):
        failures.append("Missing optimizer checkpoint")
    if (run_dir / "exit_code.txt").read_text().strip() != "0":
        failures.append("Trainer exited unsuccessfully")
    learning_signal = any(len(set(rewards)) > 1 for rewards in groups.values()) and any(
        (record["data"].get("actor/grad_norm") or 0) > 0 for record in updates
    )
    if args.require_learning_signal and not learning_signal:
        failures.append("No nonzero GRPO learning signal")
    result = {
        "passed": not failures,
        "learning_signal": learning_signal,
        "failures": failures,
        "group_rewards": dict(groups),
        "sessions": sessions,
        "updates": [
            {
                "step": record["step"],
                **{
                    key: value
                    for key, value in record["data"].items()
                    if key.startswith(("actor/", "critic/advantages/", "training/", "val-core/"))
                },
            }
            for record in updates
        ],
        "checkpoint_files": checkpoint_files,
    }
    (run_dir / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
