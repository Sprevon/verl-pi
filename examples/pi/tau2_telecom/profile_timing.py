#!/usr/bin/env python3
"""Observe a real Pi run and summarize wall-time intervals; execute on vGPUN."""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def union_seconds(intervals):
    end, total = float("-inf"), 0.0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError(f"Reversed interval: {start}, {stop}")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def interval(item):
    return item["start_unix_s"], item["end_unix_s"]


def monitor(directory, gpu, period):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_unix_s": time.time(),
        "environment": {
            key: os.environ.get(key)
            for key in ("STUDENT_MODEL", "PI_ROLLOUT_N", "PI_MAX_TURNS", "PI_DATA_DIR", "PI_MAX_RESPONSE_LENGTH")
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
    }
    (directory / "timing-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (directory / "gpu-samples.csv").open("w") as gpu_file, (directory / "server-samples.jsonl").open("w") as stats:
        child = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                f"--loop-ms={max(100, int(period * 1000))}",
            ],
            stdout=gpu_file,
        )
        address = None
        try:
            while not stopping and not (directory / "profile_exit_code.txt").exists():
                if address is None and (directory / "train.log").exists():
                    match = re.search(r"LLMServerManager: \['([^']+)'", (directory / "train.log").read_text())
                    if match:
                        address = match[1]
                if address:
                    sample = {"unix_s": time.time(), "address": address}
                    try:
                        with urllib.request.urlopen(f"http://{address}/metrics", timeout=1) as response:
                            text = response.read().decode()
                        values = defaultdict(float)
                        for line in text.splitlines():
                            match = re.match(
                                r"vllm:(num_requests_running|num_requests_waiting|generation_tokens_total|prompt_tokens_total)"
                                r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
                                line,
                            )
                            if match:
                                values[match[1]] += float(match[2])
                        sample["metrics"] = dict(values)
                    except Exception as exc:
                        sample["error"] = str(exc)
                    stats.write(json.dumps(sample) + "\n")
                    stats.flush()
                time.sleep(period)
        finally:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def trajectory(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sample = next(event for event in events if event["type"] == "sample")
    requests = [event for event in events if event["type"] == "generation_request"]
    completion = next((event for event in events if event["type"] == "session_complete"), None)
    closed = next(event for event in events if event["type"] == "session_closed")
    start, stop = sample["trace_unix_s"], closed["trace_unix_s"]
    phases = [event for event in events if event["type"] == "phase_timing"]
    rows = []
    if requests:
        rows.append({"phase": "startup", "start_unix_s": start, "end_unix_s": requests[0]["trace_unix_s"]})
    for phase in phases:
        if phase["phase"] not in {"wait_event", "sidecar_start"}:
            rows.append(dict(phase))
    for event in events:
        if event["type"] == "step_complete":
            results = {result.get("toolCallId"): result for result in event.get("tool_results", [])}
            for tool in event.get("tool_timings", []):
                row = {"phase": "tool", "generation_id": event["generation_id"], **tool}
                result = results.get(tool["tool_call_id"], {})
                details = result.get("details")
                row["is_error"] = bool(
                    tool.get("is_error") or result.get("isError") or (isinstance(details, dict) and details.get("error"))
                )
                rows.append(row)
    if completion and "evaluation_timing" in completion:
        rows.append({"phase": "evaluation", **completion["evaluation_timing"]})
    rows.sort(key=lambda row: row["start_unix_s"])
    categories = defaultdict(list)
    for row in rows:
        categories[row["phase"]].append(interval(row))
        row["duration_s"] = row["end_unix_s"] - row["start_unix_s"]
        row["offset_s"] = row["start_unix_s"] - start
    durations = {name: union_seconds(spans) for name, spans in categories.items()}
    # Tool calls may overlap each other. Their union, never their sum, enters the wall-time ledger.
    covered = union_seconds([interval(row) for row in rows])
    durations["other"] = max(0.0, stop - start - covered)
    tokens = sum(len(event["response_ids"]) for event in events if event["type"] == "generation_tokens")
    reward = next((event["result"].get("reward") for event in events if event["type"] == "evaluation_result"), None)
    return {
        "file": path.name,
        "session_id": sample["session_id"],
        "uid": sample["uid"],
        "task_id": sample["task_id"],
        "split": sample.get("split"),
        "reward": reward,
        "start_unix_s": start,
        "end_unix_s": stop,
        "wall_s": stop - start,
        "completed": completion is not None and not any(event["type"] == "rollout_error" for event in events),
        "turns": len(requests),
        "tool_calls": sum(row["phase"] == "tool" for row in rows),
        "tool_errors": sum(row["phase"] == "tool" and row.get("is_error", False) for row in rows),
        "truncated": completion.get("truncated") if completion else None,
        "response_tokens": tokens,
        "phase_seconds": durations,
        "phase_overlap_s": sum(durations.values()) - (stop - start),
        "timeline": rows,
        "sidecar_spawn_handshake_s": sum(p["duration_s"] for p in phases if p["phase"] == "sidecar_start"),
        "errors": [event.get("error") for event in events if event["type"] == "rollout_error"],
    }


def summarize(directory):
    sessions = [trajectory(path) for path in sorted((directory / "pi-traces").glob("*.jsonl"))]
    if not sessions:
        raise ValueError("No real Pi traces found")
    gpu_samples = []
    with (directory / "gpu-samples.csv").open() as source:
        for row in csv.reader(source):
            if len(row) >= 5:
                gpu_samples.append(
                    (datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(row[1]))
                )
    groups = defaultdict(list)
    server_samples = [json.loads(line) for line in (directory / "server-samples.jsonl").read_text().splitlines()]
    for session in sessions:
        groups[session["uid"]].append(session)
    batches = []
    for uid, group in groups.items():
        start, stop = min(s["start_unix_s"] for s in group), max(s["end_unix_s"] for s in group)
        requests = [interval(row) for s in group for row in s["timeline"] if row["phase"] == "llm_request"]
        request_union = union_seconds(requests)
        samples = [value for stamp, value in gpu_samples if start <= stamp <= stop]
        server_window = [row.get("metrics", {}) for row in server_samples if start <= row["unix_s"] <= stop]
        batches.append(
            {
                "uid": uid,
                "split": group[0]["split"],
                "trajectories": len(group),
                "wall_s": stop - start,
                "llm_request_sum_s": sum(b - a for a, b in requests),
                "llm_request_union_s": request_union,
                "llm_request_presence_pct": 100 * request_union / (stop - start),
                "no_llm_request_pct": 100 * (1 - request_union / (stop - start)),
                "sampled_gpu_util_mean_pct": sum(samples) / len(samples) if samples else None,
                "sampled_gpu_util_nonzero_pct": 100 * sum(value > 0 for value in samples) / len(samples)
                if samples
                else None,
                "gpu_sample_count": len(samples),
                "sampled_server_running_max": max(
                    (row["num_requests_running"] for row in server_window if "num_requests_running" in row), default=None
                ),
                "sampled_server_waiting_max": max(
                    (row["num_requests_waiting"] for row in server_window if "num_requests_waiting" in row), default=None
                ),
            }
        )
    result = {
        "metadata": json.loads((directory / "timing-metadata.json").read_text()),
        "sessions": sessions,
        "groups": batches,
    }
    (directory / "timing-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Real Pi trajectory timing",
        "",
        f"Commit: `{result['metadata']['commit']}`",
        "",
        "LLM request latency includes RPC, routing, queueing, prefill and decode; it is not GPU kernel time.",
        "GPU values are sampled nvidia-smi utilization, not useful-compute efficiency. Tool wall time uses interval unions.",
        "",
        "| Group | Split | Trajectories | Wall s | LLM union s | Request presence % | Mean sampled GPU % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in batches:
        lines.append(
            f"| {group['uid'][:8]} | {group['split']} | {group['trajectories']} | {group['wall_s']:.3f} | "
            f"{group['llm_request_union_s']:.3f} | {group['llm_request_presence_pct']:.2f} | "
            f"{group['sampled_gpu_util_mean_pct']} |"
        )
    for session in sessions:
        lines.extend(
            [
                "",
                f"## Session {session['session_id']}",
                "",
                f"Task: `{session['task_id']}`; split={session['split']}; "
                f"completed={session['completed']}; turns={session['turns']}; tokens={session['response_tokens']}; "
                f"reward={session['reward']}; wall={session['wall_s']:.3f}s.",
                f"Tool calls={session['tool_calls']}; tool errors={session['tool_errors']}; "
                f"truncated={session['truncated']}.",
                "",
                "| Phase | Seconds | Wall % |",
                "|---|---:|---:|",
            ]
        )
        for phase, duration in session["phase_seconds"].items():
            lines.append(f"| {phase} | {duration:.6f} | {100 * duration / session['wall_s']:.2f} |")
        lines.extend(["", "| Start offset s | Duration s | Phase / tool | Generation |", "|---:|---:|---|---|"])
        for row in session["timeline"]:
            lines.append(
                f"| {row['offset_s']:.6f} | {row['duration_s']:.6f} | "
                f"{row.get('name', row['phase'])}{' (error)' if row.get('is_error') else ''} | "
                f"{row.get('generation_id', '').split(':')[-1]} |"
            )
    (directory / "timing-report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "groups": batches,
                "sessions": [{k: v for k, v in session.items() if k != "timeline"} for session in sessions],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("monitor", "summarize"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    if args.mode == "monitor":
        monitor(args.run_dir, args.gpu, args.interval)
    else:
        summarize(args.run_dir)


if __name__ == "__main__":
    main()
