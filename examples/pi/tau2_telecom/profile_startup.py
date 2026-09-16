#!/usr/bin/env python3
"""Merge real Node/Python startup spans, retaining nesting rather than summing it."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def union_seconds(intervals):
    end, total = float("-inf"), 0.0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("Reversed startup interval")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def exclusive_seconds(row, rows):
    """Subtract the union of strictly contained child spans, including other PIDs."""
    start, stop = row["start_unix_s"], row["end_unix_s"]
    children = [
        (child["start_unix_s"], child["end_unix_s"])
        for child in rows
        if child is not row
        and start <= child["start_unix_s"] <= child["end_unix_s"] <= stop
        and (start, stop) != (child["start_unix_s"], child["end_unix_s"])
    ]
    return max(0.0, stop - start - union_seconds(children))


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(directory):
    python_rows = defaultdict(list)
    for path in sorted((directory / "startup-traces").glob("*.jsonl")):
        for row in read_events(path):
            python_rows[row["session_id"]].append(row)
    sessions = []
    for path in sorted((directory / "pi-traces").glob("*.jsonl")):
        events = read_events(path)
        sample = next(event for event in events if event["type"] == "sample")
        llm = next(event for event in events if event.get("phase") == "llm_request")
        received = next(event for event in events if event["type"] == "generation_request")
        session_id = sample["session_id"]
        start, stop = sample["trace_unix_s"], llm["start_unix_s"]
        rows = [event.copy() for event in events if event["type"] == "startup_timing"]
        rows += [row.copy() for row in python_rows[session_id]]
        rows += [
            {**event, "source": "worker"}
            for event in events
            if event["type"] == "phase_timing" and event["phase"] in {"sidecar_start", "tokenization"}
        ]
        rows = [row for row in rows if start <= row["start_unix_s"] <= row["end_unix_s"] <= stop]
        markers = (("generation_request_received", received["trace_unix_s"]), ("llm_request_submitted", stop))
        for name, stamp in markers:
            rows.append(
                {"phase": name, "source": "worker", "start_unix_s": stamp, "end_unix_s": stamp, "duration_s": 0.0}
            )
        rows.sort(key=lambda row: (row["start_unix_s"], -row["end_unix_s"]))
        for row in rows:
            row["offset_s"] = row["start_unix_s"] - start
            row["exclusive_s"] = exclusive_seconds(row, rows)
        covered = union_seconds([(row["start_unix_s"], row["end_unix_s"]) for row in rows])
        sessions.append(
            {
                "session_id": session_id,
                "uid": sample["uid"],
                "task_id": sample["task_id"],
                "split": sample.get("split"),
                "first_llm_submit_s": stop - start,
                "before_generation_received_s": received["trace_unix_s"] - start,
                "uncovered_s": max(0, stop - start - covered),
                "python_trace_present": bool(python_rows[session_id]),
                "timeline": rows,
            }
        )
    if not sessions or not all(session["python_trace_present"] for session in sessions):
        raise ValueError("Missing real trajectory traces or per-session Python timing files")
    result = {"sessions": sessions}
    (directory / "startup-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Real Pi startup breakdown",
        "",
        "Inclusive spans nest: extension load is inside reload; Python startup/environment is inside bindExtensions.",
        "Do not sum inclusive durations. Exclusive time subtracts the union of contained spans across processes.",
        "First LLM submitted means the Worker entered its server_manager.generate await (not GPU kernel start).",
        "Prompt preparation means session.prompt entry to the first provider callback, including skill expansion.",
        "Canonical prompt publication is a marker, not a measurement of the ticket formatter's execution time.",
        "Tau2 solo agent/user simulator init_state: not called in this Pi bridge path.",
        "CPU preflight sessions are excluded. All listed requests belong to real model rollout traces.",
    ]
    for session in sessions:
        lines.extend(
            [
                "",
                f"## Session {session['session_id']} ({session['split']})",
                "",
                f"Task: `{session['task_id']}`",
                f"First LLM submit: {session['first_llm_submit_s']:.6f} s; uncovered: {session['uncovered_s']:.6f} s.",
                "",
                "| Start offset s | Inclusive s | Exclusive s | Process | Phase |",
                "|---:|---:|---:|---|---|",
            ]
        )
        for row in session["timeline"]:
            lines.append(
                f"| {row['offset_s']:.6f} | {row['duration_s']:.6f} | {row['exclusive_s']:.6f} | "
                f"{row['source']} | {row['phase']} |"
            )
    (directory / "startup-report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps([{k: v for k, v in session.items() if k != "timeline"} for session in sessions], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    summarize(parser.parse_args().run_dir)
