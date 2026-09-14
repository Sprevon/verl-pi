#!/usr/bin/env python3
"""Exercise the real Pi SDK, canonical skill/tool loop, and Tau2 evaluator.

Generation replies are scripted only in this CPU lifecycle probe. The separate
RL launch uses the student model for every generation through LLMServerClient.
"""

import argparse
import asyncio
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
from transformers import AutoTokenizer

from verl.experimental.agent_loop.pi.client import PiSidecarClient
from verl.experimental.agent_loop.pi.recorder import is_tool_error
from verl.utils.tokenizer.tokenizer import normalize_token_ids


async def probe(task_id, model_path, max_prompt_length):
    project = Path(__file__).resolve().parents[3]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    client = PiSidecarClient(
        node_binary=os.environ["PI_NODE_BINARY"],
        entrypoint=str(project / "verl/experimental/agent_loop/pi/sidecar/main.mjs"),
        env={"TAU2_TELECOM_TASK_ID": task_id, "TAU2_PI_PYTHON": os.environ["TAU2_PI_PYTHON"]},
    )
    count = completed = tool_count = 0
    lengths, tools_used, results = [], [], []
    evaluation = None
    try:
        await client.start(
            {
                "session_id": uuid4().hex,
                "task_id": task_id,
                "cwd": os.environ["TAU2_BENCH_ROOT"],
                "training_extension": os.environ["PI_TRAINING_EXTENSION"],
                "agent_dir": os.environ["PI_AGENT_DIR"],
                "prompt_entry_type": "agent-r1-task-prompt",
                "evaluation_entry_type": "agent-r1-evaluation",
                "max_turns": 3,
            }
        )
        while True:
            event = await client.next_event(180)
            if event["type"] == "generation_request":
                count += 1
                tools = event["tools"]
                tool_count = len(tools)
                ids = normalize_token_ids(
                    tokenizer.apply_chat_template(
                        event["messages"], tools=tools, add_generation_prompt=True, enable_thinking=False
                    )
                )
                if not ids or not all(isinstance(token, int) for token in ids):
                    raise ValueError("Canonical Pi prompt did not produce a flat token ID sequence")
                lengths.append(len(ids))
                if len(ids) > max_prompt_length:
                    raise ValueError(f"Canonical Pi prompt exceeds budget: {len(ids)} > {max_prompt_length}")
                if count == 1:
                    candidates = [
                        tool["function"]
                        for tool in tools
                        if not tool["function"]["parameters"].get("required")
                        and tool["function"]["name"].startswith(("get_", "check_"))
                    ]
                    if not candidates:
                        raise ValueError("Probe needs an allowed read tool without required arguments")
                    name = candidates[0]["name"]
                    tools_used.append(name)
                    reply = {"text": "", "tool_calls": [{"id": "probe-read", "name": name, "arguments": {}}]}
                else:
                    reply = {"text": "The environment connectivity probe is complete.", "tool_calls": []}
                await client.respond(event["id"], reply)
            elif event["type"] == "step_complete":
                completed += 1
                results.extend(event["tool_results"])
            elif event["type"] == "evaluation_result":
                evaluation = event["result"]
            elif event["type"] == "session_complete":
                break
        if count < 2 or completed != count or not results or evaluation is None or "reward" not in evaluation:
            raise RuntimeError("Real Pi/Tau2 lifecycle probe is incomplete")
        if any(is_tool_error(result) for result in results):
            raise RuntimeError("Canonical read tool failed during the lifecycle probe")
        return {
            "task_id": task_id,
            "generations": count,
            "completed_turns": completed,
            "tool_count": tool_count,
            "tools_used": tools_used,
            "prompt_token_lengths": lengths,
            "evaluator_reward": evaluation["reward"],
            "probe_uses_scripted_generation": True,
        }
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=24576)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Single-card preflight requires exactly one visible CUDA device")
    task_id = str(pd.read_parquet(args.train_data).iloc[0]["extra_info"]["task_id"])
    result = asyncio.run(probe(task_id, args.model, args.max_prompt_length))
    result["versions"] = {
        name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "ray", "transferqueue")
    }
    result["gpu"] = torch.cuda.get_device_name(0)
    result["verl_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["tau2_commit"] = subprocess.check_output(
        ["git", "-C", os.environ["TAU2_BENCH_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip()
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
