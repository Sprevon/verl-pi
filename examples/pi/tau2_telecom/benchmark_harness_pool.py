#!/usr/bin/env python3
"""Compare transports with native Worker/PiAgentLoop/LLMServerManager and real GPU generation.

This isolates rollout. The benchmark replaces only the Worker's training-output
postprocessing; it does not run actor updates or the TransferQueue trainer.
"""

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

import pandas as pd
import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.agent_loop.pi.harness_pool import get_pool
from verl.workers.rollout.llm_server import LLMServerManager


@ray.remote(num_cpus=1)
class BenchmarkWorker(AgentLoopWorker):
    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        return output

    async def run_batch(self, task_id, uid, mode, concurrency, pool_name):
        from verl.experimental.agent_loop import agent_loop

        loop_config = OmegaConf.to_container(agent_loop._agent_loop_registry["pi_agent"], resolve=True)
        loop_config.update(harness_pool_size=concurrency if mode == "pool" else 0, harness_pool_name=pool_name)
        agent_loop._agent_loop_registry["pi_agent"] = OmegaConf.create(loop_config)
        started = perf_counter()
        trajectories = await asyncio.gather(
            *[
                self._run_agent_loop(
                    {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "logprobs": True},
                    {"step": 0, "sample_index": 0, "rollout_n": index, "validate": False},
                    agent_name="pi_agent",
                    trace=False,
                    extra_info={"task_id": task_id, "split": mode},
                    uid=uid,
                )
                for index in range(concurrency)
            ]
        )
        wall = perf_counter() - started
        rows = []
        for turns in trajectories:
            first = turns[0]
            tokens = [turn.response_ids for turn in turns]
            rows.append(
                {
                    "session_id": first.extra_fields["pi_session_id"],
                    "turns": len(turns),
                    "tokens": sum(map(len, tokens)),
                    "initial_prompt_sha256": hashlib.sha256(json.dumps(first.prompt_ids).encode()).hexdigest(),
                    "response_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    **first.extra_fields["reward_extra_info"],
                }
            )
        return {"uid": uid, "mode": mode, "wall_s": wall, "trajectories": rows}


def config_for(project, model):
    overrides = [
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "algorithm.adv_estimator=grpo",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.nnodes=1",
        "actor_rollout_ref.rollout.n_gpus_per_node=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.max_num_seqs=4",
        "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
        "actor_rollout_ref.rollout.max_model_len=24832",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.disable_log_stats=false",
        "actor_rollout_ref.rollout.multi_turn.format=hermes",
        "actor_rollout_ref.rollout.trace.backend=null",
        "data.max_prompt_length=24576",
        "data.max_response_length=256",
        "data.return_raw_chat=true",
        "+data.apply_chat_template_kwargs.enable_thinking=false",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={project}/examples/pi/tau2_telecom/agent_loop.yaml",
    ]
    with initialize_config_dir(config_dir=str(project / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


async def benchmark(args):
    project = Path(__file__).resolve().parents[3]
    config = config_for(project, args.model)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.run_dir / "benchmark-config.yaml")
    ray.init(num_cpus=8, include_dashboard=False)
    pool = None
    result = {}
    try:
        gpu_started = perf_counter()
        manager = await LLMServerManager.create(config)
        gpu_prepare_s = perf_counter() - gpu_started
        worker_started = perf_counter()
        worker = BenchmarkWorker.remote(config, manager.get_client())
        task = str(pd.read_parquet(args.data).iloc[0]["extra_info"]["task_id"])
        # Includes CPU worker startup plus a complete real batch to warm GPU kernels/KV cache.
        warmup = await worker.run_batch.remote(task, "gpu-warmup", "direct", args.concurrency, args.pool_name)
        worker_and_warmup_s = perf_counter() - worker_started
        loop_config = OmegaConf.to_container(
            OmegaConf.load(project / "examples/pi/tau2_telecom/agent_loop.yaml")[0], resolve=True
        )
        pool_config = {
            "node_binary": loop_config["node_binary"],
            "entrypoint": str(project / "verl/experimental/agent_loop/pi/sidecar/main.mjs"),
            "cwd": loop_config["cwd"],
            "env": loop_config["environment"],
        }
        pool_started = perf_counter()
        pool = get_pool(args.pool_name, pool_config, args.concurrency)
        preparation = await pool.prepare.remote()
        pool_prepare_s = perf_counter() - pool_started
        rows = []
        # Alternating pair order limits a monotonic cache/clock drift advantage.
        for index, mode in enumerate(["direct", "pool", "pool", "direct", "direct", "pool"]):
            row = await worker.run_batch.remote(task, f"{index}-{mode}", mode, args.concurrency, args.pool_name)
            rows.append(row)
            print(json.dumps({"mode": mode, "round": index, "wall_s": row["wall_s"]}), flush=True)
            (args.run_dir / "benchmark-rounds.json").write_text(json.dumps(rows, indent=2) + "\n")
        digests = {trajectory["initial_prompt_sha256"] for row in rows for trajectory in row["trajectories"]}
        if len(digests) != 1:
            raise RuntimeError("Initial tokenized prompts differ across benchmark trajectories")
        direct = [row["wall_s"] for row in rows if row["mode"] == "direct"]
        pooled = [row["wall_s"] for row in rows if row["mode"] == "pool"]
        result = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "task_id": task,
            "model": args.model,
            "concurrency": args.concurrency,
            "gpu_prepare_s": gpu_prepare_s,
            "worker_and_gpu_warmup_s": worker_and_warmup_s,
            "warmup": warmup,
            "pool_prepare_s": pool_prepare_s,
            "pool_preparation": preparation,
            "direct_mean_s": statistics.mean(direct),
            "pool_mean_s": statistics.mean(pooled),
            "steady_speedup": statistics.mean(direct) / statistics.mean(pooled),
            "direct_total_s": sum(direct),
            "pool_total_with_setup_s": pool_prepare_s + sum(pooled),
            "rounds": rows,
            "scope": "rollout only; no actor updates, policy changes, or TransferQueue postprocessing",
        }
        (args.run_dir / "benchmark-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if pool is not None:
            await pool.shutdown.remote()
            ray.kill(pool, no_restart=True)
        ray.shutdown()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--pool-name", required=True)
    asyncio.run(benchmark(parser.parse_args()))
