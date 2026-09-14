import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import test from "node:test";

async function scenario({ missingEvaluation = false } = {}) {
  const directory = await mkdtemp(join(tmpdir(), "verl-pi-test-"));
  const child = spawn(process.execPath, [fileURLToPath(new URL("../main.mjs", import.meta.url))], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PI_FIXTURE_EVALUATION: missingEvaluation ? "missing" : "present" },
  });
  let stderr = "";
  child.stderr.on("data", (data) => { stderr += data; });
  const exited = new Promise((resolve) => child.once("exit", resolve));
  const events = [];
  const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
  let timer;
  try {
    await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error(`Pi test timed out: ${stderr}`)), 30000);
      child.once("error", reject);
      const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
      lines.on("line", (line) => {
        try {
          const event = JSON.parse(line);
          events.push(event);
          if (event.type === "ready") {
            assert.equal(event.protocol_version, 2);
            send({
              type: "start_session", session_id: "session0", task_id: "fixture", cwd: directory,
              agent_dir: join(directory, "agent"), max_turns: 3,
              training_extension: fileURLToPath(new URL("fixture-extension.mjs", import.meta.url)),
            });
          } else if (event.type === "generation_request") {
            const first = events.filter((item) => item.type === "generation_request").length === 1;
            send({
              type: "response", response_to: event.id, ok: true,
              result: first
                ? { text: "", tool_calls: [{ id: "probe0", name: "probe", arguments: { value: 7 } }] }
                : { text: "Done", tool_calls: [] },
            });
          } else if (event.type === "session_complete" || event.type === "session_error") {
            resolve();
          }
        } catch (error) {
          reject(error);
        }
      });
    });
    return events;
  } finally {
    clearTimeout(timer);
    child.stdin.end();
    const killTimer = setTimeout(() => child.kill("SIGKILL"), 3000);
    await exited;
    clearTimeout(killTimer);
    await rm(directory, { recursive: true, force: true });
  }
}

test("real Pi SDK runs a tool and evaluates before session completion", async () => {
  const events = await scenario();
  assert.equal(events.filter((event) => event.type === "generation_request").length, 2);
  const turns = events.filter((event) => event.type === "step_complete");
  assert.equal(turns.length, 2);
  assert.equal(turns[0].tool_results[0].content[0].text, "probe:14");
  const evaluation = events.findIndex((event) => event.type === "evaluation_result");
  const completion = events.findIndex((event) => event.type === "session_complete");
  assert.ok(evaluation > 0 && completion > evaluation);
  assert.equal(events[evaluation].result.reward, 1);
  assert.equal(events[completion].turns, 2);
});

test("real Pi SDK session fails when the extension omits its evaluator result", async () => {
  const events = await scenario({ missingEvaluation: true });
  assert.ok(events.some((event) => event.type === "session_error" && /evaluation result/.test(event.error)));
  assert.ok(!events.some((event) => event.type === "session_complete"));
});
