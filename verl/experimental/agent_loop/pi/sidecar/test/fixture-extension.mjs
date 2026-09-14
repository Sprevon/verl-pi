// Real Pi SDK fixture. It does not replace the agent loop or execute an LLM.
export default function fixture(pi) {
  let calls = 0;
  pi.registerTool({
    name: "probe",
    label: "Probe",
    description: "Double a number to prove the real Pi tool runner executed.",
    parameters: {
      type: "object",
      properties: { value: { type: "number" } },
      required: ["value"],
      additionalProperties: false,
    },
    execute: async (_id, args) => {
      calls += 1;
      return { content: [{ type: "text", text: `probe:${args.value * 2}` }], details: {} };
    },
  });
  pi.on("session_start", () => {
    pi.appendEntry("pi-training-task-prompt", { task_id: "fixture", prompt: "Call probe, then finish." });
  });
  pi.on("session_shutdown", () => {
    if (process.env.PI_FIXTURE_EVALUATION !== "missing") {
      pi.appendEntry("pi-training-evaluation", { reward: calls === 1 ? 1 : 0, terminated: true });
    }
  });
}
