import { describe, expect, test } from "bun:test"
import {
  LLM_OUTPUT_MESSAGES,
  MESSAGE_CONTENTS,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
} from "@arizeai/openinference-semantic-conventions"
import { createE2EFixture, e2eAvailable, type RunResult } from "./harness.ts"
import type { ExportedSpan } from "./otlp-receiver.ts"

const suite = e2eAvailable ? describe : describe.skip

function requireSuccess(result: RunResult) {
  if (result.exitCode !== 0) {
    throw new Error(`OpenCode exited ${result.exitCode}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`)
  }
}

function one(spans: ExportedSpan[], name: string) {
  const matches = spans.filter(span => span.name === name)
  expect(matches).toHaveLength(1)
  return matches[0]!
}

function expectOk(span: ExportedSpan) {
  const code = span.status["code"]
  expect(code === 1 || code === "STATUS_CODE_OK").toBe(true)
}

type Fixture = Awaited<ReturnType<typeof createE2EFixture>>

async function requireSpans(fixture: Fixture, result: RunResult, count: number) {
  const spans = await fixture.otlp.waitForSpanCount(count)
  if (process.env["OPENCODE_E2E_PRINT_SPANS"]) {
    console.log(`[e2e] exported spans:\n${JSON.stringify(spans, null, 2)}`)
  }
  if (spans.length !== count) {
    throw new Error([
      `Expected ${count} spans, received ${spans.length}`,
      `stdout:\n${result.stdout}`,
      `stderr:\n${result.stderr}`,
      `llm hits:\n${JSON.stringify(fixture.llm.hits.map(hit => hit.body), null, 2)}`,
      `otlp errors:\n${JSON.stringify(fixture.otlp.errors)}`,
      `otlp payloads:\n${JSON.stringify(fixture.otlp.payloads, null, 2)}`,
    ].join("\n"))
  }
  return spans
}

suite("OpenCode plugin E2E", () => {
  test("exports a complete text trace with token and reasoning attributes", async () => {
    const fixture = await createE2EFixture({
      caseID: "text-trace",
      replies: [{
        type: "text",
        text: "hello from e2e",
        reasoning: "reasoning from e2e",
        usage: { input: 11, output: 7, cacheRead: 4, reasoning: 2 },
      }],
    })
    try {
      const result = await fixture.run("say hello")
      requireSuccess(result)
      expect(fixture.llm.pending()).toBe(0)
      expect(fixture.otlp.errors).toEqual([])
      const spans = await requireSpans(fixture, result, 3)

      const run = one(spans, "e2e.run")
      const interaction = one(spans, "e2e.interaction")
      const llm = one(spans, "e2e.llm")
      expect(new Set(spans.map(span => span.traceId)).size).toBe(1)
      expect(interaction.parentSpanId).toBe(run.spanId)
      expect(llm.parentSpanId).toBe(interaction.spanId)
      expect(run.attributes["openinference.span.kind"]).toBe("CHAIN")
      expect(interaction.attributes["openinference.span.kind"]).toBe("AGENT")
      expect(llm.attributes["openinference.span.kind"]).toBe("LLM")
      expect(llm.attributes["llm.model_name"]).toBe("test-model")
      expect(llm.attributes["llm.provider"]).toBe("test")
      expect(llm.attributes["llm.token_count.prompt"]).toBe(11)
      expect(llm.attributes["llm.token_count.prompt_details.cache_write"]).toBe(0)
      expect(llm.attributes["llm.token_count.prompt_details.cache_read"]).toBe(4)
      expect(llm.attributes["llm.token_count.completion"]).toBe(7)
      expect(llm.attributes["llm.token_count.completion_details.reasoning"]).toBe(2)
      expect(llm.attributes["llm.token_count.total"]).toBe(18)
      expect(llm.attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`]).toBe("reasoning")
      expect(llm.attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`]).toBe("reasoning from e2e")
      expect(llm.attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`]).toBe("text")
      expect(String(llm.attributes["output.value"])).toContain("hello from e2e")
      expect(run.attributes["run.total_tokens"]).toBe(18)
      expect(run.attributes["run.total_messages"]).toBe(1)
      expect(run.attributes["e2e.case"]).toBe("text-trace")
      expect(run.resource["service.name"]).toBe("opencode")
      expect(run.resource["e2e.resource"]).toBe("opencode-plugin-otel")
      expectOk(run)
      expectOk(interaction)
      expectOk(llm)
    } finally {
      await fixture.close()
    }
  }, 60_000)

  test("keeps steered user input in one run with distinct interactions", async () => {
    const fixture = await createE2EFixture({
      caseID: "steer",
      replies: [
        { type: "text", text: "first response before steer", usage: { input: 3, output: 2 }, hold: true },
        { type: "text", text: "guided final response", usage: { input: 4, output: 3 } },
      ],
    })
    try {
      const activeRun = await fixture.startRun("initial request")
      expect(fixture.llm.held()).toBe(1)
      await activeRun.steer("steering guidance")
      const result = await activeRun.finish()
      requireSuccess(result)
      expect(fixture.llm.pending()).toBe(0)
      expect(fixture.llm.mainHits()).toHaveLength(2)
      expect(JSON.stringify(fixture.llm.mainHits()[1]!.body)).toContain("steering guidance")
      expect(fixture.otlp.errors).toEqual([])
      const spans = await requireSpans(fixture, result, 5)

      const run = one(spans, "e2e.run")
      const interactions = spans.filter(span => span.name === "e2e.interaction")
      const llms = spans.filter(span => span.name === "e2e.llm")
      expect(interactions).toHaveLength(2)
      expect(llms).toHaveLength(2)
      expect(new Set(spans.map(span => span.traceId)).size).toBe(1)
      expect(interactions.every(span => span.parentSpanId === run.spanId)).toBe(true)
      expect(interactions.every(interaction => llms.some(llm => llm.parentSpanId === interaction.spanId))).toBe(true)

      const interactionByInput = new Map(interactions.map(span => [span.attributes["input.value"], span]))
      expect(interactionByInput.get("initial request")?.attributes["output.value"]).toBe("first response before steer")
      expect(interactionByInput.get("steering guidance")?.attributes["output.value"]).toBe("guided final response")
      expect(JSON.parse(String(run.attributes["input.value"]))).toEqual(["initial request", "steering guidance"])
      expect(run.attributes["output.value"]).toBe("guided final response")
      expect(run.attributes["run.total_interactions"]).toBe(2)
      expect(run.attributes["run.total_messages"]).toBe(2)
      expect(run.attributes["run.total_tokens"]).toBe(12)
      spans.forEach(expectOk)
    } finally {
      await fixture.close()
    }
  }, 60_000)

  test("propagates W3C context to the trace and model request", async () => {
    const traceId = "0af7651916cd43dd8448eb211c80319c"
    const remoteSpanId = "b7ad6b7169203331"
    const fixture = await createE2EFixture({
      caseID: "w3c-propagation",
      replies: [{ type: "text", text: "context propagated" }],
      pluginOptions: {
        traceparent: `00-${traceId}-${remoteSpanId}-01`,
        tracestate: "vendor=value",
        tracePropagationProviders: ["test"],
      },
    })
    try {
      const result = await fixture.run("propagate context")
      requireSuccess(result)
      expect(fixture.llm.pending()).toBe(0)
      expect(fixture.otlp.errors).toEqual([])
      const spans = await requireSpans(fixture, result, 3)

      const run = one(spans, "e2e.run")
      const interaction = one(spans, "e2e.interaction")
      const llm = one(spans, "e2e.llm")
      expect(run.traceId).toBe(traceId)
      expect(run.parentSpanId).toBe(remoteSpanId)
      expect(interaction.traceId).toBe(traceId)
      expect(interaction.parentSpanId).toBe(run.spanId)
      expect(llm.traceId).toBe(traceId)
      expect(llm.parentSpanId).toBe(interaction.spanId)

      const hit = fixture.llm.mainHits()[0]
      expect(hit).toBeDefined()
      expect(hit!.headers.get("traceparent")).toBe(`00-${traceId}-${llm.spanId}-01`)
      expect(hit!.headers.get("tracestate")).toBe("vendor=value")
    } finally {
      await fixture.close()
    }
  }, 60_000)

  test("exports tool execution and continuation spans", async () => {
    const fixture = await createE2EFixture({
      caseID: "tool-continuation",
      replies: [
        {
          type: "tool",
          name: "bash",
          input: { command: "echo e2e-tool-output", description: "Print deterministic output" },
          usage: { input: 5, output: 2 },
        },
        { type: "text", text: "tool completed", usage: { input: 4, output: 3 } },
      ],
    })
    try {
      const result = await fixture.run("use the bash tool", ["--dangerously-skip-permissions"])
      requireSuccess(result)
      expect(fixture.llm.pending()).toBe(0)
      expect(fixture.llm.mainHits()).toHaveLength(2)
      const spans = await requireSpans(fixture, result, 5)

      const run = one(spans, "e2e.run")
      const interaction = one(spans, "e2e.interaction")
      const tool = one(spans, "e2e.tool.bash")
      const llms = spans.filter(span => span.name === "e2e.llm")
      expect(llms).toHaveLength(2)
      expect(interaction.parentSpanId).toBe(run.spanId)
      expect(tool.parentSpanId).toBe(interaction.spanId)
      expect(llms.every(span => span.parentSpanId === interaction.spanId)).toBe(true)
      expect(new Set(spans.map(span => span.traceId)).size).toBe(1)
      expect(tool.attributes["tool.name"]).toBe("bash")
      expect(tool.attributes["tool.success"]).toBe(true)
      expect(String(tool.attributes["input.value"])).toContain("echo e2e-tool-output")
      expect(String(tool.attributes["output.value"])).toContain("e2e-tool-output")
      expect(run.attributes["run.total_tokens"]).toBe(14)
      expect(run.attributes["run.total_messages"]).toBe(2)
      expect(String(run.attributes["output.value"])).toContain("tool completed")
      expectOk(run)
      expectOk(interaction)
      expectOk(tool)
      llms.forEach(expectOk)
    } finally {
      await fixture.close()
    }
  }, 60_000)

  test("exports a foreground subtask trace with parent-child correlation", async () => {
    const fixture = await createE2EFixture({
      caseID: "foreground-subtask",
      replies: [
        {
          type: "tool",
          name: "task",
          input: {
            description: "Check delegated result",
            prompt: "Return the exact text: subtask completed",
            subagent_type: "explore",
          },
          usage: { input: 6, output: 2 },
        },
        { type: "text", text: "subtask completed", usage: { input: 4, output: 3 } },
        { type: "text", text: "parent received subtask", usage: { input: 5, output: 2 } },
      ],
    })
    try {
      const result = await fixture.run("delegate a subtask", ["--dangerously-skip-permissions"])
      requireSuccess(result)
      expect(fixture.llm.pending()).toBe(0)
      expect(fixture.llm.mainHits()).toHaveLength(3)
      const spans = await requireSpans(fixture, result, 8)

      const runs = spans.filter(span => span.name === "e2e.run")
      const interactions = spans.filter(span => span.name === "e2e.interaction")
      const llms = spans.filter(span => span.name === "e2e.llm")
      const task = one(spans, "e2e.tool.task")
      expect(runs).toHaveLength(2)
      expect(interactions).toHaveLength(2)
      expect(llms).toHaveLength(3)

      const parentRun = runs.find(span => span.attributes["session.is_subagent"] === false)
      const childRun = runs.find(span => span.attributes["session.is_subagent"] === true)
      const parentInteraction = interactions.find(span => span.attributes["session.is_subagent"] === false)
      const childInteraction = interactions.find(span => span.attributes["session.is_subagent"] === true)
      expect(parentRun).toBeDefined()
      expect(childRun).toBeDefined()
      expect(parentInteraction).toBeDefined()
      expect(childInteraction).toBeDefined()

      expect(parentInteraction!.parentSpanId).toBe(parentRun!.spanId)
      expect(task.parentSpanId).toBe(parentInteraction!.spanId)
      expect(childRun!.parentSpanId).toBe(task.spanId)
      expect(childInteraction!.parentSpanId).toBe(childRun!.spanId)
      expect(llms.filter(span => span.parentSpanId === parentInteraction!.spanId)).toHaveLength(2)
      expect(llms.filter(span => span.parentSpanId === childInteraction!.spanId)).toHaveLength(1)
      expect(new Set(spans.map(span => span.traceId)).size).toBe(1)

      expect(task.attributes["tool.name"]).toBe("task")
      expect(task.attributes["subagent.session.id"]).toBe(childRun!.attributes["session.id"])
      expect(task.attributes["subagent.agent.name"]).toBe("explore")
      expect(childRun!.attributes["agent.name"]).toBe("explore")
      expect(childRun!.attributes["session.parent_id"]).toBe(parentRun!.attributes["session.id"])
      expect(childRun!.attributes["task.call_id"]).toBe(task.attributes["tool.id"])
      expect(childInteraction!.attributes["output.value"]).toBe("subtask completed")
      expect(parentInteraction!.attributes["output.value"]).toBe("parent received subtask")
      expect(parentRun!.attributes["run.total_tokens"]).toBe(15)
      expect(childRun!.attributes["run.total_tokens"]).toBe(7)
      spans.forEach(expectOk)
    } finally {
      await fixture.close()
    }
  }, 60_000)

  test("loads without exporting when telemetry is disabled", async () => {
    const fixture = await createE2EFixture({
      caseID: "disabled",
      replies: [{ type: "text", text: "telemetry disabled" }],
      pluginOptions: { enabled: false },
    })
    try {
      const result = await fixture.run("do not trace this")
      requireSuccess(result)
      expect(fixture.llm.pending()).toBe(0)
      expect(fixture.otlp.payloads).toHaveLength(0)
      expect(fixture.otlp.spans()).toHaveLength(0)
    } finally {
      await fixture.close()
    }
  }, 60_000)
})
