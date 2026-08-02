import { describe, test, expect } from "bun:test"
import { context, SpanStatusCode, trace, TraceFlags } from "@opentelemetry/api"
import {
  AGENT_NAME,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_MODEL_NAME,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  LLM_TOKEN_COUNT_TOTAL,
  MimeType,
  OpenInferenceSpanKind,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  SemanticConventions,
  SESSION_ID,
  TOOL_NAME,
  USER_ID,
} from "@arizeai/openinference-semantic-conventions"
import type { Span } from "@opentelemetry/api"
import { handleSessionCreated, handleSessionIdle, handleSessionError, handleInteractionStarted } from "../../src/handlers/session.ts"
import { handleMessageUpdated, handleMessagePartUpdated, startMessageSpan } from "../../src/handlers/message.ts"
import { remoteParentContext } from "../../src/trace-context.ts"
import { makeCtx, makeTracer, type SpySpan } from "../helpers.ts"
import type {
  EventSessionCreated,
  EventSessionIdle,
  EventSessionError,
  EventMessageUpdated,
  EventMessagePartUpdated,
} from "@opencode-ai/sdk"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND

function makeSessionCreated(sessionID: string, createdAt = 1000, parentID?: string): EventSessionCreated {
  return {
    type: "session.created",
    properties: { info: { id: sessionID, projectID: "proj_test", directory: "/tmp", parentID, time: { created: createdAt } } },
  } as unknown as EventSessionCreated
}

function makeSessionIdle(sessionID: string): EventSessionIdle {
  return { type: "session.idle", properties: { sessionID } } as EventSessionIdle
}

function makeSessionError(sessionID?: string, error?: { name: string }): EventSessionError {
  return {
    type: "session.error",
    properties: { ...(sessionID !== undefined ? { sessionID } : {}), error },
  } as unknown as EventSessionError
}

function makeAssistantMessageUpdated(overrides: {
  id?: string
  parentID?: string
  sessionID?: string
  modelID?: string
  providerID?: string
  mode?: string
  cost?: number
  tokens?: { input: number; output: number; reasoning: number; cache: { read: number; write: number } }
  time?: { created: number; completed?: number }
  error?: { name: string }
  finish?: string
  summary?: boolean
}): EventMessageUpdated {
  return {
    type: "message.updated",
    properties: {
      info: {
        id: overrides.id ?? "msg_1",
        parentID: overrides.parentID ?? "user_1",
        role: "assistant",
        sessionID: overrides.sessionID ?? "ses_1",
        modelID: overrides.modelID ?? "claude-3-5-sonnet",
        providerID: overrides.providerID ?? "anthropic",
        mode: overrides.mode,
        cost: overrides.cost ?? 0.01,
        tokens: overrides.tokens ?? { input: 100, output: 50, reasoning: 0, cache: { read: 0, write: 0 } },
        time: overrides.time ?? { created: 1000, completed: 2000 },
        error: overrides.error,
        finish: overrides.finish,
        summary: overrides.summary,
      },
    },
  } as unknown as EventMessageUpdated
}

function makeToolPartUpdated(
  status: "running" | "completed" | "error",
  overrides: {
    sessionID?: string
    messageID?: string
    callID?: string
    tool?: string
    startMs?: number
    endMs?: number
    output?: string
    input?: Record<string, unknown>
    metadata?: Record<string, unknown>
  } = {},
): EventMessagePartUpdated {
  const sessionID = overrides.sessionID ?? "ses_1"
  const messageID = overrides.messageID ?? "msg_1"
  const callID = overrides.callID ?? "call_1"
  const start = overrides.startMs ?? 1000
  const end = overrides.endMs ?? 2000
  const input = overrides.input ?? {}
  const state =
    status === "running"
      ? { status: "running", input, time: { start }, ...(overrides.metadata ? { metadata: overrides.metadata } : {}) }
      : status === "completed"
        ? { status: "completed", input, time: { start, end }, output: overrides.output ?? "ok", title: "done", metadata: overrides.metadata ?? {} }
        : { status: "error", input, time: { start, end }, error: "fail", ...(overrides.metadata ? { metadata: overrides.metadata } : {}) }
  return {
    type: "message.part.updated",
    properties: { part: { type: "tool", sessionID, messageID, callID, tool: overrides.tool ?? "bash", state } },
  } as unknown as EventMessagePartUpdated
}

function makeTextPartUpdated(text: string, sessionID = "ses_1", messageID = "msg_1"): EventMessagePartUpdated {
  return {
    type: "message.part.updated",
    properties: { part: { type: "text", sessionID, messageID, text } },
  } as unknown as EventMessagePartUpdated
}

describe("run and interaction spans", () => {
  test("all span types include the resolved user ID", () => {
    const { ctx, tracer } = makeCtx("proj_test", [], { [USER_ID]: "user-1" })

    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 900, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)

    expect(tracer.spans.map((span) => span.attributes[USER_ID])).toEqual([
      "user-1",
      "user-1",
      "user-1",
      "user-1",
    ])
  })

  test("does not start a trace span on session.created", () => {
    const { ctx, tracer } = makeCtx()
    handleSessionCreated(makeSessionCreated("ses_1", 5000), ctx)
    expect(tracer.spans).toHaveLength(0)
  })

  test("subagent run span carries session attributes", () => {
    const { ctx, tracer } = makeCtx("proj_test", [], { team: "platform" })
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 900, ctx)
    handleSessionCreated(makeSessionCreated("ses_1", 1000, "ses_parent"), ctx)
    handleInteractionStarted("user_child", "ses_1", "review", "review", "anthropic/claude", 1100, ctx)
    expect(tracer.spans[2]!.attributes["session.id"]).toBe("ses_1")
    expect(tracer.spans[2]!.attributes[SESSION_ID]).toBe("ses_1")
    expect(tracer.spans[2]!.attributes["team"]).toBe("platform")
    expect(tracer.spans[2]!.attributes["agent.type"]).toBe("subagent")
    expect(tracer.spans[2]!.attributes["session.is_subagent"]).toBe(true)
  })

  test("run and interaction spans use distinct OpenInference kinds", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans[0]!.attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.CHAIN)
    expect(tracer.spans[1]!.attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.AGENT)
    expect(tracer.spans[1]!.attributes["opencode.model"]).toBe("anthropic/claude")
    expect(tracer.spans[1]!.attributes.model).toBeUndefined()
    expect(tracer.spans[0]!.attributes["opencode.run.id"]).toBe(tracer.spans[0]!.spanContext().spanId)
    expect(tracer.spans[1]!.attributes["opencode.interaction.id"]).toBe("user_1")
    expect(tracer.spans[0]!.attributes[AGENT_NAME]).toBe("build")
    expect(tracer.spans[1]!.parentSpan).toBe(tracer.spans[0])
  })

  test("run span carries is_subagent=false for root session", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_root", "ses_root", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans[0]!.attributes["session.is_subagent"]).toBe(false)
  })

  test("run span is parented to injected remote context", () => {
    const { ctx, tracer } = makeCtx()
    const rootContext = remoteParentContext("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01", undefined)
    expect(rootContext).toBeDefined()
    ctx.rootContext = () => rootContext!
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans[0]!.parentSpanContext?.traceId).toBe("0af7651916cd43dd8448eb211c80319c")
    expect(tracer.spans[0]!.parentSpanContext?.spanId).toBe("b7ad6b7169203331")
  })

  test("run span resolves root context at span creation", () => {
    const { ctx, tracer } = makeCtx()
    let rootContext = context.active()
    ctx.rootContext = () => rootContext
    rootContext = trace.setSpanContext(context.active(), {
      traceId: "4bf92f3577b34da6a3ce929d0e0e4736",
      spanId: "00f067aa0ba902b7",
      traceFlags: TraceFlags.SAMPLED,
    })
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans[0]!.parentSpanContext?.traceId).toBe("4bf92f3577b34da6a3ce929d0e0e4736")
    expect(tracer.spans[0]!.parentSpanContext?.spanId).toBe("00f067aa0ba902b7")
  })

  test("ends run span with OK status on session.idle", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    const span = tracer.spans[0]!
    expect(span.ended).toBe(true)
    expect(span.status.code).toBe(SpanStatusCode.OK)
    expect(ctx.activeRunSpans.has("ses_1")).toBe(false)
    expect(ctx.interactionSpans.has("user_1")).toBe(false)
    expect(tracer.spans[1]!.ended).toBe(true)
    expect(tracer.spans[1]!.status.code).toBe(SpanStatusCode.OK)
  })

  test("sets run total attributes before ending on idle", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    Object.assign(ctx.activeRunSpans.get("ses_1")!, { tokens: 250, cost: 0.05, messages: 3 })
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    const span = tracer.spans[0]!
    expect(span.attributes["run.total_tokens"]).toBe(250)
    expect(span.attributes["run.total_cost_usd"]).toBe(0.05)
    expect(span.attributes["run.total_messages"]).toBe(3)
    expect(span.attributes["session.total_tokens"]).toBeUndefined()
    expect(span.attributes[AGENT_NAME]).toBe("build")
    expect(span.attributes["agent.type"]).toBe("primary")
  })

  test("ends run span with ERROR status on session.error", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionError(makeSessionError("ses_1", { name: "NetworkError" }), ctx)
    const span = tracer.spans[0]!
    expect(span.ended).toBe(true)
    expect(span.status.code).toBe(SpanStatusCode.ERROR)
    expect(ctx.activeRunSpans.has("ses_1")).toBe(false)
    expect(ctx.interactionSpans.has("user_1")).toBe(false)
    expect(tracer.spans[1]!.ended).toBe(true)
    expect(tracer.spans[1]!.status.code).toBe(SpanStatusCode.ERROR)
    expect(tracer.spans[0]!.attributes[INPUT_MIME_TYPE]).toBe(MimeType.JSON)
    expect(JSON.parse(tracer.spans[0]!.attributes[INPUT_VALUE] as string)).toEqual(["prompt"])
    expect(tracer.spans[0]!.attributes[OUTPUT_VALUE]).toBeUndefined()
    expect(tracer.spans[0]!.attributes[OUTPUT_MIME_TYPE]).toBeUndefined()
  })

  test("error message is propagated to run span status", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionError(makeSessionError("ses_1", { name: "TimeoutError" }), ctx)
    expect(tracer.spans[0]!.status.message).toBe("TimeoutError")
  })

  test("idle on unknown session does not throw and creates no span", () => {
    const { ctx, tracer } = makeCtx()
    expect(() => handleSessionIdle(makeSessionIdle("ses_unknown"), ctx)).not.toThrow()
    expect(tracer.spans).toHaveLength(0)
  })

  test("session.error with undefined sessionID does not end any span", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionError(makeSessionError(undefined, { name: "UnknownError" }), ctx)
    expect(ctx.interactionSpans.has("user_1")).toBe(true)
    expect(tracer.spans[0]!.ended).toBe(false)
    expect(tracer.spans[1]!.ended).toBe(false)
  })

  test("subagent run is parented to the parent interaction", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionCreated(makeSessionCreated("ses_child", 2000, "ses_parent"), ctx)
    handleInteractionStarted("user_child", "ses_child", "review", "review", "anthropic/claude", 2100, ctx)
    expect(tracer.spans).toHaveLength(4)
    expect(tracer.spans[2]!.name).toBe("opencode.run")
    expect(tracer.spans[2]!.parentSpan).toBe(tracer.spans[1])
    expect(tracer.spans[3]!.name).toBe("opencode.interaction")
    expect(tracer.spans[3]!.parentSpan).toBe(tracer.spans[2])
  })

  test("foreground task parents the child interaction span", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      sessionID: "ses_parent",
      callID: "call_task",
      tool: "task",
      input: { subagent_type: "review" },
    }), ctx)
    handleSessionCreated(makeSessionCreated("ses_child", 1100, "ses_parent"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      sessionID: "ses_parent",
      callID: "call_task",
      tool: "task",
      input: { subagent_type: "review" },
      metadata: { parentSessionId: "ses_parent", sessionId: "ses_child" },
    }), ctx)
    const details = ctx.pendingSubagentRuns.get("ses_child")!
    ctx.pendingSubagentRuns.delete("ses_child")
    handleInteractionStarted("user_child", "ses_child", "review", "child prompt", "anthropic/claude", 1200, ctx, details)

    expect(tracer.spans).toHaveLength(5)
    expect(tracer.spans[2]!.name).toBe("opencode.tool.task")
    expect(tracer.spans[3]!.name).toBe("opencode.run")
    expect(tracer.spans[3]!.parentSpanContext?.spanId).toBe(tracer.spans[2]!.spanContext().spanId)
    expect(tracer.spans[4]!.name).toBe("opencode.interaction")
    expect(tracer.spans[4]!.parentSpan).toBe(tracer.spans[3])
    expect(tracer.spans[3]!.attributes["session.parent_id"]).toBe("ses_parent")
    expect(tracer.spans[3]!.attributes["task.call_id"]).toBe("call_task")
  })

  test("task metadata correlates a resumed foreground subagent without session.created", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      sessionID: "ses_parent",
      callID: "call_resume",
      tool: "task",
      input: { subagent_type: "review", task_id: "ses_existing" },
      metadata: { parentSessionId: "ses_parent", sessionId: "ses_existing" },
    }), ctx)
    const details = ctx.pendingSubagentRuns.get("ses_existing")!
    handleInteractionStarted("user_child", "ses_existing", "review", "resume", "anthropic/claude", 1200, ctx, details)

    expect(tracer.spans[3]!.parentSpanContext?.spanId).toBe(tracer.spans[2]!.spanContext().spanId)
    expect(tracer.spans[3]!.attributes["task.call_id"]).toBe("call_resume")
    expect(tracer.spans[4]!.parentSpan).toBe(tracer.spans[3])
  })

  test("parallel foreground tasks correlate each child interaction to its task span", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 1000, ctx)
    for (const [callID, childSessionID] of [["call_one", "ses_one"], ["call_two", "ses_two"]] as const) {
      handleMessagePartUpdated(makeToolPartUpdated("running", {
        sessionID: "ses_parent",
        callID,
        tool: "task",
        input: { subagent_type: "review" },
        metadata: { parentSessionId: "ses_parent", sessionId: childSessionID },
      }), ctx)
    }
    handleInteractionStarted(
      "user_one",
      "ses_one",
      "review",
      "one",
      "anthropic/claude",
      1100,
      ctx,
      ctx.pendingSubagentRuns.get("ses_one")!,
    )
    handleInteractionStarted(
      "user_two",
      "ses_two",
      "review",
      "two",
      "anthropic/claude",
      1200,
      ctx,
      ctx.pendingSubagentRuns.get("ses_two")!,
    )

    expect(tracer.spans[4]!.parentSpanContext?.spanId).toBe(tracer.spans[2]!.spanContext().spanId)
    expect(tracer.spans[5]!.parentSpan).toBe(tracer.spans[4])
    expect(tracer.spans[6]!.parentSpanContext?.spanId).toBe(tracer.spans[3]!.spanContext().spanId)
    expect(tracer.spans[7]!.parentSpan).toBe(tracer.spans[6])
  })

  test("subagent span falls back to a root trace when parent run is absent", () => {
    const { ctx, tracer } = makeCtx()
    handleSessionCreated(makeSessionCreated("ses_child", 1000, "ses_missing_parent"), ctx)
    expect(() => handleInteractionStarted("user_child", "ses_child", "review", "review", "anthropic/claude", 1100, ctx)).not.toThrow()
    expect(tracer.spans).toHaveLength(2)
    expect(tracer.spans[0]!.parentSpan).toBeUndefined()
    expect(tracer.spans[0]!.parentSpanContext).toBeUndefined()
    expect(tracer.spans[1]!.parentSpan).toBe(tracer.spans[0])
  })

  test("late child spans reuse the ended interaction trace context", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    startMessageSpan("ses_1", "msg_late", "user_1", "claude", "anthropic", 2000, ctx)
    expect(tracer.spans).toHaveLength(3)
    expect(tracer.spans[2]!.parentSpanContext).toBeDefined()
    expect(tracer.spans[2]!.parentSpanContext?.spanId).toBe(tracer.spans[1]!.spanContext().spanId)
    expect(tracer.spans[2]!.parentSpanContext?.traceId).toBe(tracer.spans[1]!.spanContext().traceId)
  })

  test("late child events stay on the previous interaction after a new run starts", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt one", "anthropic/claude", 1000, ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1100, ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    handleInteractionStarted("user_2", "ses_1", "build", "prompt two", "anthropic/claude", 2000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { sessionID: "ses_1", messageID: "msg_1", callID: "call_late" }), ctx)
    expect(tracer.spans).toHaveLength(6)
    expect(tracer.spans[5]!.name).toBe("opencode.tool.bash")
    expect(tracer.spans[5]!.parentSpanContext?.spanId).toBe(tracer.spans[1]!.spanContext().spanId)
    expect(tracer.spans[5]!.parentSpanContext?.spanId).not.toBe(tracer.spans[4]!.spanContext().spanId)
  })

  test("interaction span carries the final assistant output", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["llm"])
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessagePartUpdated(makeTextPartUpdated("final answer"), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1", parentID: "user_1", mode: "build" }), ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(tracer.spans[1]!.attributes[OUTPUT_VALUE]).toBe("final answer")
    expect(tracer.spans[1]!.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.TEXT)
  })

  test("interaction span ends at the terminal assistant completion time", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      parentID: "user_1",
      finish: "stop",
      time: { created: 1200, completed: 2400 },
    }), ctx)

    const interactionSpan = tracer.spans[1]!
    expect(interactionSpan.ended).toBe(false)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(interactionSpan.ended).toBe(true)
    expect(interactionSpan.endTime).toBe(2400)
    expect(interactionSpan.status.code).toBe(SpanStatusCode.OK)
    expect(interactionSpan.attributes["interaction.total_tokens"]).toBe(150)
    expect(ctx.interactionSpans.has("user_1")).toBe(false)
  })

  test("does not recreate an ended interaction when its start is replayed", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      parentID: "user_1",
      finish: "stop",
      time: { created: 1200, completed: 2400 },
    }), ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)

    handleInteractionStarted("user_1", "ses_1", "build", "", "anthropic/claude", 1000, ctx)

    expect(tracer.spans).toHaveLength(2)
    expect(tracer.spans.filter(span => span.name === "opencode.interaction")).toHaveLength(1)
    expect(ctx.interactionSpans.has("user_1")).toBe(false)
    expect(ctx.activeInteractions.has("ses_1")).toBe(false)
    expect(ctx.interactionTotals.has("user_1")).toBe(false)
  })

  test("empty duplicate updates do not overwrite queued interaction inputs", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "first", "anthropic/claude", 1000, ctx)
    handleInteractionStarted("user_2", "ses_1", "build", "second", "anthropic/claude", 1100, ctx)

    handleInteractionStarted("user_1", "ses_1", "build", "", "anthropic/claude", 1000, ctx)
    handleInteractionStarted("user_2", "ses_1", "build", "", "anthropic/claude", 1100, ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)

    expect(JSON.parse(tracer.spans[0]!.attributes[INPUT_VALUE] as string)).toEqual(["first", "second"])
  })

  test("a later non-empty update fills an interaction created without input", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "", "anthropic/claude", 1000, ctx)
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)

    expect(JSON.parse(tracer.spans[0]!.attributes[INPUT_VALUE] as string)).toEqual(["prompt"])
  })

  test("tool-call and compaction messages do not end the interaction", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_tool",
      parentID: "user_1",
      finish: "tool-calls",
      time: { created: 1200, completed: 2000 },
    }), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_compaction",
      parentID: "user_1",
      finish: "stop",
      summary: true,
      time: { created: 2100, completed: 3000 },
    }), ctx)

    expect(tracer.spans[1]!.ended).toBe(false)

    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_final",
      parentID: "user_1",
      finish: "stop",
      time: { created: 3100, completed: 4000 },
    }), ctx)

    expect(tracer.spans[1]!.ended).toBe(false)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(tracer.spans[1]!.ended).toBe(true)
    expect(tracer.spans[1]!.endTime).toBe(4000)
    expect(tracer.spans[1]!.attributes["interaction.total_messages"]).toBe(3)
  })

  test("interaction waits for the final assistant when a tool-calling message finishes with stop", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["llm"])
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessagePartUpdated(makeTextPartUpdated("checking", "ses_1", "msg_first"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      sessionID: "ses_1",
      messageID: "msg_first",
      callID: "call_1",
    }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", {
      sessionID: "ses_1",
      messageID: "msg_first",
      callID: "call_1",
      output: "done",
    }), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_first",
      parentID: "user_1",
      finish: "stop",
      time: { created: 1200, completed: 2000 },
    }), ctx)

    expect(tracer.spans[1]!.ended).toBe(false)

    handleMessagePartUpdated(makeTextPartUpdated("final answer", "ses_1", "msg_final"), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_final",
      parentID: "user_1",
      finish: "stop",
      time: { created: 2100, completed: 3000 },
    }), ctx)

    expect(tracer.spans[1]!.ended).toBe(false)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)

    expect(tracer.spans[1]!.ended).toBe(true)
    expect(tracer.spans[1]!.endTime).toBe(3000)
    expect(tracer.spans[1]!.attributes[OUTPUT_VALUE]).toBe("final answer")
  })

  test("interaction waits for a late errored assistant completion after session termination", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1200, ctx)
    handleMessagePartUpdated(makeTextPartUpdated("partial output"), ctx)

    handleSessionError(makeSessionError("ses_1", { name: "NetworkError" }), ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)

    expect(tracer.spans[1]!.ended).toBe(false)

    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_1",
      parentID: "user_1",
      error: { name: "NetworkError" },
      time: { created: 1200, completed: 2500 },
    }), ctx)

    expect(tracer.spans[1]!.ended).toBe(true)
    expect(tracer.spans[1]!.endTime).toBe(2500)
    expect(tracer.spans[1]!.status).toEqual({ code: SpanStatusCode.ERROR, message: "NetworkError" })
    expect(tracer.spans[1]!.attributes[OUTPUT_VALUE]).toBe("partial output")
    expect(ctx.pendingAssistantInteractions.size).toBe(0)
  })

  test("interaction ends from a successful assistant completion that arrives after idle", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1200, ctx)
    handleMessagePartUpdated(makeTextPartUpdated("final answer"), ctx)

    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(tracer.spans[1]!.ended).toBe(false)

    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_1",
      parentID: "user_1",
      finish: "stop",
      time: { created: 1200, completed: 2500 },
    }), ctx)

    expect(tracer.spans[1]!.ended).toBe(true)
    expect(tracer.spans[1]!.endTime).toBe(2500)
    expect(tracer.spans[1]!.status.code).toBe(SpanStatusCode.OK)
    expect(tracer.spans[1]!.attributes[OUTPUT_VALUE]).toBe("final answer")
    expect(ctx.interactionCompletions.size).toBe(0)
  })

  test("subagent interaction span carries the final assistant output", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["llm"])
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleSessionCreated(makeSessionCreated("ses_child", 1100, "ses_parent"), ctx)
    handleInteractionStarted("user_child", "ses_child", "review", "review", "anthropic/claude", 1200, ctx)
    handleMessagePartUpdated(makeTextPartUpdated("subagent result", "ses_child"), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_1",
      parentID: "user_child",
      sessionID: "ses_child",
      mode: "review",
    }), ctx)
    handleSessionIdle(makeSessionIdle("ses_child"), ctx)
    expect(tracer.spans[3]!.attributes[OUTPUT_VALUE]).toBe("subagent result")
    expect(tracer.spans[3]!.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.TEXT)
  })

  test("queued interactions share one run and retain distinct completion times", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "first", "anthropic/claude", 1000, ctx)
    handleInteractionStarted("user_2", "ses_1", "build", "second", "anthropic/claude", 1100, ctx)
    handleMessagePartUpdated(makeTextPartUpdated("first response", "ses_1", "msg_1"), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_1",
      parentID: "user_1",
      cost: 0.01,
      finish: "stop",
      time: { created: 1200, completed: 2000 },
    }), ctx)
    handleMessagePartUpdated(makeTextPartUpdated("second response", "ses_1", "msg_2"), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({
      id: "msg_2",
      parentID: "user_2",
      cost: 0.02,
      finish: "stop",
      time: { created: 2100, completed: 3000 },
      tokens: { input: 200, output: 100, reasoning: 0, cache: { read: 0, write: 0 } },
    }), ctx)

    expect(tracer.spans).toHaveLength(3)
    expect(tracer.spans[1]!.parentSpan).toBe(tracer.spans[0])
    expect(tracer.spans[2]!.parentSpan).toBe(tracer.spans[0])
    expect(tracer.spans[0]!.ended).toBe(false)
    expect(tracer.spans[1]!.ended).toBe(false)
    expect(tracer.spans[2]!.ended).toBe(false)

    handleSessionIdle(makeSessionIdle("ses_1"), ctx)

    expect(tracer.spans.every((span) => span.ended)).toBe(true)
    expect(tracer.spans[1]!.endTime).toBe(2000)
    expect(tracer.spans[2]!.endTime).toBe(3000)
    expect(tracer.spans[0]!.attributes["run.total_interactions"]).toBe(2)
    expect(tracer.spans[0]!.attributes["run.total_tokens"]).toBe(450)
    expect(tracer.spans[0]!.attributes["run.total_cost_usd"]).toBe(0.03)
    expect(tracer.spans[0]!.attributes["run.total_messages"]).toBe(2)
    expect(tracer.spans[0]!.attributes["interaction.count"]).toBeUndefined()
    expect(tracer.spans[0]!.attributes[INPUT_MIME_TYPE]).toBe(MimeType.JSON)
    expect(tracer.spans[0]!.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.TEXT)
    expect(JSON.parse(tracer.spans[0]!.attributes[INPUT_VALUE] as string)).toEqual(["first", "second"])
    expect(tracer.spans[0]!.attributes[OUTPUT_VALUE]).toBe("second response")
    expect(tracer.spans[1]!.attributes["interaction.total_tokens"]).toBe(150)
    expect(tracer.spans[1]!.attributes["interaction.total_cost_usd"]).toBe(0.01)
    expect(tracer.spans[2]!.attributes["interaction.total_tokens"]).toBe(300)
    expect(tracer.spans[2]!.attributes["interaction.total_cost_usd"]).toBe(0.02)
    expect(ctx.activeRunSpans.size).toBe(0)
    expect(ctx.interactionSpans.size).toBe(0)
    expect(ctx.interactionTotals.size).toBe(0)
  })
})

describe("tool spans", () => {
  test("starts a tool span on running status", () => {
    const { ctx, tracer } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running", { startMs: 1000 }), ctx)
    expect(tracer.spans).toHaveLength(1)
    expect(tracer.spans[0]!.name).toBe("opencode.tool.bash")
    expect(tracer.spans[0]!.startTime).toBe(1000)
    expect(ctx.pendingToolSpans.has("ses_1:call_1")).toBe(true)
  })

  test("reuses the task span when foreground metadata arrives", () => {
    const { ctx, tracer } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      tool: "task",
      input: { subagent_type: "review" },
    }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      tool: "task",
      input: { subagent_type: "review" },
      metadata: { parentSessionId: "ses_1", sessionId: "ses_child" },
    }), ctx)

    expect(tracer.spans).toHaveLength(1)
    expect(ctx.pendingToolSpans.size).toBe(1)
    expect(ctx.pendingSubagentRuns.get("ses_child")?.taskSpanContext?.spanId)
      .toBe(tracer.spans[0]!.spanContext().spanId)
    expect(tracer.spans[0]!.attributes["subagent.session.id"]).toBe("ses_child")
    expect(tracer.spans[0]!.attributes["subagent.agent.name"]).toBe("review")
  })

  test("clears foreground subagent correlation when the task fails", () => {
    const { ctx } = makeCtx()
    const metadata = { parentSessionId: "ses_1", sessionId: "ses_child" }
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      tool: "task",
      input: { subagent_type: "review" },
      metadata,
    }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("error", {
      tool: "task",
      input: { subagent_type: "review" },
      metadata,
    }), ctx)

    expect(ctx.pendingSubagentRuns.has("ses_child")).toBe(false)
  })

  test("does not retain background task correlation", () => {
    const { ctx } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      tool: "task",
      input: { subagent_type: "review" },
      metadata: { parentSessionId: "ses_1", sessionId: "ses_child", background: true },
    }), ctx)

    expect(ctx.pendingSubagentRuns.has("ses_child")).toBe(false)
  })

  test("tool span carries tool.name attribute", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { tool: "read_file" }), ctx)
    expect(tracer.spans[2]!.attributes["tool.name"]).toBe("read_file")
    expect(tracer.spans[2]!.attributes[TOOL_NAME]).toBe("read_file")
    expect(tracer.spans[2]!.attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.TOOL)
    expect(tracer.spans[2]!.attributes[AGENT_NAME]).toBe("build")
    expect(tracer.spans[2]!.attributes["agent.type"]).toBe("primary")
  })

  test("ends tool span with OK status on completion", () => {
    const { ctx, tracer } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", { endMs: 2000 }), ctx)
    const span = tracer.spans[0]!
    expect(span.ended).toBe(true)
    expect(span.status.code).toBe(SpanStatusCode.OK)
    expect(span.endTime).toBe(2000)
  })

  test("ends tool span with ERROR status on error", () => {
    const { ctx, tracer } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("error"), ctx)
    const span = tracer.spans[0]!
    expect(span.ended).toBe(true)
    expect(span.status.code).toBe(SpanStatusCode.ERROR)
  })

  test("tool span result_size_bytes matches exact byte length of multibyte output", () => {
    const { ctx, tracer } = makeCtx()
    const multibyte = "こんにちは"
    const expectedBytes = Buffer.byteLength(multibyte, "utf8")
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", { output: multibyte }), ctx)
    expect(tracer.spans[0]!.attributes["tool.result_size_bytes"]).toBe(expectedBytes)
  })

  test("tool span error attr set on error status", () => {
    const { ctx, tracer } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("error"), ctx)
    expect(tracer.spans[0]!.attributes["tool.error"]).toBe("fail")
  })

  test("tool span removed from pendingToolSpans after completion", () => {
    const { ctx } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed"), ctx)
    expect(ctx.pendingToolSpans.size).toBe(0)
  })

  test("tool span started even when completed arrives without prior running (out-of-order)", () => {
    const { ctx, tracer } = makeCtx()
    handleMessagePartUpdated(makeToolPartUpdated("completed", { startMs: 500, endMs: 1500 }), ctx)
    expect(tracer.spans).toHaveLength(1)
    expect(tracer.spans[0]!.ended).toBe(true)
    expect(tracer.spans[0]!.status.code).toBe(SpanStatusCode.OK)
  })

  test("tool span is parented to the active interaction when available", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 900, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { sessionID: "ses_1" }), ctx)
    expect(tracer.spans).toHaveLength(3)
    expect(tracer.spans[2]!.name).toBe("opencode.tool.bash")
    expect(tracer.spans[2]!.parentSpan).toBe(tracer.spans[1])
  })

  test("out-of-order tool span is parented to the active interaction when available", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 900, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", { sessionID: "ses_1", startMs: 500, endMs: 1500 }), ctx)
    expect(tracer.spans).toHaveLength(3)
    expect(tracer.spans[2]!.name).toBe("opencode.tool.bash")
    expect(tracer.spans[2]!.parentSpan).toBe(tracer.spans[1])
  })
})

describe("message (LLM) spans", () => {
  test("startMessageSpan creates an llm span", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx, "build")
    expect(tracer.spans).toHaveLength(1)
    expect(tracer.spans[0]!.name).toBe("opencode.llm")
    expect(ctx.messageSpans.has("ses_1:msg_1")).toBe(true)
    expect(ctx.llmRequestContexts.get("ses_1:user_1")?.[0]).toMatchObject({
      messageID: "msg_1",
      agent: "build",
      modelID: "claude-3-5-sonnet",
      providerID: "anthropic",
    })
  })

  test("startMessageSpan sets OpenInference LLM attributes", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-sonnet-4", "amazon-bedrock", 1000, ctx)
    expect(tracer.spans[0]!.attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.LLM)
    expect(tracer.spans[0]!.attributes[LLM_SYSTEM]).toBe("amazon-bedrock")
    expect(tracer.spans[0]!.attributes[LLM_PROVIDER]).toBe("amazon-bedrock")
    expect(tracer.spans[0]!.attributes["gen_ai.provider.name"]).toBe("aws.bedrock")
    expect(tracer.spans[0]!.attributes[LLM_MODEL_NAME]).toBe("claude-sonnet-4")
    expect(tracer.spans[0]!.attributes[OUTPUT_VALUE]).toBe("")
    expect(tracer.spans[0]!.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.TEXT)
  })

  test("startMessageSpan uses the assistant message agent", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "gpt-4o", "openai", 1000, ctx, "build")
    expect(tracer.spans[0]!.attributes[AGENT_NAME]).toBe("build")
  })

  test("startMessageSpan is a no-op when span already exists for sessionID:messageID", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)
    expect(tracer.spans).toHaveLength(1)
  })

  test("handleMessageUpdated ends message span on completion", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1", time: { created: 1000, completed: 2000 } }), ctx)
    const span = tracer.spans[0]!
    expect(span.ended).toBe(true)
    expect(span.endTime).toBe(2000)
    expect(ctx.messageSpans.has("ses_1:msg_1")).toBe(false)
    expect(ctx.llmRequestContexts.has("ses_1:user_1")).toBe(false)
  })

  test("llm span ends at the final tool handoff instead of tool completion", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { callID: "call_1", startMs: 1400 }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { callID: "call_2", startMs: 1600 }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", { callID: "call_1", startMs: 1400, endMs: 3000 }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", { callID: "call_2", startMs: 1600, endMs: 3500 }), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ time: { created: 1000, completed: 3600 } }), ctx)

    const llmSpan = tracer.spans.find((span) => span.name === "opencode.llm")!
    expect(llmSpan.endTime).toBe(1600)
    expect(llmSpan.attributes.duration_ms).toBe(600)
    expect(tracer.spans.filter((span) => span.name === "opencode.tool.bash").map((span) => span.endTime)).toEqual([3000, 3500])
  })

  test("restamped running updates do not move the llm handoff time", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { startMs: 1400 }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { startMs: 2990 }), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("completed", { startMs: 2990, endMs: 3000 }), ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ time: { created: 1000, completed: 3100 } }), ctx)

    const llmSpan = tracer.spans.find((span) => span.name === "opencode.llm")!
    const toolSpan = tracer.spans.find((span) => span.name === "opencode.tool.bash")!
    expect(llmSpan.endTime).toBe(1400)
    expect(llmSpan.attributes.duration_ms).toBe(400)
    expect(toolSpan.startTime).toBe(1400)
    expect(toolSpan.endTime).toBe(3000)
  })

  test("an older completion does not remove a newer request context", () => {
    const { ctx } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)
    startMessageSpan("ses_1", "msg_2", "user_1", "claude", "anthropic", 1500, ctx)

    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1", time: { created: 1000, completed: 2000 } }), ctx)

    expect(ctx.llmRequestContexts.get("ses_1:user_1")?.map(request => request.messageID)).toEqual(["msg_2"])
  })

  test("handleMessageUpdated sets OK status on success", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1" }), ctx)
    expect(tracer.spans[0]!.status.code).toBe(SpanStatusCode.OK)
  })

  test("handleMessageUpdated sets ERROR status on api error", () => {
    const { ctx, tracer } = makeCtx()
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1", error: { name: "RateLimitError" } }), ctx)
    expect(tracer.spans[0]!.status.code).toBe(SpanStatusCode.ERROR)
    expect(tracer.spans[0]!.status.message).toBe("RateLimitError")
  })

  test("handleMessageUpdated sets OpenInference token attributes on span", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "review", "prompt", "anthropic/claude", 900, ctx, { agentType: "subagent" })
    startMessageSpan("ses_1", "msg_1", "user_1", "claude-3-5-sonnet", "anthropic", 1000, ctx)
    handleMessageUpdated(
      makeAssistantMessageUpdated({
        id: "msg_1",
        tokens: { input: 200, output: 80, reasoning: 10, cache: { read: 30, write: 5 } },
      }),
      ctx,
    )
    const span = tracer.spans[2]!
    expect(span.attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(235)
    expect(span.attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(90)
    expect(span.attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]).toBe(10)
    expect(span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(30)
    expect(span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(5)
    expect(span.attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(325)
    expect(span.attributes[AGENT_NAME]).toBe("review")
    expect(span.attributes["agent.type"]).toBe("subagent")
  })

  test("handleMessageUpdated replaces an unknown agent from the assistant message", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "unknown", "prompt", "openai/gpt-4o", 900, ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "gpt-4o", "openai", 1000, ctx)
    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1", mode: "build" }), ctx)
    expect(tracer.spans[2]!.attributes[AGENT_NAME]).toBe("build")
    expect(ctx.activeRunSpans.get("ses_1")!.agent).toBe("build")
  })

  test("handleMessageUpdated no-ops span handling when no span exists for messageID", () => {
    const { ctx, tracer } = makeCtx()
    const spansBefore = tracer.spans.length
    const mapSizeBefore = ctx.messageSpans.size
    handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_no_span" }), ctx)
    expect(tracer.spans).toHaveLength(spansBefore)
    expect(ctx.messageSpans.size).toBe(mapSizeBefore)
  })

  test("message span is parented to the active interaction when available", () => {
    const { ctx, tracer } = makeCtx()
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 900, ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)
    expect(tracer.spans).toHaveLength(3)
    expect(tracer.spans[2]!.name).toBe("opencode.llm")
    expect(tracer.spans[2]!.parentSpan).toBe(tracer.spans[1])
  })
})

describe("orphaned span cleanup", () => {
  test("pending tool spans are ended with ERROR on session.idle", () => {
    const { ctx, tracer } = makeCtx()
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { sessionID: "ses_1" }), ctx)
    expect(ctx.pendingToolSpans.size).toBe(1)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(ctx.pendingToolSpans.size).toBe(0)
    const toolSpan = tracer.spans.find(s => s.name.startsWith("opencode.tool"))!
    expect(toolSpan.ended).toBe(true)
    expect(toolSpan.status.code).toBe(SpanStatusCode.ERROR)
  })

  test("pending tool spans for other sessions are not swept", () => {
    const { ctx } = makeCtx()
    const t = makeTracer()
    const span = t.startSpan("tool") as unknown as Span
    ctx.pendingToolSpans.set("ses_other:call_1", { tool: "bash", sessionID: "ses_other", startMs: 0, span })
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(ctx.pendingToolSpans.has("ses_other:call_1")).toBe(true)
  })

  test("pending tool spans are ended with ERROR on session.error", () => {
    const { ctx, tracer } = makeCtx()
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", { sessionID: "ses_1" }), ctx)
    handleSessionError(makeSessionError("ses_1"), ctx)
    expect(ctx.pendingToolSpans.size).toBe(0)
    const toolSpan = tracer.spans.find(s => s.name.startsWith("opencode.tool"))!
    expect(toolSpan.ended).toBe(true)
    expect(toolSpan.status.code).toBe(SpanStatusCode.ERROR)
  })

  test("pending message spans are ended with ERROR on session.idle", () => {
    const { ctx, tracer } = makeCtx()
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    startMessageSpan("ses_1", "msg_orphan", "user_1", "claude", "anthropic", 1000, ctx)
    handleSessionIdle(makeSessionIdle("ses_1"), ctx)
    expect(ctx.messageSpans.has("ses_1:msg_orphan")).toBe(false)
    expect(ctx.llmRequestContexts.has("ses_1:user_1")).toBe(false)
    const msgSpan = tracer.spans.find(s => s.name === "opencode.llm")!
    expect(msgSpan.ended).toBe(true)
    expect(msgSpan.status.code).toBe(SpanStatusCode.ERROR)
  })

  test("pending message spans are ended with ERROR on session.error", () => {
    const { ctx, tracer } = makeCtx()
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    startMessageSpan("ses_1", "msg_orphan", "user_1", "claude", "anthropic", 1000, ctx)
    handleSessionError(makeSessionError("ses_1"), ctx)
    expect(ctx.messageSpans.has("ses_1:msg_orphan")).toBe(false)
    const msgSpan = tracer.spans.find(s => s.name === "opencode.llm")!
    expect(msgSpan.ended).toBe(true)
    expect(msgSpan.status.code).toBe(SpanStatusCode.ERROR)
  })
})

describe("OPENCODE_DISABLE_TRACES=session", () => {
  test("session traces are not started", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["session"])
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans).toHaveLength(0)
  })

  test("session.idle does not throw when no run span exists", () => {
    const { ctx } = makeCtx("proj_test", ["session"])
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    expect(() => handleSessionIdle(makeSessionIdle("ses_1"), ctx)).not.toThrow()
  })

  test("session.error does not throw when no run span exists", () => {
    const { ctx } = makeCtx("proj_test", ["session"])
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    expect(() => handleSessionError(makeSessionError("ses_1"), ctx)).not.toThrow()
  })

  test("llm spans become root spans (no parent) when session traces disabled but llm enabled", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["session"])
    handleSessionCreated(makeSessionCreated("ses_1"), ctx)
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)
    expect(tracer.spans).toHaveLength(1)
    expect(tracer.spans[0]!.name).toBe("opencode.llm")
    expect(tracer.spans[0]!.parentSpan).toBeUndefined()
  })
})

describe("OPENCODE_DISABLE_TRACES=llm", () => {
  test("startMessageSpan is a no-op", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["llm"])
    startMessageSpan("ses_1", "msg_1", "user_1", "claude", "anthropic", 1000, ctx)
    expect(tracer.spans).toHaveLength(0)
    expect(ctx.messageSpans.has("msg_1")).toBe(false)
  })

  test("handleMessageUpdated does not throw when no message span exists", () => {
    const { ctx } = makeCtx("proj_test", ["llm"])
    expect(() => handleMessageUpdated(makeAssistantMessageUpdated({ id: "msg_1" }), ctx)).not.toThrow()
  })

  test("session traces remain enabled", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["llm"])
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans).toHaveLength(2)
  })
})

describe("OPENCODE_DISABLE_TRACES=tool", () => {
  test("no tool span started on running status", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["tool"])
    handleMessagePartUpdated(makeToolPartUpdated("running"), ctx)
    expect(tracer.spans).toHaveLength(0)
  })

  test("pending tool state remains available for task correlation", () => {
    const { ctx } = makeCtx("proj_test", ["tool"])
    handleMessagePartUpdated(makeToolPartUpdated("running", { startMs: 1000 }), ctx)
    expect(ctx.pendingToolSpans.has("ses_1:call_1")).toBe(true)
    expect(ctx.pendingToolSpans.get("ses_1:call_1")!.startMs).toBe(1000)
    expect(ctx.pendingToolSpans.get("ses_1:call_1")!.span).toBeUndefined()
  })

  test("subagent interaction falls back to the parent interaction", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["tool"])
    handleInteractionStarted("user_parent", "ses_parent", "build", "prompt", "anthropic/claude", 1000, ctx)
    handleMessagePartUpdated(makeToolPartUpdated("running", {
      sessionID: "ses_parent",
      tool: "task",
      input: { subagent_type: "review" },
      metadata: { parentSessionId: "ses_parent", sessionId: "ses_child" },
    }), ctx)
    handleInteractionStarted(
      "user_child",
      "ses_child",
      "review",
      "child",
      "anthropic/claude",
      1100,
      ctx,
      ctx.pendingSubagentRuns.get("ses_child")!,
    )

    expect(tracer.spans).toHaveLength(4)
    expect(tracer.spans[2]!.parentSpan).toBe(tracer.spans[1])
    expect(tracer.spans[3]!.parentSpan).toBe(tracer.spans[2])
  })

  test("no tool span created for out-of-order completed event", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["tool"])
    handleMessagePartUpdated(makeToolPartUpdated("completed", { startMs: 500, endMs: 1500 }), ctx)
    expect(tracer.spans).toHaveLength(0)
  })

  test("session traces remain enabled", () => {
    const { ctx, tracer } = makeCtx("proj_test", ["tool"])
    handleInteractionStarted("user_1", "ses_1", "build", "prompt", "anthropic/claude", 1000, ctx)
    expect(tracer.spans).toHaveLength(2)
  })
})
