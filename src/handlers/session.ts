import { SeverityNumber } from "@opentelemetry/api-logs"
import { SpanStatusCode, trace } from "@opentelemetry/api"
import type { EventSessionCreated, EventSessionIdle, EventSessionError, EventSessionStatus } from "@opencode-ai/sdk"
import {
  AGENT_NAME,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  MESSAGE_CONTENT,
  MESSAGE_ROLE,
  MimeType,
  OpenInferenceSpanKind,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  SemanticConventions,
  SESSION_ID,
} from "@arizeai/openinference-semantic-conventions"
import {
  agentAttrs,
  errorSummary,
  getSessionAgentMeta,
  setBoundedMap,
  isMetricEnabled,
  isTraceEnabled,
  resolveSessionTraceContext,
} from "../util.ts"
import type { ActiveRunSpan, HandlerContext, RunDetails, SessionAgentType } from "../types.ts"
import { endInteractionSpan } from "../interaction.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND

function setRunIOAttributes(run: ActiveRunSpan) {
  const interactions = [...run.interactionIO.values()]
  const output = interactions.at(-1)?.output
  run.span.setAttributes({
    [INPUT_VALUE]: JSON.stringify(interactions.map(({ input }) => input)),
    [INPUT_MIME_TYPE]: MimeType.JSON,
    ...(output !== undefined
      ? {
          [OUTPUT_VALUE]: output,
          [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        }
      : {}),
  })
}

function ensureRunStarted(
  sessionID: string,
  agent: string,
  startTime: number,
  ctx: HandlerContext,
  details?: RunDetails,
) {
  if (!isTraceEnabled("session", ctx)) return
  const parentSessionID = details?.parentSessionID ?? ctx.sessionParents.get(sessionID)
  const agentType: SessionAgentType = details?.agentType ?? (parentSessionID ? "subagent" : "primary")
  const isSubagent = agentType === "subagent"
  const existing = ctx.activeRunSpans.get(sessionID)
  if (existing) {
    existing.span.setAttributes({
      ...(agent !== "unknown" ? { [AGENT_NAME]: agent } : {}),
      "agent.type": agentType,
      "session.is_subagent": isSubagent,
      ...(parentSessionID ? { "session.parent_id": parentSessionID } : {}),
      ...(details?.taskCallID ? { "task.call_id": details.taskCallID } : {}),
    })
    return existing
  }

  const parentContext = details?.taskSpanContext
    ? trace.setSpanContext(ctx.rootContext(), details.taskSpanContext)
    : parentSessionID
      ? resolveSessionTraceContext(parentSessionID, ctx)
      : ctx.rootContext()
  const span = ctx.tracer.startSpan(
    `${ctx.tracePrefix}run`,
    {
      startTime,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
        [SESSION_ID]: sessionID,
        [AGENT_NAME]: agent,
        "agent.type": agentType,
        "session.is_subagent": isSubagent,
        ...(parentSessionID ? { "session.parent_id": parentSessionID } : {}),
        ...(details?.taskCallID ? { "task.call_id": details.taskCallID } : {}),
        ...ctx.commonAttrs,
      },
    },
    parentContext,
  )
  span.setAttribute("opencode.run.id", span.spanContext().spanId)
  const run = {
    span,
    interactionIDs: new Set<string>(),
    interactionIO: new Map<string, { input: string; output?: string }>(),
  }
  ctx.activeRunSpans.set(sessionID, run)
  return run
}

/** Starts or refreshes the interaction span for a single user turn, keyed by the user message ID. */
export function handleInteractionStarted(
  interactionID: string,
  sessionID: string,
  agent: string,
  promptText: string,
  model: string,
  startTime: number,
  ctx: HandlerContext,
  details?: RunDetails,
) {
  const existing = ctx.interactionSpans.get(interactionID)
  if (!existing && ctx.interactionSpanContexts.has(interactionID)) return
  ctx.activeInteractions.set(sessionID, interactionID)
  if (promptText) setBoundedMap(ctx.interactionInputs, interactionID, promptText)
  if (!isTraceEnabled("session", ctx)) return
  const run = ensureRunStarted(sessionID, agent, startTime, ctx, details)
  if (run) {
    run.interactionIDs.add(interactionID)
    const interactionIO = run.interactionIO.get(interactionID)
    run.interactionIO.set(interactionID, {
      ...interactionIO,
      input: promptText || interactionIO?.input || "",
    })
  }
  const parentSessionID = details?.parentSessionID ?? ctx.sessionParents.get(sessionID)
  const agentType: SessionAgentType = details?.agentType ?? (parentSessionID ? "subagent" : "primary")
  const isSubagent = agentType === "subagent"
  if (existing) {
    existing.setAttributes({
      "opencode.interaction.id": interactionID,
      [AGENT_NAME]: agent,
      "agent.type": agentType,
      "session.is_subagent": isSubagent,
      ...(parentSessionID ? { "session.parent_id": parentSessionID } : {}),
      ...(details?.taskCallID ? { "task.call_id": details.taskCallID } : {}),
      ...(promptText
        ? {
            [INPUT_VALUE]: promptText,
            [INPUT_MIME_TYPE]: MimeType.TEXT,
            [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
            [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: promptText,
          }
        : {}),
      "opencode.model": model,
    })
    return
  }

  const parentContext = run ? trace.setSpan(ctx.rootContext(), run.span) : ctx.rootContext()
  const interactionSpan = ctx.tracer.startSpan(
    `${ctx.tracePrefix}interaction`,
    {
      startTime,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        "opencode.interaction.id": interactionID,
        [SESSION_ID]: sessionID,
        [AGENT_NAME]: agent,
        "agent.type": agentType,
        "session.is_subagent": isSubagent,
        ...(parentSessionID ? { "session.parent_id": parentSessionID } : {}),
        ...(details?.taskCallID ? { "task.call_id": details.taskCallID } : {}),
        ...(promptText
          ? {
              [INPUT_VALUE]: promptText,
              [INPUT_MIME_TYPE]: MimeType.TEXT,
              [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
              [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: promptText,
            }
          : {}),
        "opencode.model": model,
        ...ctx.commonAttrs,
      },
    },
    parentContext,
  )
  ctx.interactionSpans.set(interactionID, interactionSpan)
  setBoundedMap(ctx.interactionSpanContexts, interactionID, interactionSpan.spanContext())
  ctx.interactionTotals.set(interactionID, { tokens: 0, cost: 0, messages: 0 })
}

/** Increments the session counter, records session state, and emits a `session.created` log event. */
export function handleSessionCreated(e: EventSessionCreated, ctx: HandlerContext) {
  const { id: sessionID, time, parentID } = e.properties.info
  const createdAt = time.created
  const isSubagent = !!parentID
  const agentType: SessionAgentType = isSubagent ? "subagent" : "primary"
  if (isMetricEnabled("session.count", ctx)) {
    ctx.instruments.sessionCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID, is_subagent: isSubagent })
  }
  setBoundedMap(ctx.sessionTotals, sessionID, { startMs: createdAt, tokens: 0, cost: 0, messages: 0, agent: "unknown", agentType })

  if (parentID) setBoundedMap(ctx.sessionParents, sessionID, parentID)

  ctx.emitLog({
    severityNumber: SeverityNumber.INFO,
    severityText: "INFO",
    timestamp: createdAt,
    observedTimestamp: Date.now(),
    body: "session.created",
    attributes: {
      "event.name": "session.created",
      "session.id": sessionID,
      is_subagent: isSubagent,
      ...agentAttrs("unknown", agentType),
      ...ctx.commonAttrs,
    },
  })
  return ctx.log("info", "otel: session.created", { sessionID, createdAt, isSubagent })
}

function sweepSession(sessionID: string, ctx: HandlerContext) {
  for (const [id, perm] of ctx.pendingPermissions) {
    if (perm.sessionID === sessionID) ctx.pendingPermissions.delete(id)
  }
  for (const [key, span] of ctx.pendingToolSpans) {
    if (span.sessionID === sessionID) {
      span.span?.setStatus({ code: SpanStatusCode.ERROR, message: "session ended before tool completed" })
      span.span?.end()
      ctx.pendingToolSpans.delete(key)
    }
  }
  ctx.pendingSubagentRuns.delete(sessionID)
  for (const [childSessionID, details] of ctx.pendingSubagentRuns) {
    if (details.parentSessionID === sessionID) ctx.pendingSubagentRuns.delete(childSessionID)
  }
  const msgPrefix = `${sessionID}:`
  for (const [key, span] of ctx.messageSpans) {
    if (key.startsWith(msgPrefix)) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: "session ended before message completed" })
      span.end()
      ctx.messageSpans.delete(key)
    }
  }
  for (const key of ctx.messageOutputs.keys()) {
    if (key.startsWith(msgPrefix) && !ctx.pendingAssistantInteractions.has(key)) {
      ctx.messageOutputs.delete(key)
    }
  }
  for (const key of ctx.llmRequestContexts.keys()) {
    if (key.startsWith(msgPrefix)) ctx.llmRequestContexts.delete(key)
  }
  ctx.activeMessageSpans.delete(sessionID)
  for (const key of ctx.llmTelemetryOutputs.keys()) {
    if (key.startsWith(msgPrefix)) ctx.llmTelemetryOutputs.delete(key)
  }
}

function endInteractions(
  sessionID: string,
  status: SpanStatusCode.OK | SpanStatusCode.ERROR,
  ctx: HandlerContext,
  error?: string,
) {
  const run = ctx.activeRunSpans.get(sessionID)
  if (!run) {
    for (const [key, pending] of ctx.pendingAssistantInteractions) {
      if (pending.sessionID === sessionID && !ctx.interactionSpans.has(pending.interactionID)) {
        ctx.pendingAssistantInteractions.delete(key)
      }
    }
    return
  }
  for (const interactionID of run.interactionIDs) {
    const hasPendingAssistant = [...ctx.pendingAssistantInteractions.values()].some(
      pending => pending.sessionID === sessionID && pending.interactionID === interactionID,
    )
    if (hasPendingAssistant) continue
    const endTime = status === SpanStatusCode.OK
      ? ctx.interactionCompletions.get(interactionID)?.endTime
      : undefined
    endInteractionSpan(interactionID, sessionID, status, ctx, endTime, error)
  }
}

/** Emits a `session.idle` log event, records totals, ends the active run, and clears pending state. */
export function handleSessionIdle(e: EventSessionIdle, ctx: HandlerContext) {
  const sessionID = e.properties.sessionID
  const totals = ctx.sessionTotals.get(sessionID)
  const { agentName, agentType } = getSessionAgentMeta(sessionID, ctx)
  ctx.sessionTotals.delete(sessionID)
  ctx.sessionDiffTotals.delete(sessionID)
  sweepSession(sessionID, ctx)
  endInteractions(sessionID, SpanStatusCode.OK, ctx)

  const attrs = { ...ctx.commonAttrs, "session.id": sessionID }
  let duration_ms: number | undefined

  if (totals) {
    duration_ms = Date.now() - totals.startMs
    if (isMetricEnabled("session.duration", ctx)) {
      ctx.instruments.sessionDurationHistogram.record(duration_ms, attrs)
    }
    if (isMetricEnabled("session.token.total", ctx)) {
      ctx.instruments.sessionTokenGauge.record(totals.tokens, attrs)
    }
    if (isMetricEnabled("session.cost.total", ctx)) {
      ctx.instruments.sessionCostGauge.record(totals.cost, attrs)
    }
  }

  const run = ctx.activeRunSpans.get(sessionID)
  if (run) {
    if (totals) {
      run.span.setAttributes({
        [AGENT_NAME]: totals.agent,
        "agent.type": totals.agentType,
        "run.total_tokens": totals.tokens,
        "run.total_cost_usd": totals.cost,
        "run.total_messages": totals.messages,
      })
    }
    setRunIOAttributes(run)
    run.span.setAttribute("run.total_interactions", run.interactionIDs.size)
    run.span.setStatus({ code: SpanStatusCode.OK })
    run.span.end()
    ctx.activeRunSpans.delete(sessionID)
  }

  ctx.emitLog({
    severityNumber: SeverityNumber.INFO,
    severityText: "INFO",
    timestamp: Date.now(),
    observedTimestamp: Date.now(),
    body: "session.idle",
    attributes: {
      "event.name": "session.idle",
      "session.id": sessionID,
      total_tokens: totals?.tokens ?? 0,
      total_cost_usd: totals?.cost ?? 0,
      total_messages: totals?.messages ?? 0,
      ...agentAttrs(agentName, agentType),
      ...ctx.commonAttrs,
    },
  })
  ctx.log("debug", "otel: session.idle", {
    sessionID,
    ...(totals ? { duration_ms, total_tokens: totals.tokens, total_cost_usd: totals.cost, total_messages: totals.messages } : {}),
  })
}

/** Emits a `session.error` log event, ends the active run with error status, and clears pending state. */
export function handleSessionError(e: EventSessionError, ctx: HandlerContext) {
  const rawID = e.properties.sessionID
  const sessionID = rawID ?? "unknown"
  const error = errorSummary(e.properties.error)
  const { agentName, agentType } = rawID ? getSessionAgentMeta(rawID, ctx) : { agentName: "unknown", agentType: "unknown" as const }
  const totals = rawID ? ctx.sessionTotals.get(rawID) : undefined
  if (rawID) {
    ctx.sessionTotals.delete(rawID)
    ctx.sessionDiffTotals.delete(rawID)
  }
  sweepSession(sessionID, ctx)
  if (rawID) endInteractions(rawID, SpanStatusCode.ERROR, ctx, error)

  if (rawID) {
    const run = ctx.activeRunSpans.get(rawID)
    if (run) {
      if (totals) {
        run.span.setAttributes({
          [AGENT_NAME]: totals.agent,
          "agent.type": totals.agentType,
          "run.total_tokens": totals.tokens,
          "run.total_cost_usd": totals.cost,
          "run.total_messages": totals.messages,
        })
      }
      setRunIOAttributes(run)
      run.span.setAttribute("run.total_interactions", run.interactionIDs.size)
      run.span.setStatus({ code: SpanStatusCode.ERROR, message: error })
      run.span.setAttribute("error", error)
      run.span.end()
      ctx.activeRunSpans.delete(rawID)
    }
  }

  ctx.emitLog({
    severityNumber: SeverityNumber.ERROR,
    severityText: "ERROR",
    timestamp: Date.now(),
    observedTimestamp: Date.now(),
    body: "session.error",
    attributes: {
      "event.name": "session.error",
      "session.id": sessionID,
      error,
      ...agentAttrs(agentName, agentType),
      ...ctx.commonAttrs,
    },
  })
  ctx.log("error", "otel: session.error", { sessionID, error })
}

/** Increments the retry counter when the session enters a retry state. */
export function handleSessionStatus(e: EventSessionStatus, ctx: HandlerContext) {
  const { sessionID, status } = e.properties
  if (status.type === "busy") {
    ensureRunStarted(sessionID, "unknown", Date.now(), ctx)
    return
  }
  if (status.type !== "retry") return
  const { attempt, message: retryMessage } = status
  if (isMetricEnabled("retry.count", ctx)) {
    ctx.instruments.retryCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID })
    ctx.log("debug", "otel: retry counter incremented", { sessionID, attempt, retryMessage })
  }
}
