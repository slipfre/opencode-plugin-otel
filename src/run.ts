import { SpanStatusCode, trace } from "@opentelemetry/api"
import {
  AGENT_NAME,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  MimeType,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  OpenInferenceSpanKind,
  SemanticConventions,
  SESSION_ID,
} from "@arizeai/openinference-semantic-conventions"
import { resolveSessionTraceContext } from "./util.ts"
import type { ActiveRunSpan, HandlerContext, RunDetails, SessionAgentType } from "./types.ts"

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

function takeRunDetails(sessionID: string, ctx: HandlerContext): RunDetails {
  const details = ctx.pendingSubagentRuns.get(sessionID)
  if (details) {
    ctx.pendingSubagentRuns.delete(sessionID)
    return details
  }
  const parentSessionID = ctx.sessionParents.get(sessionID)
  return {
    agentType: parentSessionID ? "subagent" : "primary",
    ...(parentSessionID ? { parentSessionID } : {}),
  }
}

function ensureRunStarted(
  sessionID: string,
  agent: string,
  startTime: number,
  ctx: HandlerContext,
  details?: RunDetails,
) {
  const parentSessionID = details?.parentSessionID ?? ctx.sessionParents.get(sessionID)
  const agentType: SessionAgentType = details?.agentType ?? (parentSessionID ? "subagent" : "primary")
  const isSubagent = agentType === "subagent"
  const existing = ctx.activeRunSpans.get(sessionID)
  if (existing) {
    if (agent !== "unknown") existing.agent = agent
    existing.agentType = agentType
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
    agent,
    agentType,
    tokens: 0,
    cost: 0,
    messages: 0,
    interactionIDs: new Set<string>(),
    interactionIO: new Map<string, { input: string; output?: string }>(),
  }
  ctx.activeRunSpans.set(sessionID, run)
  return run
}

function endRunSpan(
  sessionID: string,
  status: SpanStatusCode.OK | SpanStatusCode.ERROR,
  ctx: HandlerContext,
  error?: string,
) {
  const run = ctx.activeRunSpans.get(sessionID)
  if (!run) return
  run.span.setAttributes({
    [AGENT_NAME]: run.agent,
    "agent.type": run.agentType,
    "run.total_tokens": run.tokens,
    "run.total_cost_usd": run.cost,
    "run.total_messages": run.messages,
  })
  setRunIOAttributes(run)
  run.span.setAttribute("run.total_interactions", run.interactionIDs.size)
  run.span.setStatus(error ? { code: status, message: error } : { code: status })
  if (error) run.span.setAttribute("error", error)
  run.span.end()
  ctx.activeRunSpans.delete(sessionID)
}

export { endRunSpan, ensureRunStarted, takeRunDetails }
