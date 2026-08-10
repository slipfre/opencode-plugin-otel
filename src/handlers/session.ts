import { SpanStatusCode } from "@opentelemetry/api"
import type {
  EventSessionCreated,
  EventSessionCompacted,
  EventSessionIdle,
  EventSessionError,
} from "@opencode-ai/sdk"
import { errorSummary, setBoundedMap } from "../util.ts"
import type { HandlerContext } from "../types.ts"
import { interactionHandlers } from "../interaction.ts"
import { endRunSpan } from "../run.ts"
import { compactionHandlers } from "../compaction.ts"

/** Records the parent-session relationship used to identify and parent subagent runs. */
export function handleSessionCreated(e: EventSessionCreated, ctx: HandlerContext) {
  const { id: sessionID, parentID } = e.properties.info
  if (parentID) setBoundedMap(ctx.sessionParents, sessionID, parentID)
}

function sweepSession(sessionID: string, ctx: HandlerContext) {
  for (const [key, span] of ctx.pendingToolSpans) {
    if (span.sessionID === sessionID) {
      span.span.setStatus({ code: SpanStatusCode.ERROR, message: "session ended before tool completed" })
      span.span.end()
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
    if (key.startsWith(msgPrefix) && !interactionHandlers.hasPendingAssistant(key, ctx)) {
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

/** Records totals, ends the active run, and clears pending state. */
export function handleSessionIdle(e: EventSessionIdle, ctx: HandlerContext) {
  const sessionID = e.properties.sessionID
  compactionHandlers.fail(sessionID, "session ended before compaction completed", ctx)
  sweepSession(sessionID, ctx)
  interactionHandlers.endSession(sessionID, SpanStatusCode.OK, ctx)
  endRunSpan(sessionID, SpanStatusCode.OK, ctx)
  compactionHandlers.clearRecent(sessionID, ctx)
}

/** Ends the active run with error status and clears pending state. */
export function handleSessionError(e: EventSessionError, ctx: HandlerContext) {
  const rawID = e.properties.sessionID
  const sessionID = rawID ?? "unknown"
  const error = errorSummary(e.properties.error)
  compactionHandlers.fail(sessionID, error, ctx)
  sweepSession(sessionID, ctx)
  if (rawID) {
    interactionHandlers.endSession(rawID, SpanStatusCode.ERROR, ctx, error)
    endRunSpan(rawID, SpanStatusCode.ERROR, ctx, error)
  }
  compactionHandlers.clearRecent(sessionID, ctx)
  ctx.log("error", "otel: session.error", { sessionID, error })
}

function handleSessionCompacted(e: EventSessionCompacted, ctx: HandlerContext) {
  compactionHandlers.complete(e.properties.sessionID, ctx)
}

export { handleSessionCompacted }
