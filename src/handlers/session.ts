import { SpanStatusCode } from "@opentelemetry/api";
import type {
  EventSessionCreated,
  EventSessionCompacted,
  EventSessionIdle,
  EventSessionError,
  EventSessionStatus,
} from "@opencode-ai/sdk";
import { errorSummary, setBoundedMap } from "../util.ts";
import type { HandlerContext } from "../types.ts";
import { interactionHandlers } from "../interaction.ts";
import { endRunSpan } from "../run.ts";
import { compactionHandlers } from "../compaction.ts";
import { permissionHandlers } from "./permission.ts";

/** Records the parent-session relationship used to identify and parent subagent runs. */
export function handleSessionCreated(
  e: EventSessionCreated,
  ctx: HandlerContext
) {
  const { id: sessionID, parentID } = e.properties.info;
  if (parentID) {
    setBoundedMap(ctx.sessionParents, sessionID, parentID);
  }
}

function handleSessionStatus(e: EventSessionStatus, ctx: HandlerContext) {
  const { sessionID, status } = e.properties;
  if (status.type !== "retry") {
    return;
  }
  ctx.activeMessageSpans
    .get(sessionID)
    ?.span.setAttribute("opencode.llm.retry_count", status.attempt);
}

function sweepSession(
  sessionID: string,
  ctx: HandlerContext,
  messageError?: { messageID: string; error: string; errorType?: string }
) {
  permissionHandlers.endSession(
    sessionID,
    ctx,
    "session ended before permission reply"
  );
  for (const [key, span] of ctx.pendingToolSpans) {
    if (span.sessionID === sessionID) {
      span.span.setStatus({
        code: SpanStatusCode.ERROR,
        message: "session ended before tool completed",
      });
      span.span.end();
      ctx.pendingToolSpans.delete(key);
    }
  }
  ctx.pendingSubagentRuns.delete(sessionID);
  for (const [childSessionID, details] of ctx.pendingSubagentRuns) {
    if (details.parentSessionID === sessionID) {
      ctx.pendingSubagentRuns.delete(childSessionID);
    }
  }
  const msgPrefix = `${sessionID}:`;
  for (const [key, span] of ctx.messageSpans) {
    if (key.startsWith(msgPrefix)) {
      const error =
        messageError && key === `${sessionID}:${messageError.messageID}`
          ? messageError.error
          : "session ended before message completed";
      span.setStatus({ code: SpanStatusCode.ERROR, message: error });
      if (messageError && key === `${sessionID}:${messageError.messageID}`) {
        span.setAttributes({
          "llm.finish_reason": "error",
          ...(messageError.errorType
            ? { "error.type": messageError.errorType }
            : {}),
        });
      }
      span.end();
      ctx.messageSpans.delete(key);
    }
  }
  for (const key of ctx.llmSpanTimings.keys()) {
    if (key.startsWith(msgPrefix)) {
      ctx.llmSpanTimings.delete(key);
    }
  }
  for (const key of ctx.messageOutputs.keys()) {
    if (
      key.startsWith(msgPrefix) &&
      !interactionHandlers.hasPendingAssistant(key, ctx)
    ) {
      ctx.messageOutputs.delete(key);
    }
  }
  for (const key of ctx.llmRequestContexts.keys()) {
    if (key.startsWith(msgPrefix)) {
      ctx.llmRequestContexts.delete(key);
    }
  }
  ctx.activeMessageSpans.delete(sessionID);
  for (const key of ctx.llmTelemetryOutputs.keys()) {
    if (key.startsWith(msgPrefix)) {
      ctx.llmTelemetryOutputs.delete(key);
    }
  }
}

/** Records totals, ends the active run, and clears pending state. */
export function handleSessionIdle(e: EventSessionIdle, ctx: HandlerContext) {
  const sessionID = e.properties.sessionID;
  const overflow = compactionHandlers.pendingContextOverflow(sessionID, ctx);
  if (overflow) {
    compactionHandlers.fail(sessionID, overflow.error, ctx);
    sweepSession(sessionID, ctx, {
      ...overflow,
      errorType: "ContextOverflowError",
    });
    if (overflow.ownerInteractionID) {
      interactionHandlers.end(
        overflow.ownerInteractionID,
        sessionID,
        SpanStatusCode.ERROR,
        ctx,
        undefined,
        overflow.error
      );
    }
    interactionHandlers.endSession(
      sessionID,
      SpanStatusCode.ERROR,
      ctx,
      overflow.error
    );
    endRunSpan(sessionID, SpanStatusCode.ERROR, ctx, overflow.error);
    compactionHandlers.clearRecent(sessionID, ctx);
    compactionHandlers.clearContextOverflow(sessionID, ctx);
    return;
  }
  compactionHandlers.fail(
    sessionID,
    "session ended before compaction completed",
    ctx
  );
  sweepSession(sessionID, ctx);
  interactionHandlers.endSession(sessionID, SpanStatusCode.OK, ctx);
  endRunSpan(sessionID, SpanStatusCode.OK, ctx);
  compactionHandlers.clearRecent(sessionID, ctx);
}

/** Ends the active run with error status and clears pending state. */
export function handleSessionError(
  e: EventSessionError,
  ctx: HandlerContext
): "recoverable" | "terminal" {
  const rawID = e.properties.sessionID;
  const sessionID = rawID ?? "unknown";
  const error = errorSummary(e.properties.error);
  const errorName = (e.properties.error as { name?: string } | undefined)?.name;
  if (
    rawID &&
    errorName === "ContextOverflowError" &&
    compactionHandlers.deferContextOverflow(rawID, error, ctx)
  ) {
    ctx.log("warn", "otel: context overflow recovery pending", {
      sessionID,
      error,
    });
    return "recoverable";
  }
  const activeMessage = rawID ? ctx.activeMessageSpans.get(rawID) : undefined;
  const compactionOwnerInteractionID = rawID
    ? ctx.activeCompactions.get(rawID)?.ownerInteractionID
    : undefined;
  compactionHandlers.fail(sessionID, error, ctx);
  sweepSession(
    sessionID,
    ctx,
    activeMessage
      ? {
          messageID: activeMessage.messageID,
          error,
          ...(errorName ? { errorType: errorName } : {}),
        }
      : undefined
  );
  if (rawID) {
    if (compactionOwnerInteractionID) {
      interactionHandlers.end(
        compactionOwnerInteractionID,
        rawID,
        SpanStatusCode.ERROR,
        ctx,
        undefined,
        error
      );
    }
    interactionHandlers.endSession(rawID, SpanStatusCode.ERROR, ctx, error);
    endRunSpan(rawID, SpanStatusCode.ERROR, ctx, error);
  }
  compactionHandlers.clearRecent(sessionID, ctx);
  compactionHandlers.clearContextOverflow(sessionID, ctx);
  ctx.log("error", "otel: session.error", { sessionID, error });
  return "terminal";
}

function handleSessionCompacted(e: EventSessionCompacted, ctx: HandlerContext) {
  compactionHandlers.complete(e.properties.sessionID, ctx);
}

export { handleSessionCompacted, handleSessionStatus };
