import { SpanKind, SpanStatusCode, trace } from "@opentelemetry/api"
import type { CompactionPart, Part, TextPart, UserMessage } from "@opencode-ai/sdk"
import {
  OpenInferenceSpanKind,
  SemanticConventions,
  SESSION_ID,
} from "@arizeai/openinference-semantic-conventions"
import { interactionHandlers } from "./interaction.ts"
import { ensureRunStarted, takeRunDetails } from "./run.ts"
import { setBoundedMap, tryResolveInteractionTraceContext } from "./util.ts"
import type { HandlerContext } from "./types.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND

type CompactionContinuePart = TextPart & {
  synthetic: true
  metadata: Record<string, unknown> & { compaction_continue: true }
}

function createCompactionState(): Pick<
  HandlerContext,
  "userMessages" | "activeCompactions" | "compactionRecords" | "recentCompactions"
> {
  return {
    userMessages: new Map(),
    activeCompactions: new Map(),
    compactionRecords: new Map(),
    recentCompactions: new Map(),
  }
}

function isCompactionContinuePart(part: Part): part is CompactionContinuePart {
  return part.type === "text"
    && part.synthetic === true
    && part.metadata?.["compaction_continue"] === true
}

function endActiveCompaction(
  sessionID: string,
  status: SpanStatusCode.OK | SpanStatusCode.ERROR,
  ctx: HandlerContext,
  error?: string,
) {
  const active = ctx.activeCompactions.get(sessionID)
  if (!active) return
  active.span.setStatus(error ? { code: status, message: error } : { code: status })
  if (error) active.span.setAttribute("error", error)
  active.span.end()
  ctx.activeCompactions.delete(sessionID)
  if (status === SpanStatusCode.OK) {
    const record = ctx.compactionRecords.get(active.markerMessageID)
    if (record) setBoundedMap(ctx.recentCompactions, sessionID, record)
  } else {
    ctx.recentCompactions.delete(sessionID)
  }
}

const compactionHandlers = {
  recordUser(message: UserMessage, ctx: HandlerContext) {
    setBoundedMap(ctx.userMessages, message.id, {
      sessionID: message.sessionID,
      agent: message.agent,
      startTime: message.time.created,
    })
  },

  start(part: CompactionPart, ctx: HandlerContext) {
    if (ctx.compactionRecords.has(part.messageID)) return
    const current = ctx.activeCompactions.get(part.sessionID)
    if (current) {
      endActiveCompaction(
        part.sessionID,
        SpanStatusCode.ERROR,
        ctx,
        "a new compaction started before the previous compaction completed",
      )
    }

    const user = ctx.userMessages.get(part.messageID)
    const ownerInteractionID = part.auto
      ? interactionHandlers.latest(part.sessionID, ctx)
      : undefined
    if (ownerInteractionID) {
      interactionHandlers.materialize(ownerInteractionID, part.sessionID, ctx)
      interactionHandlers.bindAlias(part.messageID, part.sessionID, ownerInteractionID, ctx)
    }

    const startTime = user?.sessionID === part.sessionID ? user.startTime : Date.now()
    const run = ctx.activeRunSpans.get(part.sessionID)
      ?? ensureRunStarted(
        part.sessionID,
        user?.agent ?? "compaction",
        startTime,
        ctx,
        takeRunDetails(part.sessionID, ctx),
      )
    const parentContext = (ownerInteractionID
      ? tryResolveInteractionTraceContext(ownerInteractionID, ctx)
      : undefined) ?? trace.setSpan(ctx.rootContext(), run.span)
    const span = ctx.tracer.startSpan(
      `${ctx.tracePrefix}compaction`,
      {
        startTime,
        kind: SpanKind.INTERNAL,
        attributes: {
          [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
          [SESSION_ID]: part.sessionID,
          "opencode.compaction.id": part.messageID,
          "opencode.compaction.auto": part.auto,
          ...(ownerInteractionID ? { "opencode.interaction.id": ownerInteractionID } : {}),
          ...ctx.commonAttrs,
        },
      },
      parentContext,
    )
    const record = {
      sessionID: part.sessionID,
      markerMessageID: part.messageID,
      ownerInteractionID,
      auto: part.auto,
      spanContext: span.spanContext(),
    }
    setBoundedMap(ctx.compactionRecords, part.messageID, record)
    ctx.activeCompactions.set(part.sessionID, { ...record, span })
    ctx.recentCompactions.delete(part.sessionID)
  },

  handlePart(part: Part, ctx: HandlerContext) {
    if (part.type === "compaction") {
      compactionHandlers.start(part, ctx)
      return true
    }
    if (!isCompactionContinuePart(part)) return false
    const compaction = ctx.activeCompactions.get(part.sessionID)
      ?? ctx.recentCompactions.get(part.sessionID)
    if (compaction?.ownerInteractionID) {
      interactionHandlers.bindAlias(
        part.messageID,
        part.sessionID,
        compaction.ownerInteractionID,
        ctx,
      )
    }
    return true
  },

  resolve(sessionID: string, markerMessageID: string, ctx: HandlerContext) {
    const active = ctx.activeCompactions.get(sessionID)
    if (active?.markerMessageID === markerMessageID) {
      return {
        markerMessageID,
        ownerInteractionID: active.ownerInteractionID,
        parentContext: trace.setSpan(ctx.rootContext(), active.span),
      }
    }
    const record = ctx.compactionRecords.get(markerMessageID)
    if (record?.sessionID !== sessionID) return
    return {
      markerMessageID,
      ownerInteractionID: record.ownerInteractionID,
      parentContext: trace.setSpanContext(ctx.rootContext(), record.spanContext),
    }
  },

  recoverOwner(sessionID: string, messageID: string, ctx: HandlerContext) {
    const compaction = ctx.recentCompactions.get(sessionID)
    if (!compaction?.ownerInteractionID) return
    interactionHandlers.bindAlias(messageID, sessionID, compaction.ownerInteractionID, ctx)
    return compaction.ownerInteractionID
  },

  complete(sessionID: string, ctx: HandlerContext) {
    endActiveCompaction(sessionID, SpanStatusCode.OK, ctx)
  },

  fail(sessionID: string, error: string, ctx: HandlerContext) {
    endActiveCompaction(sessionID, SpanStatusCode.ERROR, ctx, error)
  },

  clearRecent(sessionID: string, ctx: HandlerContext) {
    ctx.recentCompactions.delete(sessionID)
  },
}

export { compactionHandlers, createCompactionState }
