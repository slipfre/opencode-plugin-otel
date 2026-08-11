import { SpanKind, SpanStatusCode, trace } from "@opentelemetry/api"
import type { EventPermissionAsked, EventPermissionReplied } from "@opencode-ai/sdk/v2"
import {
  AGENT_NAME,
  OpenInferenceSpanKind,
  SemanticConventions,
  SESSION_ID,
  TOOL_ID,
  TOOL_NAME,
} from "@arizeai/openinference-semantic-conventions"
import { MAX_PENDING, type HandlerContext } from "../types.ts"
import { setBoundedMap } from "../util.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND
type PendingPermissionSpan = HandlerContext["pendingPermissionSpans"] extends Map<string, infer V> ? V : never

function permissionKey(sessionID: string, requestID: string) {
  return `${sessionID}:${requestID}`
}

function storePermissionSpan(
  key: string,
  value: PendingPermissionSpan,
  ctx: HandlerContext,
) {
  const oldest = !ctx.pendingPermissionSpans.has(key) && ctx.pendingPermissionSpans.size >= MAX_PENDING
    ? ctx.pendingPermissionSpans.values().next().value
    : undefined
  setBoundedMap(ctx.pendingPermissionSpans, key, value)
  if (!oldest) return
  oldest.span.setStatus({ code: SpanStatusCode.ERROR, message: "permission correlation capacity exceeded" })
  oldest.span.end()
  return ctx.log("warn", "otel: pending permission span evicted", {
    sessionID: oldest.sessionID,
    callID: oldest.callID,
  })
}

const permissionHandlers = {
  asked(e: EventPermissionAsked, ctx: HandlerContext) {
    const request = e.properties
    if (!request.tool) {
      return ctx.log("debug", "otel: permission span skipped without tool correlation", {
        sessionID: request.sessionID,
        requestID: request.id,
        permission: request.permission,
      })
    }

    const key = permissionKey(request.sessionID, request.id)
    if (ctx.pendingPermissionSpans.has(key)) {
      return ctx.log("debug", "otel: duplicate permission request ignored", {
        sessionID: request.sessionID,
        requestID: request.id,
        callID: request.tool.callID,
      })
    }

    const tool = ctx.pendingToolSpans.get(`${request.sessionID}:${request.tool.callID}`)
    if (!tool) {
      return ctx.log("warn", "otel: permission span skipped because tool span was not found", {
        sessionID: request.sessionID,
        requestID: request.id,
        callID: request.tool.callID,
        permission: request.permission,
      })
    }

    const startMs = Date.now()
    const run = ctx.activeRunSpans.get(request.sessionID)
    const span = ctx.tracer.startSpan(
      `${ctx.tracePrefix}permission.check`,
      {
        startTime: startMs,
        kind: SpanKind.INTERNAL,
        attributes: {
          [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.GUARDRAIL,
          [SESSION_ID]: request.sessionID,
          [TOOL_ID]: request.tool.callID,
          [TOOL_NAME]: tool.tool,
          [AGENT_NAME]: run?.agent ?? "unknown",
          "agent.type": run?.agentType
            ?? ctx.pendingSubagentRuns.get(request.sessionID)?.agentType
            ?? (ctx.sessionParents.has(request.sessionID) ? "subagent" : "primary"),
          "permission.request.id": request.id,
          "permission.name": request.permission,
          "permission.patterns": request.patterns,
          "permission.tool.call_id": request.tool.callID,
          "permission.tool.message_id": request.tool.messageID,
          ...ctx.commonAttrs,
        },
      },
      trace.setSpan(ctx.rootContext(), tool.span),
    )
    return storePermissionSpan(key, {
      sessionID: request.sessionID,
      callID: request.tool.callID,
      startMs,
      span,
    }, ctx)
  },

  replied(e: EventPermissionReplied, ctx: HandlerContext) {
    const reply = e.properties
    const key = permissionKey(reply.sessionID, reply.requestID)
    const pending = ctx.pendingPermissionSpans.get(key)
    if (!pending) {
      return ctx.log("debug", "otel: permission reply has no pending span", {
        sessionID: reply.sessionID,
        requestID: reply.requestID,
        reply: reply.reply,
      })
    }

    ctx.pendingPermissionSpans.delete(key)
    const endMs = Date.now()
    pending.span.setAttributes({
      "permission.reply": reply.reply,
      "permission.granted": reply.reply !== "reject",
      "permission.wait_ms": Math.max(0, endMs - pending.startMs),
    })
    if (reply.reply === "reject") {
      const tool = ctx.pendingToolSpans.get(`${reply.sessionID}:${pending.callID}`)
      if (tool) tool.errorType = "PermissionRejectedError"
    }
    pending.span.setStatus({ code: SpanStatusCode.OK })
    pending.span.end(endMs)
  },

  endTool(sessionID: string, callID: string, ctx: HandlerContext, error: string) {
    for (const [key, pending] of ctx.pendingPermissionSpans) {
      if (pending.sessionID !== sessionID || pending.callID !== callID) continue
      pending.span.setStatus({ code: SpanStatusCode.ERROR, message: error })
      pending.span.end()
      ctx.pendingPermissionSpans.delete(key)
    }
  },

  endSession(sessionID: string, ctx: HandlerContext, error: string) {
    for (const [key, pending] of ctx.pendingPermissionSpans) {
      if (pending.sessionID !== sessionID) continue
      pending.span.setStatus({ code: SpanStatusCode.ERROR, message: error })
      pending.span.end()
      ctx.pendingPermissionSpans.delete(key)
    }
  },
}

export { permissionHandlers }
