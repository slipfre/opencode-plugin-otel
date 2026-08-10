import { SpanStatusCode, trace } from "@opentelemetry/api"
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
import { ensureRunStarted, takeRunDetails } from "./run.ts"
import { setBoundedMap } from "./util.ts"
import type { HandlerContext, SessionAgentType } from "./types.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND

function createInteractionState(): Pick<
  HandlerContext,
  | "interactionSpans"
  | "interactionSpanContexts"
  | "activeInteractions"
  | "assistantInteractions"
  | "pendingInteractions"
  | "pendingAssistantInteractions"
  | "interactionInputs"
  | "interactionTotals"
  | "interactionCompletions"
> {
  return {
    // interactionID -> currently recording interaction span; removed when the interaction ends.
    interactionSpans: new Map(),
    // interactionID -> retained span context used to parent late child events and reject replayed starts.
    interactionSpanContexts: new Map(),
    // sessionID -> interactionID currently selected for session-level child span correlation.
    activeInteractions: new Map(),
    // assistant message ID -> interactionID that owns the assistant and its tool/message events.
    assistantInteractions: new Map(),
    // user message ID -> canonical staged metadata awaiting an assistant with the matching parentID.
    // noReply entries may survive idle/error and are bounded by MAX_PENDING until materialized or evicted.
    pendingInteractions: new Map(),
    // "sessionID:assistantMessageID" -> owning interaction while assistant completion may still arrive late.
    // Session shutdown keeps that interaction open until the corresponding completion removes this entry.
    pendingAssistantInteractions: new Map(),
    // interactionID -> user prompt text reused as LLM input and run-level input; retained in a bounded map.
    interactionInputs: new Map(),
    // interactionID -> accumulated token, cost, and assistant-message totals; deleted when the interaction ends.
    interactionTotals: new Map(),
    // interactionID -> latest usable completion time and output for final interaction/run attributes.
    // Entries are deleted together with their interaction span.
    interactionCompletions: new Map(),
  }
}

function handleInteractionStarted(
  interactionID: string,
  sessionID: string,
  agent: string,
  promptText: string,
  model: string,
  startTime: number,
  ctx: HandlerContext,
) {
  const existing = ctx.interactionSpans.get(interactionID)
  if (!existing && ctx.interactionSpanContexts.has(interactionID)) return
  const details = takeRunDetails(sessionID, ctx)
  ctx.activeInteractions.set(sessionID, interactionID)
  if (promptText) setBoundedMap(ctx.interactionInputs, interactionID, promptText)
  const run = ensureRunStarted(sessionID, agent, startTime, ctx, details)
  run.interactionIDs.add(interactionID)
  const interactionIO = run.interactionIO.get(interactionID)
  run.interactionIO.set(interactionID, {
    ...interactionIO,
    input: promptText || interactionIO?.input || "",
  })
  const parentSessionID = details.parentSessionID
  const agentType: SessionAgentType = details.agentType
  const isSubagent = agentType === "subagent"
  if (existing) {
    existing.setAttributes({
      "opencode.interaction.id": interactionID,
      [AGENT_NAME]: agent,
      "agent.type": agentType,
      "session.is_subagent": isSubagent,
      ...(parentSessionID ? { "session.parent_id": parentSessionID } : {}),
      ...(details.taskCallID ? { "task.call_id": details.taskCallID } : {}),
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

  const parentContext = trace.setSpan(ctx.rootContext(), run.span)
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
        ...(details.taskCallID ? { "task.call_id": details.taskCallID } : {}),
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

function endInteractionSpan(
  interactionID: string,
  sessionID: string,
  status: SpanStatusCode.OK | SpanStatusCode.ERROR,
  ctx: HandlerContext,
  endTime?: number,
  error?: string,
) {
  const span = ctx.interactionSpans.get(interactionID)
  const totals = ctx.interactionTotals.get(interactionID)
  const completion = ctx.interactionCompletions.get(interactionID)
  if (span) {
    if (totals) {
      span.setAttributes({
        "interaction.total_tokens": totals.tokens,
        "interaction.total_cost_usd": totals.cost,
        "interaction.total_messages": totals.messages,
      })
    }
    if (completion?.output !== undefined) {
      span.setAttributes({
        [OUTPUT_VALUE]: completion.output,
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      })
      const run = ctx.activeRunSpans.get(sessionID)
      const interactionIO = run?.interactionIO.get(interactionID)
      if (run && interactionIO) {
        run.interactionIO.set(interactionID, { ...interactionIO, output: completion.output })
      }
    }
    span.setStatus(error ? { code: status, message: error } : { code: status })
    if (error) span.setAttribute("error", error)
    span.end(endTime)
    ctx.interactionSpans.delete(interactionID)
  }
  ctx.interactionTotals.delete(interactionID)
  ctx.interactionCompletions.delete(interactionID)
  if (ctx.activeInteractions.get(sessionID) === interactionID) {
    ctx.activeInteractions.delete(sessionID)
  }
}

function endSessionInteractions(
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

const interactionHandlers = {
  stage(
    interactionID: string,
    sessionID: string,
    agent: string,
    promptText: string,
    model: string,
    startTime: number,
    ctx: HandlerContext,
  ) {
    if (ctx.interactionSpans.has(interactionID) || ctx.interactionSpanContexts.has(interactionID)) return
    setBoundedMap(ctx.pendingInteractions, interactionID, {
      sessionID,
      agent,
      promptText,
      model,
      startTime,
    })
  },

  materialize(interactionID: string, sessionID: string, ctx: HandlerContext) {
    const pending = ctx.pendingInteractions.get(interactionID)
    if (pending?.sessionID !== sessionID) return false
    ctx.pendingInteractions.delete(interactionID)
    handleInteractionStarted(
      interactionID,
      pending.sessionID,
      pending.agent,
      pending.promptText,
      pending.model,
      pending.startTime,
      ctx,
    )
    return true
  },

  bindAssistant(assistantID: string, interactionID: string, ctx: HandlerContext) {
    setBoundedMap(ctx.assistantInteractions, assistantID, interactionID)
  },

  resolveAssistant(assistantID: string, fallbackInteractionID: string | undefined, ctx: HandlerContext) {
    return ctx.assistantInteractions.get(assistantID) ?? fallbackInteractionID
  },

  trackAssistant(
    assistantID: string,
    msgKey: string,
    sessionID: string,
    interactionID: string,
    ctx: HandlerContext,
  ) {
    setBoundedMap(ctx.assistantInteractions, assistantID, interactionID)
    setBoundedMap(ctx.pendingAssistantInteractions, msgKey, { sessionID, interactionID })
  },

  completeAssistant(msgKey: string, ctx: HandlerContext) {
    ctx.pendingAssistantInteractions.delete(msgKey)
  },

  hasPendingAssistant(msgKey: string, ctx: HandlerContext) {
    return ctx.pendingAssistantInteractions.has(msgKey)
  },

  input(interactionID: string, ctx: HandlerContext) {
    return ctx.interactionInputs.get(interactionID)
  },

  recordUsage(interactionID: string, tokens: number, cost: number, ctx: HandlerContext) {
    const existing = ctx.interactionTotals.get(interactionID)
    if (!existing) return
    setBoundedMap(ctx.interactionTotals, interactionID, {
      tokens: existing.tokens + tokens,
      cost: existing.cost + cost,
      messages: existing.messages + 1,
    })
  },

  recordCompletion(interactionID: string, endTime: number, output: string | undefined, ctx: HandlerContext) {
    if (!ctx.interactionSpans.has(interactionID)) return
    setBoundedMap(ctx.interactionCompletions, interactionID, { endTime, output })
  },

  has(interactionID: string, ctx: HandlerContext) {
    return ctx.interactionSpans.has(interactionID)
  },

  end: endInteractionSpan,
  endSession: endSessionInteractions,
}

export { createInteractionState, endInteractionSpan, handleInteractionStarted, interactionHandlers }
