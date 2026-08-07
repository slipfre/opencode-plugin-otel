import {
  SpanKind,
  SpanStatusCode,
  trace,
  type Context,
  type SpanContext,
  type SpanOptions,
} from "@opentelemetry/api"
import {
  OpenInferenceSpanKind,
  SemanticConventions,
  SESSION_ID,
} from "@arizeai/openinference-semantic-conventions"
import { MAX_PENDING, type HandlerContext } from "./types.ts"
import {
  errorSummary,
  resolveInteractionTraceContext,
  resolveSessionTraceContext,
  setBoundedMap,
} from "./util.ts"

type ErrorInfo = { name: string; data?: unknown }

type UserMessageInfo = {
  id: string
  sessionID: string
}

type PartInfo = {
  type: string
  sessionID: string
  messageID: string
  auto?: boolean
  overflow?: boolean
  metadata?: Record<string, unknown>
}

type AssistantResolution = {
  interactionID: string
  parentContext: Context
  links?: SpanOptions["links"]
  purpose: "normal" | "compaction_summary" | "continuation"
}

type CompactionState = HandlerContext["compactionsBySession"] extends Map<string, infer T> ? T : never

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND

function isContextOverflow(error: ErrorInfo | undefined) {
  return error?.name === "ContextOverflowError"
}

function endCompaction(
  state: CompactionState,
  success: boolean,
  error?: ErrorInfo,
) {
  if (!state.span || state.spanEnded) return
  state.span.setAttribute("opencode.compaction.success", success)
  if (success) {
    state.span.setStatus({ code: SpanStatusCode.OK })
  } else {
    const message = errorSummary(error)
    state.span.setAttributes({ "error": message, "opencode.compaction.recovery_error": message })
    state.span.setStatus({ code: SpanStatusCode.ERROR, message })
  }
  state.span.end()
  state.spanEnded = true
}

function setCompactionState(sessionID: string, state: CompactionState, ctx: HandlerContext) {
  if (!ctx.compactionsBySession.has(sessionID) && ctx.compactionsBySession.size >= MAX_PENDING) {
    const oldestSessionID = ctx.compactionsBySession.keys().next().value as string | undefined
    const oldest = oldestSessionID ? ctx.compactionsBySession.get(oldestSessionID) : undefined
    if (oldestSessionID && oldest) {
      endCompaction(oldest, false, {
        name: "CompactionStateEvictedError",
        data: { message: "compaction state evicted before completion" },
      })
      ctx.compactionsBySession.delete(oldestSessionID)
    }
  }
  setBoundedMap(ctx.compactionsBySession, sessionID, state)
}

function recordExternalUser(messageID: string, ctx: HandlerContext) {
  setBoundedMap(ctx.externalUserMessageIDs, messageID, true)
}

function addBoundedSet(set: Set<string>, value: string) {
  if (!set.has(value) && set.size >= MAX_PENDING) {
    const oldest = set.values().next().value as string | undefined
    if (oldest) set.delete(oldest)
  }
  set.add(value)
}

function recordUserMessage(info: UserMessageInfo, ctx: HandlerContext) {
  const state = ctx.compactionsBySession.get(info.sessionID)
  if (!state) return
  addBoundedSet(state.userMessageIDs, info.id)
}

function enterAwaitingCompaction(
  input: {
    sessionID: string
    trigger: "auto" | "overflow" | "manual"
    interactionID?: string
    originAssistantID?: string
    originAssistantSpanContext?: SpanContext
    triggerError?: ErrorInfo
    userMessageIDs?: CompactionState["userMessageIDs"]
  },
  ctx: HandlerContext,
) {
  const state: CompactionState = {
    phase: "awaiting_compaction",
    sessionID: input.sessionID,
    interactionID: input.interactionID,
    originAssistantID: input.originAssistantID,
    originAssistantSpanContext: input.originAssistantSpanContext,
    trigger: input.trigger,
    triggerError: input.triggerError,
    spanEnded: false,
    userMessageIDs: input.userMessageIDs ?? new Set(),
  }
  setCompactionState(input.sessionID, state, ctx)
  return state
}

function startCompaction(part: PartInfo, ctx: HandlerContext) {
  let current = ctx.compactionsBySession.get(part.sessionID)
  if (current?.compactionUserMessageID === part.messageID) return
  let userMessageIDs = current?.userMessageIDs
  if (current && current.phase !== "ready" && current.phase !== "awaiting_compaction") {
    const error = { name: "CompactionSupersededError", data: { message: "compaction superseded before completion" } }
    endCompaction(current, false, error)
    userMessageIDs = userMessageIDs?.has(part.messageID)
      ? new Set([part.messageID])
      : undefined
    ctx.compactionsBySession.delete(part.sessionID)
    current = undefined
  }

  const candidate = current?.phase === "ready"
    && part.auto === true
    && current.interactionID
    && ctx.interactionSpans.has(current.interactionID)
    ? current
    : undefined
  const awaiting = current?.phase === "awaiting_compaction"
    ? current
    : enterAwaitingCompaction({
        sessionID: part.sessionID,
        trigger: part.overflow === true ? "overflow" : part.auto === true ? "auto" : "manual",
        interactionID: candidate?.interactionID,
        originAssistantID: candidate?.originAssistantID,
        originAssistantSpanContext: candidate?.originAssistantSpanContext,
        userMessageIDs,
      }, ctx)
  const parentContext = awaiting.interactionID
    ? resolveInteractionTraceContext(awaiting.interactionID, ctx)
    : resolveSessionTraceContext(part.sessionID, ctx)
  const links = awaiting.originAssistantSpanContext
    ? [{ context: awaiting.originAssistantSpanContext }]
    : undefined
  const span = ctx.tracer.startSpan(
    `${ctx.tracePrefix}compaction`,
    {
      startTime: Date.now(),
      kind: SpanKind.INTERNAL,
      links,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
        [SESSION_ID]: part.sessionID,
        "opencode.compaction.id": part.messageID,
        "opencode.compaction.auto": part.auto === true,
        "opencode.compaction.overflow": awaiting.trigger === "overflow",
        "opencode.compaction.trigger": awaiting.trigger ?? "manual",
        ...(awaiting.interactionID
          ? { "opencode.compaction.origin_interaction_id": awaiting.interactionID }
          : {}),
        ...(awaiting.originAssistantID
          ? { "opencode.compaction.origin_assistant_id": awaiting.originAssistantID }
          : {}),
        ...(awaiting.triggerError
          ? { "opencode.compaction.trigger_error": errorSummary(awaiting.triggerError) }
          : {}),
        ...ctx.commonAttrs,
      },
    },
    parentContext,
  )
  setCompactionState(part.sessionID, {
    phase: "summarizing",
    sessionID: part.sessionID,
    interactionID: awaiting.interactionID,
    originAssistantID: awaiting.originAssistantID,
    originAssistantSpanContext: awaiting.originAssistantSpanContext,
    trigger: awaiting.trigger,
    compactionUserMessageID: part.messageID,
    triggerError: awaiting.triggerError,
    span,
    spanEnded: false,
    userMessageIDs: awaiting.userMessageIDs,
  }, ctx)
}

function recordPart(part: PartInfo, ctx: HandlerContext) {
  if (part.type === "compaction") {
    startCompaction(part, ctx)
    return
  }
  if (part.type !== "text" || part.metadata?.compaction_continue !== true) return
  const state = ctx.compactionsBySession.get(part.sessionID)
  if (!state || state.phase === "ready" || state.phase === "awaiting_compaction") return
  addBoundedSet(state.userMessageIDs, part.messageID)
}

function resolveAssistant(
  input: { sessionID: string; parentID: string; summary: boolean },
  ctx: HandlerContext,
): AssistantResolution {
  const state = ctx.compactionsBySession.get(input.sessionID)
  if (input.summary && state?.compactionUserMessageID === input.parentID) {
    const interactionID = state.interactionID ?? input.parentID
    const parentContext = state.span
      ? trace.setSpan(ctx.rootContext(), state.span)
      : state.interactionID
        ? resolveInteractionTraceContext(state.interactionID, ctx)
        : resolveSessionTraceContext(input.sessionID, ctx)
    return { interactionID, parentContext, purpose: "compaction_summary" }
  }

  const alias = ctx.internalUserInteractions.get(input.parentID)
  if (alias) {
    return {
      interactionID: alias.interactionID,
      parentContext: resolveInteractionTraceContext(alias.interactionID, ctx),
      links: alias.compactionSpanContext ? [{ context: alias.compactionSpanContext }] : undefined,
      purpose: "continuation",
    }
  }

  return {
    interactionID: input.parentID,
    parentContext: resolveInteractionTraceContext(input.parentID, ctx),
    purpose: "normal",
  }
}

function deferredError(sessionID: string, assistantID: string, ctx: HandlerContext) {
  return ctx.deferredAssistantErrors.get(`${sessionID}:${assistantID}`)
}

function completeAssistant(
  input: {
    sessionID: string
    assistantID: string
    parentID: string
    interactionID: string
    summary: boolean
    error?: ErrorInfo
    spanContext?: SpanContext
  },
  ctx: HandlerContext,
) {
  ctx.deferredAssistantErrors.delete(`${input.sessionID}:${input.assistantID}`)
  const state = ctx.compactionsBySession.get(input.sessionID)
  if (input.summary && state?.compactionUserMessageID === input.parentID) {
    state.summaryAssistantID = input.assistantID
    state.span?.setAttribute("opencode.compaction.summary_assistant_id", input.assistantID)
    if (input.error) {
      state.recoveryError = input.error
      state.phase = "failing"
      endCompaction(state, false, input.error)
    } else {
      state.phase = "awaiting_resume"
    }
    return { keepInteractionOpen: true }
  }

  if (!input.summary && !input.error && (!state || state.phase === "ready")) {
    setCompactionState(input.sessionID, {
      phase: "ready",
      sessionID: input.sessionID,
      interactionID: input.interactionID,
      originAssistantID: input.assistantID,
      originAssistantSpanContext: input.spanContext,
      spanEnded: false,
      userMessageIDs: new Set(),
    }, ctx)
  }

  const recoverableOverflow = state?.phase === "awaiting_compaction"
    && state.originAssistantID === input.assistantID
  return { keepInteractionOpen: recoverableOverflow }
}

function deferSessionError(sessionID: string, error: ErrorInfo, ctx: HandlerContext) {
  const state = ctx.compactionsBySession.get(sessionID)
  const active = ctx.activeMessageSpans.get(sessionID)
  if (state && state.phase !== "ready") {
    if (state.phase === "awaiting_compaction" && isContextOverflow(error)) {
      if (active) setBoundedMap(ctx.deferredAssistantErrors, `${sessionID}:${active.messageID}`, error)
      state.triggerError ??= error
      return true
    }
    state.recoveryError = error
    state.phase = "failing"
    if (active) setBoundedMap(ctx.deferredAssistantErrors, `${sessionID}:${active.messageID}`, error)
    return true
  }

  if (!isContextOverflow(error) || !active || active.summary
    || !ctx.interactionSpans.has(active.interactionID)) {
    if (state?.phase === "ready") ctx.compactionsBySession.delete(sessionID)
    return false
  }
  setBoundedMap(ctx.deferredAssistantErrors, `${sessionID}:${active.messageID}`, error)
  enterAwaitingCompaction({
    sessionID,
    trigger: "overflow",
    interactionID: active.interactionID,
    originAssistantID: active.messageID,
    originAssistantSpanContext: active.span.spanContext(),
    triggerError: error,
    userMessageIDs: state?.userMessageIDs,
  }, ctx)
  return true
}

function handleSessionCompacted(sessionID: string, ctx: HandlerContext) {
  const state = ctx.compactionsBySession.get(sessionID)
  if (!state || state.phase === "ready" || state.phase === "awaiting_compaction") return
  if (state.phase === "failing") {
    endCompaction(state, false, state.recoveryError ?? state.triggerError)
    return
  }

  const candidates = [...state.userMessageIDs]
    .filter(messageID =>
      messageID !== state.compactionUserMessageID
      && !ctx.externalUserMessageIDs.has(messageID)
    )
  state.span?.setAttribute("opencode.compaction.resume_candidate_count", candidates.length)
  if (candidates.length === 1 && state.interactionID) {
    const continuationMessageID = candidates[0]!
    setBoundedMap(ctx.internalUserInteractions, continuationMessageID, {
      interactionID: state.interactionID,
      compactionSpanContext: state.span?.spanContext(),
    })
    state.span?.setAttribute("opencode.compaction.continuation_message_id", continuationMessageID)
  } else if (candidates.length > 1) {
    state.span?.setAttribute("opencode.compaction.correlation_miss", true)
  }
  endCompaction(state, true)
  ctx.compactionsBySession.delete(sessionID)
}

function finalizeOnIdle(sessionID: string, ctx: HandlerContext) {
  const state = ctx.compactionsBySession.get(sessionID)
  if (!state) return
  if (state.phase === "ready") {
    ctx.compactionsBySession.delete(sessionID)
    return
  }
  const error = state.recoveryError
    ?? state.triggerError
    ?? { name: "CompactionInterruptedError", data: { message: "compaction ended before completion" } }
  const active = ctx.activeMessageSpans.get(sessionID)
  if (active) setBoundedMap(ctx.deferredAssistantErrors, `${sessionID}:${active.messageID}`, error)
  endCompaction(state, false, error)
  ctx.compactionsBySession.delete(sessionID)
  return errorSummary(error)
}

/** Coordinates recoverable compaction state and exact message-to-interaction correlation. */
export const compactionHandlers = {
  recordExternalUser,
  recordUserMessage,
  recordPart,
  resolveAssistant,
  deferredError,
  completeAssistant,
  deferSessionError,
  handleSessionCompacted,
  finalizeOnIdle,
}
