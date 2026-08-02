import { SpanStatusCode, SpanKind, type Span } from "@opentelemetry/api"
import type { AssistantMessage, EventMessageUpdated, EventMessagePartUpdated, ToolPart } from "@opencode-ai/sdk"
import {
  AGENT_NAME,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_COST_TOTAL,
  LLM_INPUT_MESSAGES,
  LLM_MODEL_NAME,
  LLM_OUTPUT_MESSAGES,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  LLM_TOKEN_COUNT_TOTAL,
  MimeType,
  MESSAGE_CONTENT,
  MESSAGE_ROLE,
  OpenInferenceSpanKind,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  SemanticConventions,
  SESSION_ID,
  TOOL_ID,
  TOOL_NAME,
  TOOL_PARAMETERS,
} from "@arizeai/openinference-semantic-conventions"
import {
  errorSummary,
  genAiProviderName,
  setBoundedMap,
  accumulateInteractionTotals,
  isTraceEnabled,
  resolveSessionTraceContext,
} from "../util.ts"
import type { HandlerContext, RunDetails, SessionAgentType } from "../types.ts"
import { endInteractionSpan } from "../interaction.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND
const LLM_FINISH_REASON = "llm.finish_reason"

function getRunAgentMeta(
  sessionID: string,
  ctx: HandlerContext,
): { agentName: string; agentType: SessionAgentType } {
  const run = ctx.activeRunSpans.get(sessionID)
  return {
    agentName: run?.agent ?? "unknown",
    agentType: run?.agentType
      ?? ctx.pendingSubagentRuns.get(sessionID)?.agentType
      ?? (ctx.sessionParents.has(sessionID) ? "subagent" : "primary"),
  }
}

function taskMetadata(toolPart: ToolPart) {
  if (toolPart.tool !== "task" || !("metadata" in toolPart.state)) return
  const metadata = toolPart.state.metadata
  if (!metadata || metadata.background === true || typeof metadata.sessionId !== "string") return
  const input = toolPart.state.input
  const agent = input && typeof input.subagent_type === "string"
    ? input.subagent_type
    : undefined
  return {
    childSessionID: metadata.sessionId,
    parentSessionID: typeof metadata.parentSessionId === "string" ? metadata.parentSessionId : toolPart.sessionID,
    agent,
  }
}

function removePendingSubagentRun(sessionID: string, taskCallID: string, ctx: HandlerContext) {
  if (ctx.pendingSubagentRuns.get(sessionID)?.taskCallID === taskCallID) {
    ctx.pendingSubagentRuns.delete(sessionID)
  }
}

function bindSubagentRun(toolPart: ToolPart, toolSpan: Span | undefined, ctx: HandlerContext) {
  const task = taskMetadata(toolPart)
  if (!task) return
  toolSpan?.setAttributes({
    "subagent.session.id": task.childSessionID,
    ...(task.agent ? { "subagent.agent.name": task.agent } : {}),
  })
  const existing = ctx.pendingSubagentRuns.get(task.childSessionID)
  setBoundedMap(ctx.pendingSubagentRuns, task.childSessionID, {
    agentType: "subagent",
    parentSessionID: task.parentSessionID,
    taskCallID: toolPart.callID,
    taskSpanContext: toolSpan?.spanContext() ?? existing?.taskSpanContext,
  })
}

function recordLlmOutputEnd(toolPart: ToolPart, outputEndTime: number, ctx: HandlerContext) {
  const active = ctx.activeMessageSpans.get(toolPart.sessionID)
  if (!active || active.messageID !== toolPart.messageID) return
  setBoundedMap(ctx.activeMessageSpans, toolPart.sessionID, {
    ...active,
    outputEndTime: Math.max(active.outputEndTime ?? 0, outputEndTime),
  })
}

/** Completes an assistant message span and accumulates its usage on the active run and interaction. */
export function handleMessageUpdated(e: EventMessageUpdated, ctx: HandlerContext) {
  const msg = e.properties.info
  if (msg.role !== "assistant") return
  const assistant = msg as AssistantMessage
  setBoundedMap(ctx.assistantInteractions, assistant.id, assistant.parentID)
  if (!assistant.time.completed) return

  const { sessionID } = assistant
  const msgKey = `${sessionID}:${assistant.id}`
  const activeMessage = ctx.activeMessageSpans.get(sessionID)
  const recordedOutputEndTime = activeMessage?.messageID === assistant.id
    ? activeMessage.outputEndTime ?? assistant.time.completed
    : assistant.time.completed
  const outputEndTime = Math.min(
    assistant.time.completed,
    Math.max(assistant.time.created, recordedOutputEndTime),
  )
  const duration = outputEndTime - assistant.time.created
  const run = ctx.activeRunSpans.get(sessionID)
  const runAgent = getRunAgentMeta(sessionID, ctx)
  const messageAgent = (assistant as AssistantMessage & { agent?: string }).agent ?? assistant.mode
  const agentName = messageAgent || runAgent.agentName
  const agentType = runAgent.agentType
  if (messageAgent && run) {
    run.agent = messageAgent
    run.span.setAttribute(AGENT_NAME, messageAgent)
  }
  const promptTokens = assistant.tokens.input + assistant.tokens.cache.read + assistant.tokens.cache.write
  const completionTokens = assistant.tokens.output + assistant.tokens.reasoning
  const totalTokens = promptTokens + completionTokens

  if (run) {
    run.tokens += totalTokens
    run.cost += assistant.cost
    run.messages += 1
  }
  const interactionID = ctx.assistantInteractions.get(assistant.id) ?? assistant.parentID
  accumulateInteractionTotals(interactionID, totalTokens, assistant.cost, ctx)

  const outputText = ctx.messageOutputs.get(msgKey)
  if (assistant.summary !== true && ctx.interactionSpans.has(interactionID)) {
    setBoundedMap(ctx.interactionCompletions, interactionID, {
      endTime: assistant.time.completed,
      output: outputText,
    })
  }
  const msgSpan = ctx.messageSpans.get(msgKey)
  if (msgSpan) {
    const telemetryOutput = ctx.llmTelemetryOutputs.has(msgKey)
    msgSpan.setAttributes({
      [AGENT_NAME]: agentName,
      "agent.type": agentType,
      [LLM_TOKEN_COUNT_PROMPT]: promptTokens,
      [LLM_TOKEN_COUNT_COMPLETION]: completionTokens,
      [LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]: assistant.tokens.reasoning,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: assistant.tokens.cache.read,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: assistant.tokens.cache.write,
      [LLM_TOKEN_COUNT_TOTAL]: totalTokens,
      [LLM_FINISH_REASON]: assistant.error ? "error" : (assistant.finish ?? "stop"),
      [LLM_COST_TOTAL]: assistant.cost,
      ...(outputText && !telemetryOutput
        ? {
            [OUTPUT_VALUE]: outputText,
            [OUTPUT_MIME_TYPE]: MimeType.TEXT,
            [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "assistant",
            [`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: outputText,
          }
        : {}),
      cost_usd: assistant.cost,
      duration_ms: duration,
    })
    if (assistant.error) {
      msgSpan.setStatus({ code: SpanStatusCode.ERROR, message: errorSummary(assistant.error) })
    } else {
      msgSpan.setStatus({ code: SpanStatusCode.OK })
    }
    msgSpan.end(outputEndTime)
    ctx.messageSpans.delete(msgKey)
  }
  const requestKey = `${sessionID}:${assistant.parentID}`
  const remainingRequests = ctx.llmRequestContexts.get(requestKey)?.filter(request => request.messageID !== assistant.id)
  if (remainingRequests?.length) {
    setBoundedMap(ctx.llmRequestContexts, requestKey, remainingRequests)
  } else {
    ctx.llmRequestContexts.delete(requestKey)
  }
  ctx.messageOutputs.delete(msgKey)
  if (ctx.activeMessageSpans.get(sessionID)?.messageID === assistant.id) {
    ctx.activeMessageSpans.delete(sessionID)
  }
  ctx.llmTelemetryOutputs.delete(msgKey)
  ctx.pendingAssistantInteractions.delete(msgKey)

  if (assistant.error || (!ctx.activeRunSpans.has(sessionID) && ctx.interactionSpans.has(interactionID))) {
    const interactionError = assistant.error ? errorSummary(assistant.error) : undefined
    endInteractionSpan(
      interactionID,
      sessionID,
      interactionError ? SpanStatusCode.ERROR : SpanStatusCode.OK,
      ctx,
      assistant.time.completed,
      interactionError,
    )
  }

}

/**
 * Manages tool child spans between `running` and `completed`/`error` part updates.
 *
 * For tool spans: on `running` a child span of the current run span is started and stored
 * in `pendingToolSpans`. On `completed`/`error` the span is ended with appropriate status.
 * If no `running` event was seen (out-of-order), a best-effort span is started and immediately ended.
 */
export function handleMessagePartUpdated(e: EventMessagePartUpdated, ctx: HandlerContext) {
  const part = e.properties.part

  if (part.type === "text") {
    const key = `${part.sessionID}:${part.messageID}`
    ctx.messageOutputs.set(key, `${ctx.messageOutputs.get(key) ?? ""}${part.text}`)
    return
  }

  if (part.type === "tool") {
    const toolPart = part as ToolPart
    const key = `${toolPart.sessionID}:${toolPart.callID}`

    if (toolPart.state.status === "running") {
      const pending = ctx.pendingToolSpans.get(key)
      if (pending) {
        pending.span?.setAttributes({
          [TOOL_PARAMETERS]: JSON.stringify(toolPart.state.input),
          [INPUT_VALUE]: JSON.stringify(toolPart.state.input),
        })
        bindSubagentRun(toolPart, pending.span, ctx)
        return
      }
      recordLlmOutputEnd(toolPart, toolPart.state.time.start, ctx)
      const { agentName, agentType } = getRunAgentMeta(toolPart.sessionID, ctx)
      const toolSpan = isTraceEnabled("tool", ctx)
        ? (() => {
            return ctx.tracer.startSpan(
              `${ctx.tracePrefix}tool.${toolPart.tool}`,
              {
                startTime: toolPart.state.time.start,
                kind: SpanKind.INTERNAL,
                attributes: {
                  [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
                  [SESSION_ID]: toolPart.sessionID,
                  [TOOL_ID]: toolPart.callID,
                  [TOOL_NAME]: toolPart.tool,
                  [TOOL_PARAMETERS]: JSON.stringify(toolPart.state.input),
                  [INPUT_VALUE]: JSON.stringify(toolPart.state.input),
                  [INPUT_MIME_TYPE]: MimeType.JSON,
                  [AGENT_NAME]: agentName,
                  "agent.type": agentType,
                  ...ctx.commonAttrs,
                },
              },
              resolveSessionTraceContext(toolPart.sessionID, ctx, {
                assistantMessageID: toolPart.messageID,
              }),
            )
          })()
        : undefined
      setBoundedMap(ctx.pendingToolSpans, key, {
        tool: toolPart.tool,
        sessionID: toolPart.sessionID,
        startMs: toolPart.state.time.start,
        span: toolSpan,
      })
      bindSubagentRun(toolPart, toolSpan, ctx)
      return
    }

    if (toolPart.state.status !== "completed" && toolPart.state.status !== "error") return

    const pending = ctx.pendingToolSpans.get(key)
    ctx.pendingToolSpans.delete(key)
    const start = pending?.startMs ?? toolPart.state.time.start
    const end = toolPart.state.time.end
    if (end === undefined) return
    const success = toolPart.state.status === "completed"
    const { agentName, agentType } = getRunAgentMeta(toolPart.sessionID, ctx)
    const task = taskMetadata(toolPart)
    if (task) removePendingSubagentRun(task.childSessionID, toolPart.callID, ctx)

    if (isTraceEnabled("tool", ctx)) {
      const toolSpan = pending?.span ?? (() => {
        return ctx.tracer.startSpan(
          `${ctx.tracePrefix}tool.${toolPart.tool}`,
          {
            startTime: start,
            kind: SpanKind.INTERNAL,
            attributes: {
              [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
              [SESSION_ID]: toolPart.sessionID,
              [TOOL_ID]: toolPart.callID,
              [TOOL_NAME]: toolPart.tool,
              [TOOL_PARAMETERS]: JSON.stringify(toolPart.state.input),
              [INPUT_VALUE]: JSON.stringify(toolPart.state.input),
              [INPUT_MIME_TYPE]: MimeType.JSON,
              ...ctx.commonAttrs,
            },
          },
          resolveSessionTraceContext(toolPart.sessionID, ctx, {
            assistantMessageID: toolPart.messageID,
          }),
        )
      })()
      toolSpan.setAttributes({ [AGENT_NAME]: agentName, "agent.type": agentType })
      toolSpan.setAttribute("tool.success", success)
      if (success) {
        const output = (toolPart.state as { output: string }).output
        toolSpan.setAttributes({
          [OUTPUT_VALUE]: output,
          [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        })
        toolSpan.setAttribute("tool.result_size_bytes", Buffer.byteLength(output, "utf8"))
        toolSpan.setStatus({ code: SpanStatusCode.OK })
      } else {
        const err = (toolPart.state as { error: string }).error
        toolSpan.setAttributes({
          [OUTPUT_VALUE]: err,
          [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        })
        toolSpan.setAttribute("tool.error", err)
        toolSpan.setStatus({ code: SpanStatusCode.ERROR, message: err })
      }
      toolSpan.end(end)
    }

  }
}

/**
 * Starts an LLM span for an assistant message when it first appears in `message.updated`.
 * The span is parented to the active run and carries `gen_ai.*` semantic
 * attributes for the model and provider. It is ended in `handleMessageUpdated` once the
 * message completes.
 *
 * Only called for assistant messages that have not yet completed (`time.completed` absent).
 */
export function startMessageSpan(
  sessionID: string,
  messageID: string,
  parentID: string,
  modelID: string,
  providerID: string,
  startTime: number,
  ctx: HandlerContext,
  messageAgent?: string,
) {
  const msgKey = `${sessionID}:${messageID}`
  setBoundedMap(ctx.assistantInteractions, messageID, parentID)
  setBoundedMap(ctx.pendingAssistantInteractions, msgKey, { sessionID, interactionID: parentID })
  if (!isTraceEnabled("llm", ctx)) return
  if (ctx.messageSpans.has(msgKey)) return
  const run = ctx.activeRunSpans.get(sessionID)
  const runAgent = getRunAgentMeta(sessionID, ctx)
  const agentName = messageAgent || runAgent.agentName
  const agentType = runAgent.agentType
  if (messageAgent && run) {
    run.agent = messageAgent
    run.span.setAttribute(AGENT_NAME, messageAgent)
  }
  const inputText = ctx.interactionInputs.get(parentID)

  const msgSpan = ctx.tracer.startSpan(
    `${ctx.tracePrefix}llm`,
    {
      startTime,
      kind: SpanKind.CLIENT,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        [SESSION_ID]: sessionID,
        [AGENT_NAME]: agentName,
        "agent.type": agentType,
        [LLM_SYSTEM]: providerID,
        [LLM_PROVIDER]: providerID,
        "gen_ai.provider.name": genAiProviderName(providerID),
        [LLM_MODEL_NAME]: modelID,
        [OUTPUT_VALUE]: "",
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        ...(inputText
          ? {
              [INPUT_VALUE]: inputText,
              [INPUT_MIME_TYPE]: MimeType.TEXT,
              [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]: "user",
              [`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]: inputText,
            }
          : {}),
        ...ctx.commonAttrs,
      },
    },
    resolveSessionTraceContext(sessionID, ctx, { interactionID: parentID, assistantMessageID: messageID }),
  )
  setBoundedMap(ctx.messageSpans, msgKey, msgSpan)
  const requestKey = `${sessionID}:${parentID}`
  setBoundedMap(ctx.llmRequestContexts, requestKey, [
    ...(ctx.llmRequestContexts.get(requestKey) ?? []),
    {
      messageID,
      agent: messageAgent ?? agentName,
      modelID,
      providerID,
      spanContext: msgSpan.spanContext(),
    },
  ])
  setBoundedMap(ctx.activeMessageSpans, sessionID, { messageID, span: msgSpan })
}
