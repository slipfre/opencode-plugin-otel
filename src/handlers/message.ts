import { SpanStatusCode, SpanKind, type Span } from "@opentelemetry/api";
import type {
  AssistantMessage,
  EventMessageUpdated,
  EventMessagePartUpdated,
  ToolPart,
} from "@opencode-ai/sdk";
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
} from "@arizeai/openinference-semantic-conventions";
import {
  errorSummary,
  genAiProviderName,
  setBoundedMap,
  resolveSessionTraceContext,
  tryResolveInteractionTraceContext,
} from "../util.ts";
import type { HandlerContext, SessionAgentType } from "../types.ts";
import { interactionHandlers } from "../interaction.ts";
import { compactionHandlers } from "../compaction.ts";
import { permissionHandlers } from "./permission.ts";

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND;
const LLM_FINISH_REASON = "llm.finish_reason";

function resolveToolTraceContext(
  sessionID: string,
  assistantMessageID: string,
  ctx: HandlerContext
) {
  const interactionID = interactionHandlers.resolveAssistant(
    assistantMessageID,
    undefined,
    ctx
  );
  if (interactionID) {
    const interactionContext = tryResolveInteractionTraceContext(
      interactionID,
      ctx
    );
    if (interactionContext) {
      return interactionContext;
    }
  }
  return resolveSessionTraceContext(sessionID, ctx);
}

function getRunAgentMeta(
  sessionID: string,
  ctx: HandlerContext
): { agentName: string; agentType: SessionAgentType } {
  const run = ctx.activeRunSpans.get(sessionID);
  return {
    agentName: run?.agent ?? "unknown",
    agentType:
      run?.agentType ??
      ctx.pendingSubagentRuns.get(sessionID)?.agentType ??
      (ctx.sessionParents.has(sessionID) ? "subagent" : "primary"),
  };
}

function taskMetadata(toolPart: ToolPart) {
  if (toolPart.tool !== "task" || !("metadata" in toolPart.state)) {
    return;
  }
  const metadata = toolPart.state.metadata;
  if (
    !metadata ||
    metadata.background === true ||
    typeof metadata.sessionId !== "string"
  ) {
    return;
  }
  const input = toolPart.state.input;
  const agent =
    input && typeof input.subagent_type === "string"
      ? input.subagent_type
      : undefined;
  return {
    childSessionID: metadata.sessionId,
    parentSessionID:
      typeof metadata.parentSessionId === "string"
        ? metadata.parentSessionId
        : toolPart.sessionID,
    agent,
  };
}

function removePendingSubagentRun(
  sessionID: string,
  taskCallID: string,
  ctx: HandlerContext
) {
  if (ctx.pendingSubagentRuns.get(sessionID)?.taskCallID === taskCallID) {
    ctx.pendingSubagentRuns.delete(sessionID);
  }
}

function bindSubagentRun(
  toolPart: ToolPart,
  toolSpan: Span,
  ctx: HandlerContext
) {
  const task = taskMetadata(toolPart);
  if (!task) {
    return;
  }
  toolSpan.setAttributes({
    "subagent.session.id": task.childSessionID,
    ...(task.agent ? { "subagent.agent.name": task.agent } : {}),
  });
  setBoundedMap(ctx.pendingSubagentRuns, task.childSessionID, {
    agentType: "subagent",
    parentSessionID: task.parentSessionID,
    taskCallID: toolPart.callID,
    taskSpanContext: toolSpan.spanContext(),
  });
}

function recordLlmOutputEnd(
  toolPart: ToolPart,
  outputEndTime: number,
  ctx: HandlerContext
) {
  const active = ctx.activeMessageSpans.get(toolPart.sessionID);
  if (!active || active.messageID !== toolPart.messageID) {
    return;
  }
  setBoundedMap(ctx.activeMessageSpans, toolPart.sessionID, {
    ...active,
    outputEndTime: Math.max(active.outputEndTime ?? 0, outputEndTime),
  });
}

/** Completes an assistant message span and accumulates its usage on the active run and interaction. */
export function handleMessageUpdated(
  e: EventMessageUpdated,
  ctx: HandlerContext
) {
  const msg = e.properties.info;
  if (msg.role !== "assistant") {
    return;
  }
  const assistant = msg as AssistantMessage;
  const interactionID =
    interactionHandlers.resolveAssistant(
      assistant.id,
      assistant.parentID,
      ctx
    ) ??
    compactionHandlers.recoverOwner(
      assistant.sessionID,
      assistant.parentID,
      ctx
    );
  interactionHandlers.bindAssistant(assistant.id, interactionID, ctx);
  if (!assistant.time.completed) {
    return;
  }

  const { sessionID } = assistant;
  const msgKey = `${sessionID}:${assistant.id}`;
  const activeMessage = ctx.activeMessageSpans.get(sessionID);
  const recordedOutputEndTime =
    activeMessage?.messageID === assistant.id
      ? (activeMessage.outputEndTime ?? assistant.time.completed)
      : assistant.time.completed;
  const outputEndTime = Math.min(
    assistant.time.completed,
    Math.max(assistant.time.created, recordedOutputEndTime)
  );
  const duration = outputEndTime - assistant.time.created;
  const run = ctx.activeRunSpans.get(sessionID);
  const runAgent = getRunAgentMeta(sessionID, ctx);
  const messageAgent =
    (assistant as AssistantMessage & { agent?: string }).agent ??
    assistant.mode;
  const agentName = messageAgent || runAgent.agentName;
  const agentType = runAgent.agentType;
  const contextOverflow = compactionHandlers.contextOverflowForMessage(
    sessionID,
    assistant.id,
    ctx
  );
  const assistantError = assistant.error
    ? errorSummary(assistant.error)
    : contextOverflow?.error;
  if (messageAgent && run && assistant.summary !== true) {
    run.agent = messageAgent;
    run.span.setAttribute(AGENT_NAME, messageAgent);
  }
  const promptTokens =
    assistant.tokens.input +
    assistant.tokens.cache.read +
    assistant.tokens.cache.write;
  const completionTokens = assistant.tokens.output + assistant.tokens.reasoning;
  const totalTokens = promptTokens + completionTokens;

  if (run) {
    run.tokens += totalTokens;
    run.cost += assistant.cost;
    run.messages += 1;
  }
  if (interactionID) {
    interactionHandlers.recordUsage(
      interactionID,
      totalTokens,
      assistant.cost,
      ctx
    );
  }

  const outputText = ctx.messageOutputs.get(msgKey);
  if (assistant.summary !== true && interactionID) {
    interactionHandlers.recordCompletion(
      interactionID,
      assistant.time.completed,
      outputText,
      ctx
    );
  }
  const msgSpan = ctx.messageSpans.get(msgKey);
  if (msgSpan) {
    const telemetryOutput = ctx.llmTelemetryOutputs.has(msgKey);
    msgSpan.setAttributes({
      [AGENT_NAME]: agentName,
      "agent.type": agentType,
      [LLM_TOKEN_COUNT_PROMPT]: promptTokens,
      [LLM_TOKEN_COUNT_COMPLETION]: completionTokens,
      [LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]:
        assistant.tokens.reasoning,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: assistant.tokens.cache.read,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]:
        assistant.tokens.cache.write,
      [LLM_TOKEN_COUNT_TOTAL]: totalTokens,
      [LLM_FINISH_REASON]: assistantError
        ? "error"
        : (assistant.finish ?? "stop"),
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
      ...(contextOverflow ? { "error.type": "ContextOverflowError" } : {}),
    });
    if (assistantError) {
      msgSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: assistantError,
      });
    } else {
      msgSpan.setStatus({ code: SpanStatusCode.OK });
    }
    msgSpan.end(outputEndTime);
    ctx.messageSpans.delete(msgKey);
  }
  const requestKey = `${sessionID}:${assistant.parentID}`;
  const remainingRequests = ctx.llmRequestContexts
    .get(requestKey)
    ?.filter((request) => request.messageID !== assistant.id);
  if (remainingRequests?.length) {
    setBoundedMap(ctx.llmRequestContexts, requestKey, remainingRequests);
  } else {
    ctx.llmRequestContexts.delete(requestKey);
  }
  ctx.messageOutputs.delete(msgKey);
  ctx.llmSpanStartTimes.delete(msgKey);
  if (ctx.activeMessageSpans.get(sessionID)?.messageID === assistant.id) {
    ctx.activeMessageSpans.delete(sessionID);
  }
  ctx.llmTelemetryOutputs.delete(msgKey);
  interactionHandlers.completeAssistant(msgKey, ctx);

  const compaction = compactionHandlers.resolve(
    sessionID,
    assistant.parentID,
    ctx
  );
  if (assistant.error && compaction) {
    compactionHandlers.fail(sessionID, errorSummary(assistant.error), ctx);
  }

  if (
    interactionID &&
    (assistant.error ||
      (!ctx.activeRunSpans.has(sessionID) &&
        interactionHandlers.has(interactionID, ctx)))
  ) {
    const interactionError = assistant.error
      ? errorSummary(assistant.error)
      : undefined;
    interactionHandlers.end(
      interactionID,
      sessionID,
      interactionError ? SpanStatusCode.ERROR : SpanStatusCode.OK,
      ctx,
      assistant.time.completed,
      interactionError
    );
  }
}

/**
 * Manages tool child spans between `running` and `completed`/`error` part updates.
 *
 * For tool spans: on `running` a child span of the current run span is started and stored
 * in `pendingToolSpans`. On `completed`/`error` the span is ended with appropriate status.
 * If no `running` event was seen (out-of-order), a best-effort span is started and immediately ended.
 */
export function handleMessagePartUpdated(
  e: EventMessagePartUpdated,
  ctx: HandlerContext
) {
  const part = e.properties.part;

  if (compactionHandlers.handlePart(part, ctx)) {
    return;
  }

  if (part.type === "step-start") {
    const key = `${part.sessionID}:${part.messageID}`;
    const startTime = ctx.llmSpanStartTimes.get(key);
    const span = ctx.messageSpans.get(key);
    if (startTime === undefined || !span) {
      return;
    }
    const eventTime = (e.properties as { time?: unknown }).time;
    const firstChunkTime =
      typeof eventTime === "number" && Number.isFinite(eventTime)
        ? eventTime
        : Date.now();
    const timeToFirstChunk = firstChunkTime - startTime;
    if (timeToFirstChunk >= 0) {
      span.setAttribute(
        "opencode.llm.time_to_first_chunk_ms",
        timeToFirstChunk
      );
      ctx.llmSpanStartTimes.delete(key);
    }
    return;
  }

  if (part.type === "text") {
    const key = `${part.sessionID}:${part.messageID}`;
    ctx.messageOutputs.set(
      key,
      `${ctx.messageOutputs.get(key) ?? ""}${part.text}`
    );
    return;
  }

  if (part.type === "tool") {
    const toolPart = part as ToolPart;
    const key = `${toolPart.sessionID}:${toolPart.callID}`;

    if (toolPart.state.status === "running") {
      const pending = ctx.pendingToolSpans.get(key);
      if (pending) {
        pending.span.setAttributes({
          [TOOL_PARAMETERS]: JSON.stringify(toolPart.state.input),
          [INPUT_VALUE]: JSON.stringify(toolPart.state.input),
        });
        bindSubagentRun(toolPart, pending.span, ctx);
        return;
      }
      recordLlmOutputEnd(toolPart, toolPart.state.time.start, ctx);
      const { agentName, agentType } = getRunAgentMeta(toolPart.sessionID, ctx);
      const toolSpan = ctx.tracer.startSpan(
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
        resolveToolTraceContext(toolPart.sessionID, toolPart.messageID, ctx)
      );
      setBoundedMap(ctx.pendingToolSpans, key, {
        tool: toolPart.tool,
        sessionID: toolPart.sessionID,
        startMs: toolPart.state.time.start,
        span: toolSpan,
      });
      bindSubagentRun(toolPart, toolSpan, ctx);
      return;
    }

    if (
      toolPart.state.status !== "completed" &&
      toolPart.state.status !== "error"
    ) {
      return;
    }

    const pending = ctx.pendingToolSpans.get(key);
    ctx.pendingToolSpans.delete(key);
    const start = pending?.startMs ?? toolPart.state.time.start;
    const end = toolPart.state.time.end;
    if (end === undefined) {
      return;
    }
    permissionHandlers.endTool(
      toolPart.sessionID,
      toolPart.callID,
      ctx,
      "tool ended before permission reply"
    );
    const success = toolPart.state.status === "completed";
    const { agentName, agentType } = getRunAgentMeta(toolPart.sessionID, ctx);
    const task = taskMetadata(toolPart);
    if (task) {
      removePendingSubagentRun(task.childSessionID, toolPart.callID, ctx);
    }

    const toolSpan =
      pending?.span ??
      (() => {
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
          resolveToolTraceContext(toolPart.sessionID, toolPart.messageID, ctx)
        );
      })();
    toolSpan.setAttributes({
      [AGENT_NAME]: agentName,
      "agent.type": agentType,
    });
    toolSpan.setAttribute("tool.success", success);
    if (success) {
      const output = (toolPart.state as { output: string }).output;
      toolSpan.setAttributes({
        [OUTPUT_VALUE]: output,
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      });
      toolSpan.setAttribute(
        "tool.result_size_bytes",
        Buffer.byteLength(output, "utf8")
      );
      toolSpan.setStatus({ code: SpanStatusCode.OK });
    } else {
      const err = (toolPart.state as { error: string }).error;
      toolSpan.setAttributes({
        [OUTPUT_VALUE]: err,
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      });
      if (pending?.errorType) {
        toolSpan.setAttribute("error.type", pending.errorType);
      }
      toolSpan.setStatus({ code: SpanStatusCode.ERROR, message: err });
    }
    toolSpan.end(end);
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
  messageAgent?: string
) {
  const msgKey = `${sessionID}:${messageID}`;
  const compaction = compactionHandlers.resolve(sessionID, parentID, ctx);
  const interactionID =
    compaction?.ownerInteractionID ??
    interactionHandlers.owner(parentID, sessionID, ctx) ??
    compactionHandlers.recoverOwner(sessionID, parentID, ctx);
  interactionHandlers.trackAssistant(
    messageID,
    msgKey,
    sessionID,
    interactionID,
    ctx
  );
  if (ctx.messageSpans.has(msgKey)) {
    return;
  }
  const run = ctx.activeRunSpans.get(sessionID);
  const runAgent = getRunAgentMeta(sessionID, ctx);
  const agentName = messageAgent || runAgent.agentName;
  const agentType = runAgent.agentType;
  if (messageAgent && run && !compaction) {
    run.agent = messageAgent;
    run.span.setAttribute(AGENT_NAME, messageAgent);
  }
  const inputText = interactionHandlers.input(parentID, ctx);
  const parentContext =
    compaction?.parentContext ??
    (interactionID
      ? tryResolveInteractionTraceContext(interactionID, ctx)
      : undefined) ??
    resolveSessionTraceContext(sessionID, ctx);

  const msgSpan = ctx.tracer.startSpan(
    `${ctx.tracePrefix}llm`,
    {
      startTime,
      kind: SpanKind.CLIENT,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        [SESSION_ID]: sessionID,
        "opencode.message.id": messageID,
        "opencode.llm.retry_count": 0,
        [AGENT_NAME]: agentName,
        "agent.type": agentType,
        [LLM_SYSTEM]: providerID,
        [LLM_PROVIDER]: providerID,
        "gen_ai.provider.name": genAiProviderName(providerID),
        [LLM_MODEL_NAME]: modelID,
        [LLM_TOKEN_COUNT_PROMPT]: 0,
        [LLM_TOKEN_COUNT_COMPLETION]: 0,
        [LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]: 0,
        [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: 0,
        [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: 0,
        [LLM_TOKEN_COUNT_TOTAL]: 0,
        [LLM_FINISH_REASON]: "unknown",
        [LLM_COST_TOTAL]: 0,
        cost_usd: 0,
        duration_ms: 0,
        [OUTPUT_VALUE]: "",
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        ...(compaction
          ? {
              "opencode.compaction.id": compaction.markerMessageID,
              "opencode.llm.purpose": "compaction",
              "opencode.compaction.overflow": compaction.overflow,
              ...(compaction.triggerMessageID
                ? {
                    "opencode.compaction.trigger_message.id":
                      compaction.triggerMessageID,
                  }
                : {}),
            }
          : {}),
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
    parentContext
  );
  setBoundedMap(ctx.messageSpans, msgKey, msgSpan);
  setBoundedMap(ctx.llmSpanStartTimes, msgKey, startTime);
  const requestKey = `${sessionID}:${parentID}`;
  setBoundedMap(ctx.llmRequestContexts, requestKey, [
    ...(ctx.llmRequestContexts.get(requestKey) ?? []),
    {
      messageID,
      agent: messageAgent ?? agentName,
      modelID,
      providerID,
      spanContext: msgSpan.spanContext(),
    },
  ]);
  setBoundedMap(ctx.activeMessageSpans, sessionID, {
    messageID,
    span: msgSpan,
  });
}
