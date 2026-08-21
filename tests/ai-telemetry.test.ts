import { describe, expect, test } from "bun:test";
import type {
  OnStartEvent,
  OnStepFinishEvent,
  OnStepStartEvent,
  TelemetryIntegration,
} from "ai";
import {
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  LLM_INVOCATION_PARAMETERS,
  LLM_OUTPUT_MESSAGES,
  LLM_TOOLS,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENTS,
  MESSAGE_ROLE,
  MimeType,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
  TOOL_CALL_FUNCTION_NAME,
  TOOL_CALL_ID,
  TOOL_JSON_SCHEMA,
} from "@arizeai/openinference-semantic-conventions";
import type {
  EventMessagePartUpdated,
  EventMessageUpdated,
  EventSessionStatus,
} from "@opencode-ai/sdk";
import { registerAiTelemetry } from "../src/ai-telemetry.ts";
import { handleChatHeaders } from "../src/handlers/chat-headers.ts";
import {
  handleMessagePartUpdated,
  handleMessageUpdated,
  startMessageSpan,
} from "../src/handlers/message.ts";
import { handleSessionStatus } from "../src/handlers/session.ts";
import { LLM_TELEMETRY_REQUEST_HEADER } from "../src/types.ts";
import { makeCtx } from "./helpers.ts";

type TelemetryGlobal = typeof globalThis & {
  AI_SDK_TELEMETRY_INTEGRATIONS?: TelemetryIntegration[];
};

type ChatHeadersInput = Parameters<typeof handleChatHeaders>[0];

function integration(): TelemetryIntegration {
  const integrations =
    (globalThis as TelemetryGlobal).AI_SDK_TELEMETRY_INTEGRATIONS ?? [];
  const value = integrations.at(-1);
  if (!value) {
    throw new Error("AI SDK telemetry integration not registered");
  }
  return value;
}

type TelemetryLifecycle = {
  sessionID: string;
  headers: Record<string, string>;
  metadata: Record<string, unknown>;
};

function bindTelemetryLifecycle(
  ctx: ReturnType<typeof makeCtx>["ctx"],
  overrides: {
    sessionID?: string;
    parentID?: string;
    agent?: string;
    modelID?: string;
    providerID?: string;
  } = {}
): TelemetryLifecycle {
  const sessionID = overrides.sessionID ?? "ses_1";
  const parentID = overrides.parentID ?? "user_1";
  const request = ctx.llmRequestContexts
    .get(`${sessionID}:${parentID}`)
    ?.at(-1);
  if (!request) {
    throw new Error("LLM request context not found");
  }
  const modelID = overrides.modelID ?? request.modelID;
  const providerID = overrides.providerID ?? request.providerID;
  const hookHeaders = { "X-Test": "value", Authorization: "Bearer test-token" };
  handleChatHeaders(
    {
      sessionID,
      agent: overrides.agent ?? request.agent,
      model: { id: modelID, providerID } as ChatHeadersInput["model"],
      provider: {
        source: "config",
        info: { id: providerID },
        options: {},
      } as ChatHeadersInput["provider"],
      message: { id: parentID } as ChatHeadersInput["message"],
    },
    { headers: hookHeaders },
    ctx
  );
  // OpenCode creates a new prepared headers object after all chat.headers hooks run.
  // Keeping this clone in the fixture prevents object-identity correlation from passing falsely.
  const headers = { "x-session-affinity": sessionID, ...hookHeaders };
  return { sessionID, headers, metadata: { sessionId: sessionID } };
}

function startEvent(lifecycle: TelemetryLifecycle): OnStartEvent {
  return {
    model: { provider: "anthropic", modelId: "claude" },
    system: "You are concise.",
    prompt: undefined,
    messages: [{ role: "user", content: "hello" }],
    tools: {
      bash: {
        description: "Run a command",
        inputSchema: {
          type: "object",
          properties: { command: { type: "string" } },
        },
        execute: () => undefined,
      },
    },
    toolChoice: "auto",
    activeTools: ["bash"],
    maxOutputTokens: 4096,
    temperature: 0.2,
    topP: 0.9,
    topK: 40,
    presencePenalty: undefined,
    frequencyPenalty: undefined,
    stopSequences: undefined,
    seed: undefined,
    maxRetries: 0,
    timeout: undefined,
    headers: lifecycle.headers,
    providerOptions: {
      anthropic: { thinking: { type: "enabled", budgetTokens: 1024 } },
    },
    stopWhen: undefined,
    output: undefined,
    abortSignal: undefined,
    include: undefined,
    functionId: "session.llm",
    metadata: lifecycle.metadata,
    experimental_context: undefined,
  } as unknown as OnStartEvent;
}

function stepStartEvent(
  lifecycle: TelemetryLifecycle,
  stepNumber = 0
): OnStepStartEvent {
  return {
    stepNumber,
    model: { provider: "anthropic", modelId: "claude" },
    system: "You are concise.",
    messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
    tools: {
      bash: {
        description: "Run a command",
        inputSchema: {
          type: "object",
          properties: { command: { type: "string" } },
        },
        execute: () => undefined,
      },
    },
    toolChoice: { type: "auto" },
    activeTools: ["bash"],
    steps: [],
    providerOptions: {
      anthropic: { thinking: { type: "enabled", budgetTokens: 1024 } },
    },
    timeout: undefined,
    headers: lifecycle.headers,
    stopWhen: undefined,
    output: undefined,
    abortSignal: undefined,
    include: undefined,
    functionId: "session.llm",
    metadata: lifecycle.metadata,
    experimental_context: undefined,
  } as unknown as OnStepStartEvent;
}

function stepFinishEvent(
  lifecycle: TelemetryLifecycle,
  stepNumber = 0
): OnStepFinishEvent {
  return {
    stepNumber,
    model: { provider: "anthropic", modelId: "claude" },
    functionId: "session.llm",
    metadata: lifecycle.metadata,
    experimental_context: undefined,
    content: [
      { type: "reasoning", text: "thinking" },
      { type: "text", text: "hello back" },
      {
        type: "tool-call",
        toolCallId: "call_1",
        toolName: "bash",
        input: { command: "pwd" },
      },
    ],
    text: "hello back",
    reasoning: [{ type: "reasoning", text: "thinking" }],
    reasoningText: "thinking",
    files: [],
    sources: [],
    toolCalls: [
      {
        type: "tool-call",
        toolCallId: "call_1",
        toolName: "bash",
        input: { command: "pwd" },
      },
    ],
    staticToolCalls: [],
    dynamicToolCalls: [],
    toolResults: [],
    staticToolResults: [],
    dynamicToolResults: [],
    finishReason: "tool-calls",
    rawFinishReason: "tool_use",
    usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
    warnings: [],
    request: {
      body: { model: "claude", messages: [{ role: "user", content: "hello" }] },
    },
    response: {
      id: "resp_1",
      modelId: "claude",
      timestamp: new Date("2026-01-01T00:00:00.000Z"),
      headers: { "X-Request-ID": "req_123", "Set-Cookie": "session=test" },
      messages: [
        { role: "assistant", content: [{ type: "text", text: "hello back" }] },
      ],
      body: {
        type: "message",
        content: [{ type: "text", text: "hello back" }],
      },
    },
    providerMetadata: { anthropic: { cacheCreationInputTokens: 2 } },
  } as unknown as OnStepFinishEvent;
}

function assistantCompleted(): EventMessageUpdated {
  return {
    type: "message.updated",
    properties: {
      info: {
        id: "msg_1",
        parentID: "user_1",
        role: "assistant",
        sessionID: "ses_1",
        modelID: "claude",
        providerID: "anthropic",
        cost: 0.01,
        tokens: {
          input: 10,
          output: 5,
          reasoning: 1,
          cache: { read: 0, write: 0 },
        },
        time: { created: 1000, completed: 2000 },
      },
    },
  } as unknown as EventMessageUpdated;
}

function sessionRetry(attempt: number, message: string): EventSessionStatus {
  return {
    type: "session.status",
    properties: {
      sessionID: "ses_1",
      status: { type: "retry", attempt, message, next: 0 },
    },
  };
}

function stepStarted(time: number): EventMessagePartUpdated {
  return {
    type: "message.part.updated",
    properties: {
      time,
      part: {
        id: "part_step",
        type: "step-start",
        sessionID: "ses_1",
        messageID: "msg_1",
      },
    },
  } as unknown as EventMessagePartUpdated;
}

function assistantCompletedAfterRetry(): EventMessageUpdated {
  return {
    type: "message.updated",
    properties: {
      info: {
        id: "msg_1",
        parentID: "user_1",
        role: "assistant",
        sessionID: "ses_1",
        modelID: "claude",
        providerID: "anthropic",
        cost: 0.01,
        tokens: {
          input: 10,
          output: 51,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        time: { created: 1000, completed: 4300 },
      },
    },
  } as unknown as EventMessageUpdated;
}

describe("AI SDK telemetry integration", () => {
  test("records actual retry starts and final-attempt token timing", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      ctx
    );
    const lifecycle = bindTelemetryLifecycle(ctx);
    const unregister = registerAiTelemetry(ctx);
    const originalNow = Date.now;

    try {
      Date.now = () => 1100;
      await integration().onStart?.(startEvent(lifecycle));

      handleSessionStatus(sessionRetry(1, "rate limited"), ctx);
      Date.now = () => 2000;
      await integration().onStart?.(startEvent(lifecycle));
      handleMessagePartUpdated(stepStarted(2200), ctx);
      const active = ctx.activeMessageSpans.get("ses_1")!;
      ctx.activeMessageSpans.set("ses_1", {
        ...active,
        outputEndTime: 2300,
      });

      handleSessionStatus(sessionRetry(2, "timeout"), ctx);
      Date.now = () => 3000;
      await integration().onStart?.(startEvent(lifecycle));
      handleMessagePartUpdated(stepStarted(3300), ctx);
      handleMessageUpdated(assistantCompletedAfterRetry(), ctx);

      const span = tracer.spans[0]!;
      expect(span.attributes["opencode.llm.retry_count"]).toBe(2);
      expect(
        JSON.parse(String(span.attributes["opencode.llm.retry_history"]))
      ).toEqual([
        { attempt: 1, reason: "rate limited", start_offset_ms: 1000 },
        { attempt: 2, reason: "timeout", start_offset_ms: 2000 },
      ]);
      expect(span.events).toEqual([
        {
          name: "opencode.llm.retry.started",
          attributes: {
            "opencode.llm.retry.attempt": 1,
            "opencode.llm.retry.reason": "rate limited",
            "opencode.llm.retry.start_offset_ms": 1000,
          },
          startTime: 2000,
        },
        {
          name: "opencode.llm.retry.started",
          attributes: {
            "opencode.llm.retry.attempt": 2,
            "opencode.llm.retry.reason": "timeout",
            "opencode.llm.retry.start_offset_ms": 2000,
          },
          startTime: 3000,
        },
      ]);
      expect(span.attributes["opencode.llm.time_to_first_chunk_ms"]).toBe(300);
      expect(
        span.attributes["opencode.llm.estimated_time_per_output_token_ms"]
      ).toBe(20);
    } finally {
      Date.now = originalNow;
      unregister();
    }
  });

  test("adds OpenInference input and output to the active llm span", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      ctx
    );
    const lifecycle = bindTelemetryLifecycle(ctx);
    const unregister = registerAiTelemetry(ctx);

    try {
      await integration().onStart?.(startEvent(lifecycle));
      await integration().onStepStart?.(stepStartEvent(lifecycle));
      await integration().onStepFinish?.(stepFinishEvent(lifecycle));

      const span = tracer.spans[0]!;
      const input = JSON.parse(String(span.attributes[INPUT_VALUE]));
      const output = JSON.parse(String(span.attributes[OUTPUT_VALUE]));
      const parameters = JSON.parse(
        String(span.attributes[LLM_INVOCATION_PARAMETERS])
      );
      const tool = JSON.parse(
        String(span.attributes[`${LLM_TOOLS}.0.${TOOL_JSON_SCHEMA}`])
      );

      expect(span.attributes[INPUT_MIME_TYPE]).toBe(MimeType.JSON);
      expect(input).toEqual([
        { role: "system", content: "You are concise." },
        { role: "user", content: [{ type: "text", text: "hello" }] },
      ]);
      expect(parameters.maxOutputTokens).toBe(4096);
      expect(parameters.providerOptions.anthropic.thinking.budgetTokens).toBe(
        1024
      );
      expect(parameters.headers).toBeUndefined();
      expect(parameters.maxRetries).toBeUndefined();
      expect(tool).toEqual({
        type: "function",
        function: {
          name: "bash",
          description: "Run a command",
          parameters: {
            type: "object",
            properties: { command: { type: "string" } },
          },
        },
      });
      expect(span.attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]).toBe(
        "system"
      );
      expect(span.attributes[`${LLM_INPUT_MESSAGES}.1.${MESSAGE_ROLE}`]).toBe(
        "user"
      );
      expect(
        span.attributes[
          `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`
        ]
      ).toBe("text");
      expect(
        span.attributes[
          `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`
        ]
      ).toBe("hello");
      expect(
        JSON.parse(String(span.attributes["http.request.headers"]))
      ).toEqual({
        "x-session-affinity": "ses_1",
        "x-test": "value",
        authorization: "Bearer test-token",
      });
      expect(lifecycle.headers[LLM_TELEMETRY_REQUEST_HEADER]).toBeUndefined();
      expect(span.attributes["http.request.header.x-test"]).toBeUndefined();

      expect(span.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
      expect(output).toEqual([
        {
          role: "assistant",
          content: [
            { type: "reasoning", text: "thinking" },
            { type: "text", text: "hello back" },
            {
              type: "tool_use",
              id: "call_1",
              name: "bash",
              arguments: { command: "pwd" },
            },
          ],
        },
      ]);
      expect(span.attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`]).toBe(
        "assistant"
      );
      expect(
        span.attributes[
          `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`
        ]
      ).toBe("reasoning");
      expect(
        span.attributes[
          `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`
        ]
      ).toBe("hello back");
      expect(
        span.attributes[
          `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.2.${TOOL_CALL_ID}`
        ]
      ).toBe("call_1");
      expect(
        span.attributes[
          `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.2.${TOOL_CALL_FUNCTION_NAME}`
        ]
      ).toBe("bash");
      expect(
        span.attributes[
          `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.2.${TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
        ]
      ).toBe('{"command":"pwd"}');
      expect(
        JSON.parse(String(span.attributes["http.response.headers"]))
      ).toEqual({
        "x-request-id": "req_123",
        "set-cookie": "session=test",
      });
      expect(
        span.attributes["http.response.header.x-request-id"]
      ).toBeUndefined();
      expect(JSON.stringify(output)).not.toContain("providerMetadata");
      expect(JSON.stringify(output)).not.toContain("request");
      expect(JSON.stringify(output)).not.toContain("usage");
    } finally {
      unregister();
    }
  });

  test("does not let a concurrent title generation overwrite the main llm payload", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      ctx
    );
    const main = bindTelemetryLifecycle(ctx);
    const title = bindTelemetryLifecycle(ctx, { agent: "title" });
    const unregister = registerAiTelemetry(ctx);

    try {
      await integration().onStart?.(startEvent(main));
      await integration().onStart?.({
        ...startEvent(title),
        maxOutputTokens: 32,
      } as OnStartEvent);

      await integration().onStepStart?.(stepStartEvent(main));
      await integration().onStepStart?.({
        ...stepStartEvent(title),
        system: "Generate a title.",
        messages: [{ role: "user", content: "secret title prompt" }],
      } as OnStepStartEvent);

      await integration().onStepFinish?.(stepFinishEvent(main));
      const titleFinish = stepFinishEvent(title);
      await integration().onStepFinish?.({
        ...titleFinish,
        content: [{ type: "text", text: "Secret title" }],
        text: "Secret title",
        response: {
          ...titleFinish.response,
          messages: [
            {
              role: "assistant",
              content: [{ type: "text", text: "Secret title" }],
            },
          ],
        },
      } as OnStepFinishEvent);

      const span = tracer.spans[0]!;
      const parameters = JSON.parse(
        String(span.attributes[LLM_INVOCATION_PARAMETERS])
      );
      const input = JSON.parse(String(span.attributes[INPUT_VALUE]));
      const output = JSON.parse(String(span.attributes[OUTPUT_VALUE]));
      expect(parameters.maxOutputTokens).toBe(4096);
      expect(input[1].content[0].text).toBe("hello");
      expect(output[0].content[1].text).toBe("hello back");
      expect(JSON.stringify(span.attributes)).not.toContain("secret title");
      expect(
        ctx.llmTelemetryBindings.byLifecycleMetadata.has(title.metadata)
      ).toBe(false);
    } finally {
      unregister();
    }
  });

  test("keeps only the current AI SDK call payload", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      ctx
    );
    const lifecycle = bindTelemetryLifecycle(ctx);
    const unregister = registerAiTelemetry(ctx);

    try {
      await integration().onStart?.(startEvent(lifecycle));
      await integration().onStepStart?.(stepStartEvent(lifecycle, 0));
      await integration().onStepFinish?.(stepFinishEvent(lifecycle, 0));
      await integration().onStepStart?.(stepStartEvent(lifecycle, 1));
      await integration().onStepFinish?.(stepFinishEvent(lifecycle, 1));

      const input = JSON.parse(
        String(tracer.spans[0]!.attributes[INPUT_VALUE])
      );
      const output = JSON.parse(
        String(tracer.spans[0]!.attributes[OUTPUT_VALUE])
      );
      expect(input).toHaveLength(2);
      expect(output).toHaveLength(1);
      expect(input.steps).toBeUndefined();
      expect(output.steps).toBeUndefined();
    } finally {
      unregister();
    }
  });

  test("restores OpenAI OAuth instructions as a system message", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan("ses_1", "msg_1", "user_1", "gpt-5", "openai", 1000, ctx);
    const lifecycle = bindTelemetryLifecycle(ctx);
    const unregister = registerAiTelemetry(ctx);

    try {
      await integration().onStart?.(startEvent(lifecycle));
      await integration().onStepStart?.({
        ...stepStartEvent(lifecycle),
        system: undefined,
        providerOptions: { openai: { instructions: "You are OpenCode." } },
      } as OnStepStartEvent);

      const span = tracer.spans[0]!;
      const input = JSON.parse(String(span.attributes[INPUT_VALUE]));
      expect(input[0]).toEqual({
        role: "system",
        content: "You are OpenCode.",
      });
      expect(span.attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]).toBe(
        "system"
      );
    } finally {
      unregister();
    }
  });

  test("preserves telemetry output when the assistant message completes", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      ctx
    );
    ctx.messageOutputs.set("ses_1:msg_1", "normalized fallback");
    const lifecycle = bindTelemetryLifecycle(ctx);
    const unregister = registerAiTelemetry(ctx);

    try {
      await integration().onStart?.(startEvent(lifecycle));
      await integration().onStepFinish?.(stepFinishEvent(lifecycle));
      handleMessageUpdated(assistantCompleted(), ctx);

      const span = tracer.spans[0]!;
      expect(
        JSON.parse(String(span.attributes[OUTPUT_VALUE]))[0].content[1].text
      ).toBe("hello back");
      expect(span.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
      expect(span.ended).toBe(true);
      expect(ctx.activeMessageSpans.has("ses_1")).toBe(false);
      expect(ctx.llmTelemetryOutputs.has("ses_1:msg_1")).toBe(false);
    } finally {
      unregister();
    }
  });

  test("ignores telemetry belonging to another workspace instance", async () => {
    const first = makeCtx();
    const second = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      first.ctx
    );
    const lifecycle = bindTelemetryLifecycle(first.ctx);
    const unregisterFirst = registerAiTelemetry(first.ctx);
    const unregisterSecond = registerAiTelemetry(second.ctx);

    try {
      await integration().onStart?.(startEvent(lifecycle));
      await integration().onStepStart?.(stepStartEvent(lifecycle));
      expect(first.tracer.spans[0]!.attributes[INPUT_VALUE]).toBeDefined();
      expect(second.tracer.spans).toHaveLength(0);
    } finally {
      unregisterFirst();
      unregisterSecond();
    }
  });

  test("ignores AI SDK operations outside the session llm path", async () => {
    const { ctx, tracer } = makeCtx();
    startMessageSpan(
      "ses_1",
      "msg_1",
      "user_1",
      "claude",
      "anthropic",
      1000,
      ctx
    );
    const lifecycle = bindTelemetryLifecycle(ctx);
    const unregister = registerAiTelemetry(ctx);

    try {
      await integration().onStepStart?.({
        ...stepStartEvent(lifecycle),
        functionId: "agent.generate",
      } as OnStepStartEvent);
      expect(tracer.spans[0]!.attributes[INPUT_VALUE]).toBeUndefined();
    } finally {
      unregister();
    }
  });
});
