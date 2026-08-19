import type { ProviderContext } from "@opencode-ai/plugin";
import type { Model, UserMessage } from "@opencode-ai/sdk";
import { createTraceState } from "@opentelemetry/api";
import { USER_ID } from "@arizeai/openinference-semantic-conventions";
import { LLM_TELEMETRY_REQUEST_HEADER, type HandlerContext } from "../types.ts";
import { injectTraceContext } from "../trace-context.ts";
import { UNKNOWN_USER_ID } from "../user-id.ts";
import { setBoundedMap } from "../util.ts";

const VALID_TRACESTATE_VALUE = /^[ -~]{0,255}[!-~]$/;

function injectUserIDTracestate(
  headers: Record<string, string>,
  ctx: HandlerContext
): void {
  const key = ctx.userIDTracestateKey;
  if (!key) {
    return;
  }
  const userID = ctx.commonAttrs[USER_ID];
  if (
    !userID ||
    userID === UNKNOWN_USER_ID ||
    !VALID_TRACESTATE_VALUE.test(userID) ||
    /[,=]/.test(userID)
  ) {
    return;
  }
  headers["tracestate"] = createTraceState(headers["tracestate"])
    .set(key, userID)
    .serialize();
}

/**
 * Injects the matching LLM span context for explicitly enabled providers, along
 * with the resolved `user.id` as an extra `tracestate` member.
 */
export function handleChatHeaders(
  input: {
    sessionID: string;
    agent: string;
    model: Model;
    provider: ProviderContext;
    message: UserMessage;
  },
  output: { headers: Record<string, string> },
  ctx: HandlerContext
): void {
  const providerID = input.model.providerID;
  const request = ctx.llmRequestContexts
    .get(`${input.sessionID}:${input.message.id}`)
    ?.findLast(
      (candidate) =>
        candidate.agent === input.agent &&
        candidate.modelID === input.model.id &&
        candidate.providerID === providerID
    );
  if (!request) {
    return;
  }

  const msgKey = `${input.sessionID}:${request.messageID}`;
  const span = ctx.messageSpans.get(msgKey);
  if (span) {
    // OpenCode clones the chat.headers result while merging provider headers, so object
    // identity cannot correlate this hook with AI SDK callbacks. The ID survives that
    // clone and is removed by onStart before the provider request is sent.
    const requestID = crypto.randomUUID();
    output.headers[LLM_TELEMETRY_REQUEST_HEADER] = requestID;
    setBoundedMap(ctx.llmTelemetryBindings.pendingByRequestID, requestID, {
      msgKey,
      span,
    });
  }

  if (
    !ctx.tracePropagationProviders.has(providerID) &&
    !ctx.tracePropagationProviders.has("*")
  ) {
    return;
  }
  injectTraceContext(request.spanContext, output.headers);
  injectUserIDTracestate(output.headers, ctx);
}
