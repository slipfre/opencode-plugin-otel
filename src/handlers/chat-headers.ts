import type { ProviderContext } from "@opencode-ai/plugin"
import type { Model, UserMessage } from "@opencode-ai/sdk"
import { LLM_TELEMETRY_REQUEST_HEADER, type HandlerContext } from "../types.ts"
import { injectTraceContext } from "../trace-context.ts"
import { setBoundedMap } from "../util.ts"

/** Injects the matching LLM span context for explicitly enabled providers. */
export function handleChatHeaders(
  input: { sessionID: string; agent: string; model: Model; provider: ProviderContext; message: UserMessage },
  output: { headers: Record<string, string> },
  ctx: HandlerContext,
): void {
  const providerID = input.model.providerID
  const request = ctx.llmRequestContexts.get(`${input.sessionID}:${input.message.id}`)?.findLast(candidate =>
    candidate.agent === input.agent
    && candidate.modelID === input.model.id
    && candidate.providerID === providerID
  )
  if (!request) return

  const msgKey = `${input.sessionID}:${request.messageID}`
  const span = ctx.messageSpans.get(msgKey)
  if (span) {
    // OpenCode clones the chat.headers result while merging provider headers, so object
    // identity cannot correlate this hook with AI SDK callbacks. The ID survives that
    // clone and is removed by onStart before the provider request is sent.
    const requestID = crypto.randomUUID()
    output.headers[LLM_TELEMETRY_REQUEST_HEADER] = requestID
    setBoundedMap(ctx.llmTelemetryBindings.pendingByRequestID, requestID, { msgKey, span })
  }

  if (!ctx.tracePropagationProviders.has(providerID) && !ctx.tracePropagationProviders.has("*")) return
  injectTraceContext(request.spanContext, output.headers)
}
