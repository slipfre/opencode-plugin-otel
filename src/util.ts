import { trace } from "@opentelemetry/api"
import { MAX_PENDING } from "./types.ts"
import type { HandlerContext } from "./types.ts"

const GEN_AI_PROVIDER_NAMES: Readonly<Record<string, string>> = {
  "amazon-bedrock": "aws.bedrock",
  azure: "azure.ai.openai",
  "azure-cognitive-services": "azure.ai.openai",
  google: "gcp.gemini",
  "google-vertex": "gcp.vertex_ai",
  "google-vertex-anthropic": "gcp.vertex_ai",
  mistral: "mistral_ai",
  xai: "x_ai",
}

/** Returns a human-readable summary string from an opencode error object. */
export function errorSummary(err: { name: string; data?: unknown } | undefined): string {
  if (!err) return "unknown"
  if (err.data && typeof err.data === "object" && "message" in err.data) {
    return `${err.name}: ${(err.data as { message: string }).message}`
  }
  return err.name
}

/** Returns the canonical OTel GenAI provider name, preserving unknown provider IDs. */
export function genAiProviderName(providerID: string): string {
  return GEN_AI_PROVIDER_NAMES[providerID] ?? providerID
}

/**
 * Inserts a key/value pair into `map`, evicting the oldest entry first when the map
 * has reached `MAX_PENDING` capacity to prevent unbounded memory growth.
 */
export function setBoundedMap<K, V>(map: Map<K, V>, key: K, value: V) {
  if (!map.has(key) && map.size >= MAX_PENDING) {
    const [firstKey] = map.keys()
    if (firstKey !== undefined) map.delete(firstKey)
  }
  map.set(key, value)
}

/** Resolves an interaction context from the live span first, then from the retained ended span context. */
export function resolveInteractionTraceContext(
  interactionID: string,
  ctx: Pick<HandlerContext, "rootContext" | "interactionSpans" | "interactionSpanContexts">,
) {
  const baseCtx = ctx.rootContext()
  const interactionSpan = ctx.interactionSpans.get(interactionID)
  if (interactionSpan) return trace.setSpan(baseCtx, interactionSpan)
  const interactionSpanContext = ctx.interactionSpanContexts.get(interactionID)
  return interactionSpanContext ? trace.setSpanContext(baseCtx, interactionSpanContext) : baseCtx
}

/** Resolves the best available trace parent for a session event or message/tool child span. */
export function resolveSessionTraceContext(
  sessionID: string,
  ctx: HandlerContext,
  input?: { assistantMessageID?: string; interactionID?: string },
) {
  const baseCtx = ctx.rootContext()
  if (input?.interactionID) return resolveInteractionTraceContext(input.interactionID, ctx)
  const assistantInteractionID = input?.assistantMessageID
    ? ctx.assistantInteractions.get(input.assistantMessageID)
    : undefined
  if (assistantInteractionID) return resolveInteractionTraceContext(assistantInteractionID, ctx)
  const activeInteractionID = ctx.activeInteractions.get(sessionID)
  if (activeInteractionID) return resolveInteractionTraceContext(activeInteractionID, ctx)
  const activeRun = ctx.activeRunSpans.get(sessionID)
  return activeRun ? trace.setSpan(baseCtx, activeRun.span) : baseCtx
}

/**
 * Returns `true` if the trace type is not in the disabled set.
 * Valid names are `"session"`, `"llm"`, and `"tool"`.
 */
export function isTraceEnabled(name: string, ctx: { disabledTraces: Set<string> }): boolean {
  return !ctx.disabledTraces.has(name)
}

export function accumulateInteractionTotals(
  interactionID: string,
  tokens: number,
  cost: number,
  ctx: HandlerContext,
) {
  const existing = ctx.interactionTotals.get(interactionID)
  if (!existing) return
  setBoundedMap(ctx.interactionTotals, interactionID, {
    tokens: existing.tokens + tokens,
    cost: existing.cost + cost,
    messages: existing.messages + 1,
  })
}
