import { trace } from "@opentelemetry/api";
import { MAX_PENDING } from "./types.ts";
import type { HandlerContext } from "./types.ts";

const GEN_AI_PROVIDER_NAMES: Readonly<Record<string, string>> = {
  "amazon-bedrock": "aws.bedrock",
  azure: "azure.ai.openai",
  "azure-cognitive-services": "azure.ai.openai",
  google: "gcp.gemini",
  "google-vertex": "gcp.vertex_ai",
  "google-vertex-anthropic": "gcp.vertex_ai",
  mistral: "mistral_ai",
  xai: "x_ai",
};

/** Returns a human-readable summary string from an opencode error object. */
export function errorSummary(
  err: { name: string; data?: unknown } | undefined
): string {
  if (!err) {
    return "unknown";
  }
  if (err.data && typeof err.data === "object" && "message" in err.data) {
    return `${err.name}: ${(err.data as { message: string }).message}`;
  }
  return err.name;
}

/** Returns the canonical OTel GenAI provider name, preserving unknown provider IDs. */
export function genAiProviderName(providerID: string): string {
  return GEN_AI_PROVIDER_NAMES[providerID] ?? providerID;
}

/**
 * Inserts a key/value pair into `map`, evicting the oldest entry first when the map
 * has reached `MAX_PENDING` capacity to prevent unbounded memory growth.
 */
export function setBoundedMap<K, V>(map: Map<K, V>, key: K, value: V) {
  if (!map.has(key) && map.size >= MAX_PENDING) {
    const [firstKey] = map.keys();
    if (firstKey !== undefined) {
      map.delete(firstKey);
    }
  }
  map.set(key, value);
}

/** Resolves an interaction context from the live span first, then from the retained ended span context. */
export function resolveInteractionTraceContext(
  interactionID: string,
  ctx: Pick<
    HandlerContext,
    "rootContext" | "interactionSpans" | "interactionSpanContexts"
  >
) {
  return (
    tryResolveInteractionTraceContext(interactionID, ctx) ?? ctx.rootContext()
  );
}

function tryResolveInteractionTraceContext(
  interactionID: string,
  ctx: Pick<
    HandlerContext,
    "rootContext" | "interactionSpans" | "interactionSpanContexts"
  >
) {
  const baseCtx = ctx.rootContext();
  const interactionSpan = ctx.interactionSpans.get(interactionID);
  if (interactionSpan) {
    return trace.setSpan(baseCtx, interactionSpan);
  }
  const interactionSpanContext = ctx.interactionSpanContexts.get(interactionID);
  return interactionSpanContext
    ? trace.setSpanContext(baseCtx, interactionSpanContext)
    : undefined;
}

/** Resolves the best available trace parent for the current session execution. */
export function resolveSessionTraceContext(
  sessionID: string,
  ctx: HandlerContext
) {
  const baseCtx = ctx.rootContext();
  const activeInteractionID = ctx.activeInteractions.get(sessionID);
  if (activeInteractionID) {
    const interactionContext = tryResolveInteractionTraceContext(
      activeInteractionID,
      ctx
    );
    if (interactionContext) {
      return interactionContext;
    }
  }
  const activeRun = ctx.activeRunSpans.get(sessionID);
  return activeRun ? trace.setSpan(baseCtx, activeRun.span) : baseCtx;
}

export { tryResolveInteractionTraceContext };
