import type { Context, Counter, Gauge, Histogram, Span, SpanContext, Tracer } from "@opentelemetry/api"
import type { LogRecord } from "@opentelemetry/api-logs"

/** Numeric priority map for log levels; higher value = higher severity. */
export const LEVELS = { debug: 0, info: 1, warn: 2, error: 3 } as const

/** Union of supported log level names. */
export type Level = keyof typeof LEVELS

/** Maximum number of entries kept in bounded correlation maps and queues. */
export const MAX_PENDING = 500

/** Temporary correlation header removed before the AI SDK calls the model provider. */
export const LLM_TELEMETRY_REQUEST_HEADER = "x-opencode-plugin-otel-request-id"

/** Structured logger forwarded to the opencode `client.app.log` API. */
export type PluginLogger = (
  level: Level,
  message: string,
  extra?: Record<string, unknown>,
) => Promise<void>

/** OTel attributes common to every emitted span, log, and metric. */
export type CommonAttrs = Readonly<Record<string, string>>

/** In-flight tool execution tracked between `running` and `completed`/`error` part updates. */
export type PendingToolSpan = {
  tool: string
  sessionID: string
  startMs: number
  span?: Span
}

/** Permission prompt tracked between `permission.updated` and `permission.replied`. */
export type PendingPermission = {
  type: string
  title: string
  sessionID: string
}

/** OTel metric instruments created once at plugin startup and shared via `HandlerContext`. */
export type Instruments = {
  sessionCounter: Counter
  tokenCounter: Counter
  costCounter: Counter
  linesCounter: Counter
  linesTotalGauge: Gauge
  commitCounter: Counter
  toolDurationHistogram: Histogram
  cacheCounter: Counter
  sessionDurationHistogram: Histogram
  messageCounter: Counter
  sessionTokenGauge: Histogram
  sessionCostGauge: Histogram
  modelUsageCounter: Counter
  retryCounter: Counter
  subtaskCounter: Counter
}

/** Session role emitted by opencode: either the primary/root agent or a spawned subagent. */
export type SessionAgentType = "primary" | "subagent"

export type RunDetails = {
  agentType: SessionAgentType
  parentSessionID?: string
  taskCallID?: string
  taskSpanContext?: SpanContext
}

/** Accumulated per-session totals used for gauge snapshots on session.idle. */
export type SessionTotals = {
  startMs: number
  tokens: number
  cost: number
  messages: number
  agent: string
  agentType: SessionAgentType
}

type InteractionTotals = {
  tokens: number
  cost: number
  messages: number
}

export type ActiveRunSpan = {
  span: Span
  interactionIDs: Set<string>
  interactionIO: Map<string, { input: string; output?: string }>
}

/** Live LLM request span metadata used by the outbound header hook. */
export type LlmRequestContext = {
  messageID: string
  agent: string
  modelID: string
  providerID: string
  spanContext: SpanContext
}

/** Exact LLM span selected for one AI SDK generation lifecycle. */
export type LlmTelemetryTarget = {
  msgKey: string
  span: Span
}

/** Request-identity bindings that route one AI SDK lifecycle to its exact LLM span. */
export type LlmTelemetryBindings = {
  pendingByRequestID: Map<string, LlmTelemetryTarget>
  byLifecycleMetadata: WeakMap<object, LlmTelemetryTarget>
}

/** Shared context threaded through every event handler. */
export type HandlerContext = {
  log: PluginLogger
  emitLog: (record: LogRecord) => void
  instruments: Instruments
  commonAttrs: CommonAttrs
  pendingToolSpans: Map<string, PendingToolSpan>
  pendingPermissions: Map<string, PendingPermission>
  sessionTotals: Map<string, SessionTotals>
  sessionDiffTotals: Map<string, { additions: number; deletions: number }>
  disabledMetrics: Set<string>
  disabledTraces: Set<string>
  tracer: Tracer
  tracePrefix: string
  rootContext: () => Context
  activeRunSpans: Map<string, ActiveRunSpan>
  interactionSpans: Map<string, Span>
  interactionSpanContexts: Map<string, SpanContext>
  activeInteractions: Map<string, string>
  assistantInteractions: Map<string, string>
  pendingAssistantInteractions: Map<string, { sessionID: string; interactionID: string }>
  pendingSubagentRuns: Map<string, RunDetails>
  interactionInputs: Map<string, string>
  interactionTotals: Map<string, InteractionTotals>
  sessionParents: Map<string, string>
  messageSpans: Map<string, Span>
  messageOutputs: Map<string, string>
  llmRequestContexts: Map<string, LlmRequestContext[]>
  llmTelemetryBindings: LlmTelemetryBindings
  tracePropagationProviders: Set<string>
  activeMessageSpans: Map<string, { messageID: string; span: Span; outputEndTime?: number }>
  llmTelemetryOutputs: Map<string, true>
}
