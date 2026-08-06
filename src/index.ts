import type { Plugin } from "@opencode-ai/plugin"
import { ROOT_CONTEXT } from "@opentelemetry/api"
import pkg from "../package.json" with { type: "json" }
import type {
  EventSessionCreated,
  EventSessionIdle,
  EventSessionError,
  EventMessageUpdated,
  EventMessagePartUpdated,
} from "@opencode-ai/sdk"
import { LEVELS, type Level, type HandlerContext } from "./types.ts"
import { loadConfig, parseAttributePairs, resolveHelperPath, resolveLogLevel, type OtelPluginOptions } from "./config.ts"
import { probeEndpoint } from "./probe.ts"
import { setupOtel } from "./otel.ts"
import { remoteParentContext } from "./trace-context.ts"
import {
  handleSessionCreated,
  handleSessionIdle,
  handleSessionError,
  handleInteractionStarted,
} from "./handlers/session.ts"
import { handleMessageUpdated, handleMessagePartUpdated, startMessageSpan } from "./handlers/message.ts"
import { handleChatHeaders } from "./handlers/chat-headers.ts"
import { registerAiTelemetry } from "./ai-telemetry.ts"
import { createUserIDManager } from "./user-id.ts"

const PLUGIN_VERSION: string = (pkg as { version?: string }).version ?? "unknown"

/**
 * OpenCode plugin that exports traces via OpenTelemetry (OTLP over gRPC or HTTP).
 * All instrumentation is gated on `OPENCODE_ENABLE_TELEMETRY`.
 */
export const OtelPlugin: Plugin = async ({ project, client, directory, worktree }, options) => {
  const config = loadConfig(options as OtelPluginOptions)
  const otlpHeadersHelper = resolveHelperPath(config.otlpHeadersHelper, directory, worktree)
  let minLevel: Level = "info"

  const log: HandlerContext["log"] = async (level, message, extra) => {
    if (LEVELS[level] < LEVELS[minLevel]) return
    await client.app.log({ body: { service: "opencode-plugin-otel", level, message, extra } })
  }

  if (!config.enabled) {
    await log("info", "telemetry disabled (set OPENCODE_ENABLE_TELEMETRY to enable)")
    return {}
  }

  await log("info", "starting up", {
    version: PLUGIN_VERSION,
    endpoint: config.endpoint,
    protocol: config.protocol,
    spanAttributeCountLimit: config.spanAttributeCountLimit,
    tracePrefix: config.tracePrefix,
    headersHelperSet: !!config.otlpHeadersHelper,
    userIDEnabled: config.userIDEnabled,
    userIDTimeout: config.userIDTimeout,
    userIDRetryCount: config.userIDRetryCount,
    userIDCooldown: config.userIDCooldown,
  })

  await log("debug", "config loaded", {
    headersSet: !!config.otlpHeaders,
    headersHelperSet: !!config.otlpHeadersHelper,
    resourceAttributesSet: !!config.resourceAttributes,
    spanAttributesSet: !!config.spanAttributes,
  })

  const probe = await probeEndpoint(config.endpoint)
  if (probe.ok) {
    await log("info", "OTLP endpoint reachable", { endpoint: config.endpoint, ms: probe.ms })
  } else {
    await log("warn", "OTLP endpoint unreachable — exports may fail", {
      endpoint: config.endpoint,
      error: probe.error,
    })
  }

  const providers = await setupOtel(
    config.endpoint,
    config.protocol,
    PLUGIN_VERSION,
    config.otlpHeaders,
    otlpHeadersHelper,
    config.spanAttributeCountLimit,
  )
  const { tracerProvider } = providers
  await log("info", "OTel SDK initialized")

  const tracer = tracerProvider.getTracer("com.opencode")
  const remoteContext = remoteParentContext(config.traceparent, config.tracestate)
  if (config.traceparent && !remoteContext) {
    await log("warn", "invalid OPENCODE_TRACEPARENT ignored", { traceparentLength: config.traceparent.length })
  }
  const rootContext = remoteContext ? () => remoteContext : () => ROOT_CONTEXT
  const pendingToolSpans = new Map()
  const activeRunSpans = new Map()
  const interactionSpans = new Map()
  const interactionSpanContexts = new Map()
  const activeInteractions = new Map()
  const assistantInteractions = new Map()
  const pendingAssistantInteractions = new Map()
  const pendingSubagentRuns = new Map()
  const interactionInputs = new Map()
  const interactionTotals = new Map()
  const interactionCompletions = new Map()
  const sessionParents = new Map()
  const messageSpans = new Map()
  const messageOutputs = new Map()
  const llmRequestContexts = new Map()
  const llmTelemetryBindings: HandlerContext["llmTelemetryBindings"] = {
    pendingByRequestID: new Map(),
    byLifecycleMetadata: new WeakMap(),
  }
  const activeMessageSpans = new Map()
  const llmTelemetryOutputs = new Map()
  const commonAttrs = {
    ...parseAttributePairs(config.spanAttributes),
    "project.id": project.id,
  } as const

  const ctx: HandlerContext = {
    log,
    commonAttrs,
    pendingToolSpans,
    tracer,
    tracePrefix: config.tracePrefix,
    rootContext,
    activeRunSpans,
    interactionSpans,
    interactionSpanContexts,
    activeInteractions,
    assistantInteractions,
    pendingAssistantInteractions,
    pendingSubagentRuns,
    interactionInputs,
    interactionTotals,
    interactionCompletions,
    sessionParents,
    messageSpans,
    messageOutputs,
    llmRequestContexts,
    llmTelemetryBindings,
    tracePropagationProviders: config.tracePropagationProviders,
    activeMessageSpans,
    llmTelemetryOutputs,
  }
  const userIDManager = createUserIDManager(config, ctx, commonAttrs)

  const unregisterAiTelemetry = registerAiTelemetry(ctx)

  let shuttingDown = false

  async function flushTelemetry(reason: string) {
    if (shuttingDown) return
    await tracerProvider.forceFlush()
    await log("debug", "otel: traces flushed", { reason })
  }

  async function shutdown() {
    if (shuttingDown) return
    shuttingDown = true
    await tracerProvider.forceFlush()
    await tracerProvider.shutdown()
  }

  const handleSigterm = () => { shutdown().then(() => process.exit(0)).catch(() => process.exit(1)) }
  const handleSigint = () => { shutdown().then(() => process.exit(0)).catch(() => process.exit(1)) }
  const handleBeforeExit = () => { shutdown().catch(() => {}) }

  process.on("SIGTERM", handleSigterm)
  process.on("SIGINT", handleSigint)
  process.on("beforeExit", handleBeforeExit)

  const safe = <T extends unknown[]>(
    name: string,
    fn: (...args: T) => Promise<void> | void,
  ): ((...args: T) => Promise<void>) =>
    async (...args: T) => {
      try {
        await fn(...args)
      } catch (err) {
        await log("error", `otel: unhandled error in ${name}`, {
          error: err instanceof Error ? err.message : String(err),
          stack: err instanceof Error ? err.stack : undefined,
        })
      }
    }

  return {
    dispose: async () => {
      unregisterAiTelemetry()
      process.off("SIGTERM", handleSigterm)
      process.off("SIGINT", handleSigint)
      process.off("beforeExit", handleBeforeExit)
      await shutdown()
    },

    config: async (cfg) => {
      await userIDManager.configure(cfg.provider)

      if (cfg.logLevel) {
        const next = resolveLogLevel(cfg.logLevel, minLevel)
        if (next !== minLevel) {
          minLevel = next
          await log("info", `log level set to "${minLevel}"`)
        } else if (cfg.logLevel.toLowerCase() !== minLevel) {
          await log("warn", `unknown log level "${cfg.logLevel}", keeping "${minLevel}"`)
        }
      }
    },

    "chat.headers": safe("chat.headers", async (input, output) => {
      userIDManager.refreshInBackground()
      handleChatHeaders(input, output, ctx)
    }),

    "chat.message": safe("chat.message", async (input, output) => {
      userIDManager.refreshInBackground()
      const agent = input.agent ?? "unknown"
      const startTime = Date.now()
      const promptText = output.parts.map((part) => {
        switch (part.type) {
          case "text":
            return part.text
          case "file":
            return part.filename ?? part.url
          case "agent":
            return part.name
          case "subtask":
            return part.description
          default:
            return ""
        }
      }).filter(Boolean).join("\n")
      const model = input.model ? `${input.model.providerID}/${input.model.modelID}` : "unknown"
      handleInteractionStarted(
        output.message.id,
        input.sessionID,
        agent,
        promptText,
        model,
        startTime,
        ctx,
      )
    }),

    event: safe("event", async ({ event }) => {
      userIDManager.refreshInBackground()
      switch (event.type) {
        case "session.created":
          await handleSessionCreated(event as EventSessionCreated, ctx)
          break
        case "session.idle":
          handleSessionIdle(event as EventSessionIdle, ctx)
          await flushTelemetry("session.idle")
          break
        case "session.error":
          handleSessionError(event as EventSessionError, ctx)
          await flushTelemetry("session.error")
          break
        case "message.updated": {
          const msgEvt = event as EventMessageUpdated
          const info = msgEvt.properties.info
          if (info.role === "user") {
            break
          }
          if (info.role === "assistant" && !info.time?.completed) {
            startMessageSpan(
              info.sessionID,
              info.id,
              info.parentID,
              info.modelID ?? "unknown",
              info.providerID ?? "unknown",
              info.time?.created ?? Date.now(),
              ctx,
              info.mode,
            )
          }
          await handleMessageUpdated(msgEvt, ctx)
          if (info.role === "assistant" && info.time?.completed) {
            await flushTelemetry("message.completed")
          }
          break
        }
        case "message.part.updated":
          await handleMessagePartUpdated(event as EventMessagePartUpdated, ctx)
          break
      }
    }),
  }
}
