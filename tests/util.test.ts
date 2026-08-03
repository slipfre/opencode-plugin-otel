import { describe, test, expect } from "bun:test"
import { trace } from "@opentelemetry/api"
import {
  errorSummary,
  genAiProviderName,
  setBoundedMap,
  isTraceEnabled,
  resolveInteractionTraceContext,
  resolveSessionTraceContext,
} from "../src/util.ts"
import { MAX_PENDING } from "../src/types.ts"
import { makeCtx } from "./helpers.ts"

describe("errorSummary", () => {
  test("returns 'unknown' for undefined", () => {
    expect(errorSummary(undefined)).toBe("unknown")
  })

  test("returns name when no data", () => {
    expect(errorSummary({ name: "APIError" })).toBe("APIError")
  })

  test("returns name when data has no message", () => {
    expect(errorSummary({ name: "APIError", data: { code: 500 } })).toBe("APIError")
  })

  test("returns name: message when data has message", () => {
    expect(errorSummary({ name: "APIError", data: { message: "rate limited" } })).toBe(
      "APIError: rate limited",
    )
  })

  test("returns name when data is a primitive", () => {
    expect(errorSummary({ name: "APIError", data: "oops" })).toBe("APIError")
  })
})

describe("genAiProviderName", () => {
  test.each([
    ["amazon-bedrock", "aws.bedrock"],
    ["azure", "azure.ai.openai"],
    ["azure-cognitive-services", "azure.ai.openai"],
    ["google", "gcp.gemini"],
    ["google-vertex", "gcp.vertex_ai"],
    ["google-vertex-anthropic", "gcp.vertex_ai"],
    ["mistral", "mistral_ai"],
    ["xai", "x_ai"],
  ])("maps %s to %s", (providerID, expected) => {
    expect(genAiProviderName(providerID)).toBe(expected)
  })

  test("preserves provider IDs without a canonical mapping", () => {
    expect(genAiProviderName("openrouter")).toBe("openrouter")
    expect(genAiProviderName("custom-gateway")).toBe("custom-gateway")
  })
})

describe("setBoundedMap", () => {
  test("adds an entry to the map", () => {
    const map = new Map<string, number>()
    setBoundedMap(map, "a", 1)
    expect(map.get("a")).toBe(1)
  })

  test("evicts the oldest entry when at capacity", () => {
    const map = new Map<string, number>()
    for (let i = 0; i < MAX_PENDING; i++) {
      setBoundedMap(map, `key-${i}`, i)
    }
    expect(map.size).toBe(MAX_PENDING)
    expect(map.has("key-0")).toBe(true)

    setBoundedMap(map, "overflow", 999)
    expect(map.size).toBe(MAX_PENDING)
    expect(map.has("key-0")).toBe(false)
    expect(map.has("overflow")).toBe(true)
  })

  test("does not evict when below capacity", () => {
    const map = new Map<string, number>()
    setBoundedMap(map, "a", 1)
    setBoundedMap(map, "b", 2)
    expect(map.size).toBe(2)
    expect(map.has("a")).toBe(true)
  })

  test("overwrites an existing key without evicting", () => {
    const map = new Map<string, number>()
    setBoundedMap(map, "a", 1)
    setBoundedMap(map, "a", 2)
    expect(map.get("a")).toBe(2)
    expect(map.size).toBe(1)
  })

  test("updates an existing key at capacity without evicting another entry", () => {
    const map = new Map<string, number>()
    for (let i = 0; i < MAX_PENDING; i++) {
      setBoundedMap(map, `key-${i}`, i)
    }

    setBoundedMap(map, "key-10", 1000)

    expect(map.size).toBe(MAX_PENDING)
    expect(map.get("key-10")).toBe(1000)
    expect(map.has("key-0")).toBe(true)
  })
})

describe("isTraceEnabled", () => {
  test("returns true when disabled set is empty", () => {
    expect(isTraceEnabled("llm", { disabledTraces: new Set() })).toBe(true)
  })

  test("returns false when trace type is in the disabled set", () => {
    expect(isTraceEnabled("tool", { disabledTraces: new Set(["tool"]) })).toBe(false)
  })

  test("returns false for llm when llm is disabled", () => {
    expect(isTraceEnabled("llm", { disabledTraces: new Set(["llm"]) })).toBe(false)
  })

  test("returns false for tool when tool is disabled", () => {
    expect(isTraceEnabled("tool", { disabledTraces: new Set(["tool"]) })).toBe(false)
  })

  test("returns true when a different trace type is disabled", () => {
    expect(isTraceEnabled("llm", { disabledTraces: new Set(["tool"]) })).toBe(true)
  })

  test("is case-sensitive — does not match mismatched case", () => {
    expect(isTraceEnabled("llm", { disabledTraces: new Set(["LLM"]) })).toBe(true)
  })

  test("unknown trace names in disabled set do not affect known types", () => {
    expect(isTraceEnabled("llm", { disabledTraces: new Set(["does_not_exist"]) })).toBe(true)
  })
})

describe("trace context resolution", () => {
  test("resolves a live interaction span", () => {
    const { ctx } = makeCtx()
    const interaction = ctx.tracer.startSpan("interaction")
    setBoundedMap(ctx.interactionSpans, "user_1", interaction)

    expect(trace.getSpan(resolveInteractionTraceContext("user_1", ctx))).toBe(interaction)
  })

  test("resolves the retained context of an ended interaction", () => {
    const { ctx } = makeCtx()
    const interaction = ctx.tracer.startSpan("interaction")
    setBoundedMap(ctx.interactionSpanContexts, "user_1", interaction.spanContext())

    expect(trace.getSpanContext(resolveInteractionTraceContext("user_1", ctx))?.spanId)
      .toBe(interaction.spanContext().spanId)
  })

  test("falls back from the session interaction to the active run", () => {
    const { ctx } = makeCtx()
    const run = ctx.tracer.startSpan("run")
    setBoundedMap(ctx.activeRunSpans, "ses_1", {
      span: run,
      agent: "build",
      agentType: "primary",
      tokens: 0,
      cost: 0,
      messages: 0,
      interactionIDs: new Set(),
      interactionIO: new Map(),
    })

    expect(trace.getSpan(resolveSessionTraceContext("ses_1", ctx))).toBe(run)
  })
})
