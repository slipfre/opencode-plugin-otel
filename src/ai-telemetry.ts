import type { Attributes } from "@opentelemetry/api"
import type { OnStartEvent, OnStepFinishEvent, OnStepStartEvent, TelemetryIntegration } from "ai"
import { registerTelemetryIntegration } from "ai"
import {
  IMAGE_URL,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  LLM_INVOCATION_PARAMETERS,
  LLM_OUTPUT_MESSAGES,
  LLM_TOOLS,
  MESSAGE_CONTENT,
  MESSAGE_CONTENT_IMAGE,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENTS,
  MESSAGE_ROLE,
  MESSAGE_TOOL_CALL_ID,
  MimeType,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
  TOOL_CALL_FUNCTION_NAME,
  TOOL_CALL_ID,
  TOOL_JSON_SCHEMA,
} from "@arizeai/openinference-semantic-conventions"
import { setBoundedMap } from "./util.ts"
import type { HandlerContext } from "./types.ts"

type Listener = {
  onStart(event: OnStartEvent): void
  onStepStart(event: OnStepStartEvent): void
  onStepFinish(event: OnStepFinishEvent): void
}

type Broker = {
  installed: boolean
  listeners: Set<Listener>
}

type TelemetryGlobal = typeof globalThis & {
  __opencodePluginOtelAiTelemetry?: Broker
}

type CleanContent =
  | { type: "text" | "reasoning"; text: string }
  | { type: "image"; imageUrl: string }
  | { type: "tool_use"; id?: string; name?: string; arguments?: unknown }

type CleanMessage = {
  role: string
  content: string | CleanContent[]
  toolCallId?: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function json(value: unknown): string {
  const seen = new WeakSet<object>()
  try {
    return JSON.stringify(value, (_key, current: unknown) => {
      if (typeof current === "bigint") return current.toString()
      if (typeof current === "function" || typeof current === "symbol") return undefined
      if (current instanceof Uint8Array) return Buffer.from(current).toString("base64")
      if (current && typeof current === "object") {
        if (seen.has(current)) return "[Circular]"
        seen.add(current)
      }
      return current
    }) ?? "null"
  } catch (error) {
    return JSON.stringify({ serializationError: error instanceof Error ? error.message : String(error) })
  }
}

function sessionID(metadata: Record<string, unknown> | undefined): string | undefined {
  const value = metadata?.sessionId
  return typeof value === "string" ? value : undefined
}

function active(
  input: { functionId: string | undefined; metadata: Record<string, unknown> | undefined },
  ctx: HandlerContext,
) {
  if (input.functionId !== "session.llm") return
  const session = sessionID(input.metadata)
  if (!session) return
  const current = ctx.activeMessageSpans.get(session)
  if (!current) return
  return { msgKey: `${session}:${current.messageID}`, ...current }
}

function imageUrl(value: unknown, mediaType: unknown): string | undefined {
  if (value instanceof URL) return value.toString()
  if (typeof value === "string") {
    if (/^(?:data:|https?:\/\/)/.test(value) || typeof mediaType !== "string") return value
    return `data:${mediaType};base64,${value}`
  }
  if (value instanceof Uint8Array && typeof mediaType === "string") {
    return `data:${mediaType};base64,${Buffer.from(value).toString("base64")}`
  }
  if (value instanceof ArrayBuffer && typeof mediaType === "string") {
    return `data:${mediaType};base64,${Buffer.from(value).toString("base64")}`
  }
}

function cleanContentPart(value: unknown): CleanContent | undefined {
  if (!isRecord(value) || typeof value.type !== "string") return
  if ((value.type === "text" || value.type === "reasoning") && typeof value.text === "string") {
    return { type: value.type, text: value.text }
  }
  if (value.type === "tool-call") {
    return {
      type: "tool_use",
      ...(typeof value.toolCallId === "string" ? { id: value.toolCallId } : {}),
      ...(typeof value.toolName === "string" ? { name: value.toolName } : {}),
      ...(value.input !== undefined ? { arguments: value.input } : value.args !== undefined ? { arguments: value.args } : {}),
    }
  }
  const file = value.type === "file" && isRecord(value.file) ? value.file : value
  if (value.type === "image" || (value.type === "file" && typeof file.mediaType === "string" && file.mediaType.startsWith("image/"))) {
    const url = imageUrl(value.image ?? file.data ?? file.base64 ?? file.uint8Array ?? file.url, file.mediaType)
    if (url) return { type: "image", imageUrl: url }
  }
}

function toolResultContent(value: unknown): string {
  if (typeof value === "string") return value
  if (isRecord(value) && value.value !== undefined) {
    return typeof value.value === "string" ? value.value : json(value.value)
  }
  return json(value)
}

function cleanMessages(values: unknown[]): CleanMessage[] {
  const messages: CleanMessage[] = []
  for (const value of values) {
    if (!isRecord(value) || typeof value.role !== "string") continue
    const role = value.role
    if (role === "tool" && Array.isArray(value.content)) {
      let added = false
      for (const part of value.content) {
        if (!isRecord(part) || part.type !== "tool-result") continue
        messages.push({
          role,
          content: toolResultContent(part.output),
          ...(typeof part.toolCallId === "string" ? { toolCallId: part.toolCallId } : {}),
        })
        added = true
      }
      if (added) continue
    }
    if (typeof value.content === "string") {
      messages.push({ role, content: value.content })
      continue
    }
    if (!Array.isArray(value.content)) continue
    const content = value.content.map(cleanContentPart).filter((part): part is CleanContent => part !== undefined)
    if (content.length > 0) messages.push({ role, content })
  }
  return messages
}

function providerInstructions(value: unknown): string | undefined {
  if (!isRecord(value)) return
  if (typeof value.instructions === "string") return value.instructions
  for (const options of Object.values(value)) {
    if (isRecord(options) && typeof options.instructions === "string") return options.instructions
  }
}

function inputMessages(event: OnStepStartEvent): CleanMessage[] {
  const system = event.system === undefined
    ? []
    : typeof event.system === "string"
      ? [{ role: "system", content: event.system }]
      : Array.isArray(event.system)
        ? event.system
        : [event.system]
  const messages = cleanMessages([...system, ...event.messages])
  const instructions = providerInstructions(event.providerOptions)
  if (instructions && !messages.some((message) => message.role === "system")) {
    messages.unshift({ role: "system", content: instructions })
  }
  return messages
}

function outputMessages(event: OnStepFinishEvent): CleanMessage[] {
  const messages = cleanMessages(event.response.messages).filter((message) => message.role === "assistant")
  const generated = event.content.map(cleanContentPart).filter((part): part is CleanContent => part !== undefined)
  if (generated.length === 0) return messages
  const assistant = messages.find((message) => message.role === "assistant")
  if (assistant) assistant.content = generated
  else messages.push({ role: "assistant", content: generated })
  return messages
}

function messageAttributes(prefix: string, messages: CleanMessage[]): Attributes {
  const attributes: Attributes = {}
  messages.forEach((message, messageIndex) => {
    const messagePrefix = `${prefix}.${messageIndex}`
    attributes[`${messagePrefix}.${MESSAGE_ROLE}`] = message.role
    if (message.toolCallId) attributes[`${messagePrefix}.${MESSAGE_TOOL_CALL_ID}`] = message.toolCallId
    if (typeof message.content === "string") {
      attributes[`${messagePrefix}.${MESSAGE_CONTENT}`] = message.content
      return
    }
    message.content.forEach((content, contentIndex) => {
      const contentPrefix = `${messagePrefix}.${MESSAGE_CONTENTS}.${contentIndex}`
      attributes[`${contentPrefix}.${MESSAGE_CONTENT_TYPE}`] = content.type
      if (content.type === "text" || content.type === "reasoning") {
        attributes[`${contentPrefix}.${MESSAGE_CONTENT_TEXT}`] = content.text
      } else if (content.type === "image") {
        attributes[`${contentPrefix}.${MESSAGE_CONTENT_IMAGE}.${IMAGE_URL}`] = content.imageUrl
      } else if (content.type === "tool_use") {
        if (content.id) attributes[`${contentPrefix}.${TOOL_CALL_ID}`] = content.id
        if (content.name) attributes[`${contentPrefix}.${TOOL_CALL_FUNCTION_NAME}`] = content.name
        if (content.arguments !== undefined) {
          attributes[`${contentPrefix}.${TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] = json(content.arguments)
        }
      }
    })
  })
  return attributes
}

function headerAttribute(key: "http.request.headers" | "http.response.headers", headers: Record<string, string | undefined> | undefined): Attributes {
  if (!headers) return {}
  const values: Record<string, string> = {}
  for (const [name, value] of Object.entries(headers)) {
    const normalized = name.trim().toLowerCase()
    if (!normalized || value === undefined) continue
    values[normalized] = value
  }
  return Object.keys(values).length > 0 ? { [key]: json(values) } : {}
}

function invocationParameters(event: OnStartEvent): Record<string, unknown> {
  return Object.fromEntries(Object.entries({
    maxOutputTokens: event.maxOutputTokens,
    temperature: event.temperature,
    topP: event.topP,
    topK: event.topK,
    presencePenalty: event.presencePenalty,
    frequencyPenalty: event.frequencyPenalty,
    stopSequences: event.stopSequences,
    seed: event.seed,
    toolChoice: event.toolChoice,
    providerOptions: event.providerOptions,
  }).filter(([, value]) => value !== undefined))
}

function inputSchema(value: unknown): unknown {
  return isRecord(value) && value.jsonSchema !== undefined ? value.jsonSchema : value
}

function toolAttributes(event: OnStepStartEvent): Attributes {
  const attributes: Attributes = {}
  if (!event.tools) return attributes
  const activeTools = event.activeTools ? new Set<string>(event.activeTools) : undefined
  const tools = Object.entries(event.tools).filter(([name]) => !activeTools || activeTools.has(name))
  tools.forEach(([name, value], index) => {
    if (!isRecord(value)) return
    attributes[`${LLM_TOOLS}.${index}.${TOOL_JSON_SCHEMA}`] = json({
      type: "function",
      function: {
        name,
        ...(typeof value.description === "string" ? { description: value.description } : {}),
        parameters: inputSchema(value.inputSchema) ?? {},
      },
    })
  })
  return attributes
}

function handleAiTelemetryStart(event: OnStartEvent, ctx: HandlerContext) {
  const current = active(event, ctx)
  if (!current) return
  current.span.setAttribute(LLM_INVOCATION_PARAMETERS, json(invocationParameters(event)))
}

function handleAiTelemetryStepStart(event: OnStepStartEvent, ctx: HandlerContext) {
  const current = active(event, ctx)
  if (!current) return
  const messages = inputMessages(event)
  current.span.setAttributes({
    [INPUT_VALUE]: json(messages),
    [INPUT_MIME_TYPE]: MimeType.JSON,
    ...messageAttributes(LLM_INPUT_MESSAGES, messages),
    ...toolAttributes(event),
    ...headerAttribute("http.request.headers", event.headers),
  })
}

function handleAiTelemetryStepFinish(event: OnStepFinishEvent, ctx: HandlerContext) {
  const current = active(event, ctx)
  if (!current) return
  const messages = outputMessages(event)
  current.span.setAttributes({
    [OUTPUT_VALUE]: json(messages),
    [OUTPUT_MIME_TYPE]: MimeType.JSON,
    ...messageAttributes(LLM_OUTPUT_MESSAGES, messages),
    ...headerAttribute("http.response.headers", event.response.headers),
  })
  setBoundedMap(ctx.llmTelemetryOutputs, current.msgKey, true)
}

function getBroker(): Broker {
  const root = globalThis as TelemetryGlobal
  root.__opencodePluginOtelAiTelemetry ??= { installed: false, listeners: new Set() }
  const broker = root.__opencodePluginOtelAiTelemetry
  if (!broker.installed) {
    const integration: TelemetryIntegration = {
      onStart(event) {
        for (const listener of broker.listeners) {
          try { listener.onStart(event) } catch {}
        }
      },
      onStepStart(event) {
        for (const listener of broker.listeners) {
          try { listener.onStepStart(event) } catch {}
        }
      },
      onStepFinish(event) {
        for (const listener of broker.listeners) {
          try { listener.onStepFinish(event) } catch {}
        }
      },
    }
    registerTelemetryIntegration(integration)
    broker.installed = true
  }
  return broker
}

export function registerAiTelemetry(ctx: HandlerContext): () => void {
  const broker = getBroker()
  const listener: Listener = {
    onStart: (event) => handleAiTelemetryStart(event, ctx),
    onStepStart: (event) => handleAiTelemetryStepStart(event, ctx),
    onStepFinish: (event) => handleAiTelemetryStepFinish(event, ctx),
  }
  broker.listeners.add(listener)
  return () => broker.listeners.delete(listener)
}
