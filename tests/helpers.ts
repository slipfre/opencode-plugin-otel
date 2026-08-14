import type { HandlerContext } from "../src/types.ts";
import { createInteractionState } from "../src/interaction.ts";
import { createCompactionState } from "../src/compaction.ts";
import type {
  SpanOptions,
  Tracer,
  Context,
  SpanContext,
  SpanStatus,
  Attributes,
  Link,
} from "@opentelemetry/api";
import {
  ROOT_CONTEXT,
  SpanKind,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";

export type SpyPluginLog = {
  calls: Array<{
    level: string;
    message: string;
    extra?: Record<string, unknown>;
  }>;
  fn: HandlerContext["log"];
};

export type SpySpan = {
  name: string;
  startTime?: number;
  endTime?: number | undefined;
  ended: boolean;
  status: SpanStatus;
  attributes: Record<string, unknown>;
  kind: SpanKind;
  links: Link[];
  parentSpan: SpySpan | undefined;
  parentSpanContext: SpanContext | undefined;
  setStatus(status: SpanStatus): SpySpan;
  setAttribute(key: string, value: unknown): SpySpan;
  setAttributes(attrs: Attributes): SpySpan;
  end(endTime?: number): void;
  isRecording(): boolean;
  spanContext(): SpanContext;
  addEvent(name: string): SpySpan;
  recordException(): SpySpan;
  updateName(name: string): SpySpan;
};

export type SpyTracer = {
  spans: SpySpan[];
  startSpan(name: string, options?: SpanOptions, ctx?: Context): SpySpan;
};

function makePluginLog(): SpyPluginLog {
  const spy: SpyPluginLog = {
    calls: [],
    fn: async (level, message, extra) => {
      spy.calls.push({ level, message, extra });
    },
  };
  return spy;
}

function makeSpan(
  name: string,
  startTime?: number,
  parentSpan?: SpySpan,
  parentSpanContext?: SpanContext,
  ownSpanContext?: SpanContext,
  kind = SpanKind.INTERNAL,
  links: Link[] = []
): SpySpan {
  const context = ownSpanContext ?? {
    traceId: "00000000000000000000000000000001",
    spanId: "0000000000000001",
    traceFlags: 1,
  };
  const span: SpySpan = {
    name,
    startTime,
    endTime: undefined,
    ended: false,
    status: { code: SpanStatusCode.UNSET },
    attributes: {},
    kind,
    links,
    parentSpan,
    parentSpanContext,
    setStatus(s) {
      span.status = s;
      return span;
    },
    setAttribute(k, v) {
      span.attributes[k] = v;
      return span;
    },
    setAttributes(attrs) {
      Object.assign(span.attributes, attrs);
      return span;
    },
    end(t) {
      span.ended = true;
      span.endTime = t;
    },
    isRecording() {
      return !span.ended;
    },
    spanContext() {
      return context;
    },
    addEvent() {
      return span;
    },
    recordException() {
      return span;
    },
    updateName(n) {
      span.name = n;
      return span;
    },
  };
  return span;
}

export function makeTracer(): SpyTracer {
  let nextSpanID = 1;
  let nextTraceID = 1;
  const tracer: SpyTracer = {
    spans: [],
    startSpan(name, options, ctx) {
      const parentFromCtx = ctx
        ? (trace.getSpan(ctx) as SpySpan | undefined)
        : undefined;
      const parentSpanContext = ctx
        ? (trace.getSpanContext(ctx) ?? undefined)
        : undefined;
      const ownSpanContext: SpanContext = {
        traceId:
          parentSpanContext?.traceId ??
          (nextTraceID++).toString(16).padStart(32, "0"),
        spanId: (nextSpanID++).toString(16).padStart(16, "0"),
        traceFlags: parentSpanContext?.traceFlags ?? 1,
        ...(parentSpanContext?.traceState
          ? { traceState: parentSpanContext.traceState }
          : {}),
      };
      const span = makeSpan(
        name,
        typeof options?.startTime === "number" ? options.startTime : undefined,
        parentFromCtx,
        parentSpanContext,
        ownSpanContext,
        options?.kind ?? SpanKind.INTERNAL,
        options?.links ? [...options.links] : []
      );
      if (options?.attributes) {
        Object.assign(span.attributes, options.attributes);
      }
      tracer.spans.push(span);
      return span;
    },
  };
  return tracer;
}

export type MockContext = {
  ctx: HandlerContext;
  pluginLog: SpyPluginLog;
  tracer: SpyTracer;
};

export function makeCtx(
  projectID = "proj_test",
  extraCommonAttrs: Record<string, string> = {}
): MockContext {
  const pluginLog = makePluginLog();
  const tracer = makeTracer();

  const ctx: HandlerContext = {
    log: pluginLog.fn,
    commonAttrs: { "project.id": projectID, ...extraCommonAttrs },
    pendingToolSpans: new Map(),
    pendingPermissionSpans: new Map(),
    tracer: tracer as unknown as Tracer,
    tracePrefix: "opencode.",
    rootContext: () => ROOT_CONTEXT,
    activeRunSpans: new Map(),
    ...createInteractionState(),
    ...createCompactionState(),
    pendingSubagentRuns: new Map(),
    sessionParents: new Map(),
    messageSpans: new Map(),
    messageOutputs: new Map(),
    llmRequestContexts: new Map(),
    llmTelemetryBindings: {
      pendingByRequestID: new Map(),
      byLifecycleMetadata: new WeakMap(),
    },
    tracePropagationProviders: new Set(),
    activeMessageSpans: new Map(),
    llmTelemetryOutputs: new Map(),
  };

  return { ctx, pluginLog, tracer };
}
