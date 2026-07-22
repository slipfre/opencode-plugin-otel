import { defaultTextMapGetter, defaultTextMapSetter, ROOT_CONTEXT, trace, type Context } from "@opentelemetry/api"
import { W3CTraceContextPropagator } from "@opentelemetry/core"

const propagator = new W3CTraceContextPropagator()

/** Writes W3C `traceparent`/`tracestate` headers for the span in `ctx` into `headers`; no-op without a valid span context. */
export function injectTraceContext(ctx: Context, headers: Record<string, string>) {
  propagator.inject(ctx, headers, defaultTextMapSetter)
}

/** Builds a remote parent context from W3C trace-context headers. */
export function remoteParentContext(traceparent: string | undefined, tracestate: string | undefined): Context | undefined {
  if (!traceparent) return undefined

  const carrier = tracestate ? { traceparent, tracestate } : { traceparent }
  const extracted = propagator.extract(ROOT_CONTEXT, carrier, defaultTextMapGetter)
  return trace.getSpanContext(extracted) ? extracted : undefined
}
