import { trace } from "@opentelemetry/api"
import { BasicTracerProvider, BatchSpanProcessor } from "@opentelemetry/sdk-trace-base"
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc"
import { OTLPTraceExporter as OTLPHttpTraceExporter } from "@opentelemetry/exporter-trace-otlp-http"
import { OTLPTraceExporter as OTLPProtoTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto"
import { resourceFromAttributes } from "@opentelemetry/resources"
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions"
import { ATTR_HOST_ARCH } from "@opentelemetry/semantic-conventions/incubating"
import { parseAttributePairs } from "./config.ts"
import {
  createGrpcMetadata,
  DynamicHeaders,
  parseOtlpHeaders,
  RefreshingSpanExporter,
  type HeadersMap,
} from "./headers.ts"

/**
 * Builds an OTel `Resource` seeded with `service.name`, `app.version`, `os.type`, and
 * `host.arch`. Additional attributes from `OTEL_RESOURCE_ATTRIBUTES` are merged in and
 * may override the defaults.
 */
export function buildResource(version: string) {
  const attrs: Record<string, string> = {
    [ATTR_SERVICE_NAME]: "opencode",
    "app.version": version,
    "os.type": process.platform,
    [ATTR_HOST_ARCH]: process.arch,
    ...parseAttributePairs(process.env["OTEL_RESOURCE_ATTRIBUTES"]),
  }
  return resourceFromAttributes(attrs)
}

/** Handles returned by `setupOtel`, used for graceful shutdown. */
export type OtelProviders = {
  tracerProvider: BasicTracerProvider
}

function buildHttpTraceUrl(endpoint: string) {
  const url = new URL(endpoint)
  const normalizedPath = url.pathname.endsWith("/") ? url.pathname.slice(0, -1) : url.pathname
  url.pathname = `${normalizedPath}/v1/traces`
  return url.toString()
}

/**
 * Initialises the OTel SDK with a `BasicTracerProvider` backed by an OTLP
 * exporter pointed at `endpoint`, and registers it as the global provider.
 */
export async function setupOtel(
  endpoint: string,
  protocol: "grpc" | "http/protobuf" | "http/json",
  version: string,
  otlpHeaders?: string,
  otlpHeadersHelper?: string,
  spanAttributeCountLimit = 4096,
): Promise<OtelProviders> {
  const resource = buildResource(version)
  const staticHeaders = parseOtlpHeaders(otlpHeaders)
  const dynamicHeaders = new DynamicHeaders(staticHeaders, otlpHeadersHelper)
  if (otlpHeadersHelper) {
    try {
      await dynamicHeaders.refresh()
    } catch (error) {
      console.warn("[opencode-plugin-otel] Failed to prewarm OTLP headers helper. Falling back to refresh-on-auth-failure.", error)
    }
  }
  const makeTraceExporter = (headers: HeadersMap) => protocol === "http/protobuf"
    ? new OTLPProtoTraceExporter({ url: buildHttpTraceUrl(endpoint), headers })
    : protocol === "http/json"
      ? new OTLPHttpTraceExporter({ url: buildHttpTraceUrl(endpoint), headers })
      : new OTLPTraceExporter({ url: endpoint, metadata: createGrpcMetadata(headers) })
  const traceExporter = otlpHeadersHelper
    ? new RefreshingSpanExporter(makeTraceExporter, dynamicHeaders)
    : makeTraceExporter(staticHeaders)

  const tracerProvider = new BasicTracerProvider({
    resource,
    spanLimits: { attributeCountLimit: spanAttributeCountLimit },
    spanProcessors: [new BatchSpanProcessor(traceExporter)],
  })
  trace.setGlobalTracerProvider(tracerProvider)

  return { tracerProvider }
}
