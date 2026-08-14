import { describe, test, expect, afterEach } from "bun:test";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { OTLPTraceExporter as OTLPHttpTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OTLPTraceExporter as OTLPProtoTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { buildResource, setupOtel, type OtelProviders } from "../src/otel.ts";

let providers: OtelProviders | undefined;

function exporterOf(currentProviders: OtelProviders) {
  const tracerProvider = currentProviders.tracerProvider as unknown as {
    _activeSpanProcessor: { _spanProcessors: Array<{ _exporter: unknown }> };
  };
  const spanProcessor = tracerProvider._activeSpanProcessor._spanProcessors[0];
  if (!spanProcessor) {
    throw new Error("Expected an active trace exporter");
  }
  return spanProcessor._exporter;
}

describe("buildResource", () => {
  const originalEnv = process.env["OTEL_RESOURCE_ATTRIBUTES"];
  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env["OTEL_RESOURCE_ATTRIBUTES"];
    } else {
      process.env["OTEL_RESOURCE_ATTRIBUTES"] = originalEnv;
    }
  });

  test("includes service.name, app.version, os.type, host.arch", () => {
    delete process.env["OTEL_RESOURCE_ATTRIBUTES"];
    const resource = buildResource("1.2.3");
    const attrs = resource.attributes;
    expect(attrs["service.name"]).toBe("opencode");
    expect(attrs["app.version"]).toBe("1.2.3");
    expect(attrs["os.type"]).toBe(process.platform);
    expect(attrs["host.arch"]).toBe(process.arch);
  });

  test("merges OTEL_RESOURCE_ATTRIBUTES from env", () => {
    process.env["OTEL_RESOURCE_ATTRIBUTES"] = "team=platform,env=prod";
    const attrs = buildResource("0.0.1").attributes;
    expect(attrs["team"]).toBe("platform");
    expect(attrs["env"]).toBe("prod");
  });

  test("trims whitespace in resource attributes", () => {
    process.env["OTEL_RESOURCE_ATTRIBUTES"] = " team = platform ";
    expect(buildResource("0.0.1").attributes["team"]).toBe("platform");
  });

  test("resource attribute values may contain equals signs", () => {
    process.env["OTEL_RESOURCE_ATTRIBUTES"] = "auth=Bearer abc=123";
    expect(buildResource("0.0.1").attributes["auth"]).toBe("Bearer abc=123");
  });

  test("env resource attributes override defaults", () => {
    process.env["OTEL_RESOURCE_ATTRIBUTES"] = "service.name=my-override";
    expect(buildResource("0.0.1").attributes["service.name"]).toBe(
      "my-override"
    );
  });
});

describe("setupOtel", () => {
  afterEach(async () => {
    const current = providers;
    providers = undefined;
    if (current) {
      await current.tracerProvider.shutdown();
    }
  });

  test("uses the protobuf HTTP exporter for http/protobuf", async () => {
    providers = await setupOtel(
      "http://collector:4318",
      "http/protobuf",
      "1.2.3"
    );
    expect(exporterOf(providers)).toBeInstanceOf(OTLPProtoTraceExporter);
  });

  test("uses the gRPC exporter for grpc", async () => {
    providers = await setupOtel("http://collector:4317", "grpc", "1.2.3");
    expect(exporterOf(providers)).toBeInstanceOf(OTLPTraceExporter);
  });

  test("uses the JSON HTTP exporter for http/json", async () => {
    providers = await setupOtel("http://collector:4318", "http/json", "1.2.3");
    expect(exporterOf(providers)).toBeInstanceOf(OTLPHttpTraceExporter);
  });

  test("supports more than 128 span attributes", async () => {
    providers = await setupOtel("http://collector:4317", "grpc", "1.2.3");
    const span = providers.tracerProvider
      .getTracer("test")
      .startSpan("many-attributes");
    for (let index = 0; index < 256; index++) {
      span.setAttribute(`attribute.${index}`, index);
    }
    const readable = span as unknown as {
      attributes: Record<string, unknown>;
      droppedAttributesCount: number;
    };
    expect(readable.attributes["attribute.255"]).toBe(255);
    expect(readable.droppedAttributesCount).toBe(0);
  });
});
