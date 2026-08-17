export type ExportedSpan = {
  name: string;
  traceId: string;
  spanId: string;
  parentSpanId: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: Record<string, unknown>;
  resource: Record<string, unknown>;
  status: Record<string, unknown>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function decodeValue(value: unknown): unknown {
  if (!isRecord(value)) {
    return value;
  }
  if ("stringValue" in value) {
    return value.stringValue;
  }
  if ("boolValue" in value) {
    return value.boolValue;
  }
  if ("intValue" in value) {
    const number = Number(value.intValue);
    return Number.isSafeInteger(number) ? number : value.intValue;
  }
  if ("doubleValue" in value) {
    return value.doubleValue;
  }
  if ("bytesValue" in value) {
    return value.bytesValue;
  }
  if (isRecord(value.arrayValue)) {
    return records(value.arrayValue.values).map(decodeValue);
  }
  if (isRecord(value.kvlistValue)) {
    return decodeAttributes(value.kvlistValue.values);
  }
  return value;
}

function decodeAttributes(value: unknown): Record<string, unknown> {
  const attributes: Record<string, unknown> = {};
  for (const item of records(value)) {
    if (typeof item.key !== "string") {
      continue;
    }
    attributes[item.key] = decodeValue(item.value);
  }
  return attributes;
}

function flattenExport(value: unknown): ExportedSpan[] {
  if (!isRecord(value)) {
    return [];
  }
  const spans: ExportedSpan[] = [];
  for (const resourceSpan of records(value.resourceSpans)) {
    const resource = isRecord(resourceSpan.resource)
      ? decodeAttributes(resourceSpan.resource.attributes)
      : {};
    const groups = [
      ...records(resourceSpan.scopeSpans),
      ...records(resourceSpan.instrumentationLibrarySpans),
    ];
    for (const group of groups) {
      for (const span of records(group.spans)) {
        spans.push({
          name: typeof span.name === "string" ? span.name : "",
          traceId: typeof span.traceId === "string" ? span.traceId : "",
          spanId: typeof span.spanId === "string" ? span.spanId : "",
          parentSpanId:
            typeof span.parentSpanId === "string" ? span.parentSpanId : "",
          kind: typeof span.kind === "number" ? span.kind : Number(span.kind),
          startTimeUnixNano:
            typeof span.startTimeUnixNano === "string"
              ? span.startTimeUnixNano
              : "",
          endTimeUnixNano:
            typeof span.endTimeUnixNano === "string"
              ? span.endTimeUnixNano
              : "",
          attributes: decodeAttributes(span.attributes),
          resource,
          status: isRecord(span.status) ? span.status : {},
        });
      }
    }
  }
  return spans;
}

export function startOtlpReceiver() {
  const payloads: unknown[] = [];
  const headers: Headers[] = [];
  const errors: string[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      if (
        request.method !== "POST" ||
        new URL(request.url).pathname !== "/v1/traces"
      ) {
        return new Response("not found", { status: 404 });
      }
      headers.push(new Headers(request.headers));
      try {
        payloads.push(await request.json());
      } catch (error) {
        errors.push(error instanceof Error ? error.message : String(error));
        return Response.json({}, { status: 400 });
      }
      return Response.json({});
    },
  });
  const spans = () => payloads.flatMap(flattenExport);
  return {
    endpoint: `http://${server.hostname}:${server.port}`,
    payloads,
    headers,
    errors,
    spans,
    async waitForSpanCount(count: number, timeoutMs = 5_000) {
      const deadline = Date.now() + timeoutMs;
      while (spans().length < count && Date.now() < deadline) {
        await Bun.sleep(20);
      }
      return spans();
    },
    stop: () => server.stop(true),
  };
}
