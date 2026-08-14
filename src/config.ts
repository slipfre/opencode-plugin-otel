import { LEVELS, type Level } from "./types.ts";

const DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT = 4096;
const DEFAULT_USER_ID_TIMEOUT = 3000;
const DEFAULT_USER_ID_RETRY_COUNT = 2;
const DEFAULT_USER_ID_COOLDOWN = 5 * 60 * 1000;
const MAX_USER_ID_RETRY_COUNT = 10;

/** Configuration values resolved from `OPENCODE_*` environment variables. */
export type PluginConfig = {
  enabled: boolean;
  endpoint: string;
  userIDEnabled: boolean;
  userIDEndpoint: string;
  userIDAuthHeader: string | undefined;
  userIDTimeout: number;
  userIDRetryCount: number;
  userIDCooldown: number;
  protocol: "grpc" | "http/protobuf" | "http/json";
  tracePrefix: string;
  otlpHeaders: string | undefined;
  otlpHeadersHelper: string | undefined;
  resourceAttributes: string | undefined;
  spanAttributes: string | undefined;
  spanAttributeCountLimit: number;
  traceparent: string | undefined;
  tracestate: string | undefined;
  tracePropagationProviders: Set<string>;
};

export function parseAttributePairs(
  raw: string | undefined
): Record<string, string> {
  const attrs: Record<string, string> = {};
  if (!raw) {
    return attrs;
  }

  for (const pair of raw.split(",")) {
    const idx = pair.indexOf("=");
    if (idx <= 0) {
      continue;
    }
    const key = pair.slice(0, idx).trim();
    const value = pair.slice(idx + 1).trim();
    if (!key) {
      continue;
    }
    attrs[key] = value;
  }

  return attrs;
}

/**
 * Options accepted via the opencode plugin tuple form
 * (`["opencode-plugin-otel", { ... }]`). Every field is optional; a provided
 * value takes precedence over the matching `OPENCODE_*` environment variable,
 * which in turn wins over the built-in default. Field names mirror the resolved
 * {@link PluginConfig}.
 */
export type OtelPluginOptions = {
  enabled?: boolean;
  endpoint?: string;
  userIDEnabled?: boolean;
  userIDEndpoint?: string;
  userIDAuthHeader?: string;
  userIDTimeout?: number;
  userIDRetryCount?: number;
  userIDCooldown?: number;
  protocol?: "grpc" | "http/protobuf" | "http/json";
  tracePrefix?: string;
  otlpHeaders?: string;
  otlpHeadersHelper?: string;
  resourceAttributes?: string;
  spanAttributes?: string;
  spanAttributeCountLimit?: number;
  traceparent?: string;
  tracestate?: string;
  tracePropagationProviders?: string[];
};

const VALID_PROTOCOLS = new Set<PluginConfig["protocol"]>([
  "grpc",
  "http/protobuf",
  "http/json",
]);

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function pickBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function pickPositiveInt(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : undefined;
}

function pickNonNegativeInt(
  value: unknown,
  maximum = Number.MAX_SAFE_INTEGER
): number | undefined {
  return typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= 0 &&
    value <= maximum
    ? value
    : undefined;
}

function pickBooleanString(value: unknown): boolean | undefined {
  if (typeof value !== "string") {
    return;
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "true" || normalized === "1") {
    return true;
  }
  if (normalized === "false" || normalized === "0") {
    return false;
  }
}

function pickStringList(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.filter((entry): entry is string => typeof entry === "string");
}

function pickProtocol(value: unknown): PluginConfig["protocol"] | undefined {
  return typeof value === "string" &&
    VALID_PROTOCOLS.has(value as PluginConfig["protocol"])
    ? (value as PluginConfig["protocol"])
    : undefined;
}

/** Parses a positive integer from an environment variable, returning `fallback` if absent or invalid. */
export function parseEnvInt(key: string, fallback: number): number {
  const raw = process.env[key];
  if (!raw) {
    return fallback;
  }
  if (!/^[1-9]\d*$/.test(raw)) {
    return fallback;
  }
  const n = Number(raw);
  return Number.isSafeInteger(n) ? n : fallback;
}

function parseEnvNonNegativeInt(
  key: string,
  fallback: number,
  maximum = Number.MAX_SAFE_INTEGER
): number {
  const raw = process.env[key];
  if (!raw || !/^\d+$/.test(raw)) {
    return fallback;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value <= maximum ? value : fallback;
}

/** Returns `true` when the environment variable is present and non-empty. */
function hasNonEmptyEnv(key: string): boolean {
  return !!process.env[key];
}

function splitList(raw: string | undefined): string[] {
  return (raw ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function normalizeList(values: string[]): string[] {
  return values.map((s) => s.trim()).filter(Boolean);
}

/**
 * Resolves the plugin config from plugin `options` and `OPENCODE_*` environment
 * variables. For every field a provided option wins over the environment
 * variable, which in turn wins over the built-in default.
 *
 * Copies the resolved headers and resource attributes into
 * `OTEL_EXPORTER_OTLP_HEADERS` and `OTEL_RESOURCE_ATTRIBUTES` so the OTel SDK
 * picks them up automatically when initialised.
 */
export function loadConfig(options: OtelPluginOptions = {}): PluginConfig {
  const resolvedOptions =
    typeof options === "object" && options !== null ? options : {};
  const otlpHeaders =
    pickString(resolvedOptions.otlpHeaders) ??
    process.env["OPENCODE_OTLP_HEADERS"];
  const otlpHeadersHelper =
    pickString(resolvedOptions.otlpHeadersHelper) ??
    process.env["OPENCODE_OTLP_HEADERS_HELPER"];
  const resourceAttributes =
    pickString(resolvedOptions.resourceAttributes) ??
    process.env["OPENCODE_RESOURCE_ATTRIBUTES"];
  const spanAttributes =
    pickString(resolvedOptions.spanAttributes) ??
    process.env["OPENCODE_SPAN_ATTRIBUTES"];
  const traceparent =
    pickString(resolvedOptions.traceparent) ??
    process.env["OPENCODE_TRACEPARENT"];
  const tracestate =
    pickString(resolvedOptions.tracestate) ??
    process.env["OPENCODE_TRACESTATE"];
  const protocol =
    pickProtocol(resolvedOptions.protocol) ??
    pickProtocol(process.env["OPENCODE_OTLP_PROTOCOL"]) ??
    "grpc";

  if (otlpHeaders) {
    process.env["OTEL_EXPORTER_OTLP_HEADERS"] = otlpHeaders;
  }
  if (resourceAttributes) {
    process.env["OTEL_RESOURCE_ATTRIBUTES"] = resourceAttributes;
  }

  const optionTracePropagationProviders = pickStringList(
    resolvedOptions.tracePropagationProviders
  );
  const tracePropagationProviders = new Set(
    optionTracePropagationProviders
      ? normalizeList(optionTracePropagationProviders)
      : splitList(process.env["OPENCODE_TRACE_PROPAGATION_PROVIDERS"])
  );

  return {
    enabled:
      pickBoolean(resolvedOptions.enabled) ??
      hasNonEmptyEnv("OPENCODE_ENABLE_TELEMETRY"),
    endpoint:
      pickString(resolvedOptions.endpoint) ??
      process.env["OPENCODE_OTLP_ENDPOINT"] ??
      "http://localhost:4317",
    userIDEnabled:
      pickBoolean(resolvedOptions.userIDEnabled) ??
      pickBooleanString(process.env["OPENCODE_USER_ID_ENABLED"]) ??
      true,
    userIDEndpoint:
      pickString(resolvedOptions.userIDEndpoint) ??
      process.env["OPENCODE_USER_ID_ENDPOINT"] ??
      "queryUserByToken",
    userIDAuthHeader:
      pickString(resolvedOptions.userIDAuthHeader) ??
      process.env["OPENCODE_USER_ID_X-Blackbox-Auth"],
    userIDTimeout:
      pickPositiveInt(resolvedOptions.userIDTimeout) ??
      parseEnvInt("OPENCODE_USER_ID_TIMEOUT", DEFAULT_USER_ID_TIMEOUT),
    userIDRetryCount:
      pickNonNegativeInt(
        resolvedOptions.userIDRetryCount,
        MAX_USER_ID_RETRY_COUNT
      ) ??
      parseEnvNonNegativeInt(
        "OPENCODE_USER_ID_RETRY_COUNT",
        DEFAULT_USER_ID_RETRY_COUNT,
        MAX_USER_ID_RETRY_COUNT
      ),
    userIDCooldown:
      pickNonNegativeInt(resolvedOptions.userIDCooldown) ??
      parseEnvNonNegativeInt(
        "OPENCODE_USER_ID_COOLDOWN",
        DEFAULT_USER_ID_COOLDOWN
      ),
    protocol,
    spanAttributeCountLimit:
      pickPositiveInt(resolvedOptions.spanAttributeCountLimit) ??
      parseEnvInt(
        "OPENCODE_SPAN_ATTRIBUTE_COUNT_LIMIT",
        DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT
      ),
    tracePrefix:
      pickString(resolvedOptions.tracePrefix) ??
      process.env["OPENCODE_TRACE_PREFIX"] ??
      "opencode.",
    otlpHeaders,
    otlpHeadersHelper,
    resourceAttributes,
    spanAttributes,
    traceparent,
    tracestate,
    tracePropagationProviders,
  };
}

export function resolveHelperPath(
  helper: string | undefined,
  directory: string | undefined,
  worktree: string | undefined
): string | undefined {
  if (!helper) {
    return helper;
  }
  const projectRoot = worktree ?? directory ?? process.cwd();
  return helper
    .replaceAll("${PROJECT_ROOT}", projectRoot)
    .replaceAll("${WORKTREE}", worktree ?? projectRoot)
    .replaceAll("${DIRECTORY}", directory ?? projectRoot);
}

/**
 * Resolves an opencode log level string to a `Level`.
 * Returns `current` unchanged when the input does not match a known level.
 */
export function resolveLogLevel(logLevel: string, current: Level): Level {
  const candidate = logLevel.toLowerCase();
  if (candidate in LEVELS) {
    return candidate as Level;
  }
  return current;
}
