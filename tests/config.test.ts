import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  loadConfig,
  parseAttributePairs,
  parseEnvInt,
  resolveHelperPath,
  resolveLogLevel,
} from "../src/config.ts";

const ENV_KEYS = [
  "OPENCODE_ENABLE_TELEMETRY",
  "OPENCODE_OTLP_ENDPOINT",
  "OPENCODE_OTLP_PROTOCOL",
  "OPENCODE_TRACE_PREFIX",
  "OPENCODE_OTLP_HEADERS",
  "OPENCODE_OTLP_HEADERS_HELPER",
  "OPENCODE_RESOURCE_ATTRIBUTES",
  "OPENCODE_SPAN_ATTRIBUTES",
  "OPENCODE_SPAN_ATTRIBUTE_COUNT_LIMIT",
  "OPENCODE_TRACEPARENT",
  "OPENCODE_TRACESTATE",
  "OPENCODE_TRACE_PROPAGATION_PROVIDERS",
  "OPENCODE_USER_ID_ENABLED",
  "OPENCODE_USER_ID_ENDPOINT",
  "OPENCODE_USER_ID_X-Blackbox-Auth",
  "OPENCODE_USER_ID_TIMEOUT",
  "OPENCODE_USER_ID_RETRY_COUNT",
  "OPENCODE_USER_ID_COOLDOWN",
  "OPENCODE_USER_ID_TRACESTATE_ENABLED",
  "OPENCODE_USER_ID_TRACESTATE_KEY",
  "OTEL_EXPORTER_OTLP_HEADERS",
  "OTEL_RESOURCE_ATTRIBUTES",
] as const;

const originalEnv = new Map<string, string | undefined>();

beforeEach(() => {
  for (const key of ENV_KEYS) {
    originalEnv.set(key, process.env[key]);
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    const value = originalEnv.get(key);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  originalEnv.clear();
});

describe("parseAttributePairs", () => {
  test("parses comma-separated attributes", () => {
    expect(parseAttributePairs("team=platform,env=prod")).toEqual({
      team: "platform",
      env: "prod",
    });
  });

  test("trims whitespace and preserves equals signs in values", () => {
    expect(
      parseAttributePairs(" team = platform , auth = Bearer abc=123 ")
    ).toEqual({
      team: "platform",
      auth: "Bearer abc=123",
    });
  });

  test("ignores malformed pairs", () => {
    expect(parseAttributePairs("good=value,bad,=missing")).toEqual({
      good: "value",
    });
  });

  test("returns an empty object for an absent value", () => {
    expect(parseAttributePairs(undefined)).toEqual({});
  });
});

describe("parseEnvInt", () => {
  test("returns a positive integer", () => {
    process.env["OPENCODE_USER_ID_TIMEOUT"] = "1234";
    expect(parseEnvInt("OPENCODE_USER_ID_TIMEOUT", 10)).toBe(1234);
  });

  test.each(["", "0", "-1", "1.5", "nope"])("falls back for %j", (value) => {
    process.env["OPENCODE_USER_ID_TIMEOUT"] = value;
    expect(parseEnvInt("OPENCODE_USER_ID_TIMEOUT", 10)).toBe(10);
  });
});

describe("loadConfig", () => {
  test("uses trace-only defaults", () => {
    const cfg = loadConfig();
    expect(cfg.enabled).toBe(false);
    expect(cfg.endpoint).toBe("http://localhost:4317");
    expect(cfg.protocol).toBe("grpc");
    expect(cfg.tracePrefix).toBe("opencode.");
    expect(cfg.spanAttributeCountLimit).toBe(4096);
    expect(cfg.tracePropagationProviders.size).toBe(0);
  });

  test("enables tracing when OPENCODE_ENABLE_TELEMETRY is non-empty", () => {
    process.env["OPENCODE_ENABLE_TELEMETRY"] = "1";
    expect(loadConfig().enabled).toBe(true);
  });

  test("reads endpoint, protocol, and trace prefix", () => {
    process.env["OPENCODE_OTLP_ENDPOINT"] =
      "https://collector.example.com/otlp";
    process.env["OPENCODE_OTLP_PROTOCOL"] = "http/protobuf";
    process.env["OPENCODE_TRACE_PREFIX"] = "custom.";
    const cfg = loadConfig();
    expect(cfg.endpoint).toBe("https://collector.example.com/otlp");
    expect(cfg.protocol).toBe("http/protobuf");
    expect(cfg.tracePrefix).toBe("custom.");
  });

  test("falls back for an invalid protocol", () => {
    process.env["OPENCODE_OTLP_PROTOCOL"] = "udp";
    expect(loadConfig().protocol).toBe("grpc");
  });

  test("reads trace parent settings", () => {
    process.env["OPENCODE_TRACEPARENT"] =
      "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
    process.env["OPENCODE_TRACESTATE"] = "vendor=value";
    const cfg = loadConfig();
    expect(cfg.traceparent).toBe(process.env["OPENCODE_TRACEPARENT"]);
    expect(cfg.tracestate).toBe("vendor=value");
  });

  test("copies headers and resource attributes to OTel environment variables", () => {
    process.env["OPENCODE_OTLP_HEADERS"] = "Authorization=Bearer token";
    process.env["OPENCODE_RESOURCE_ATTRIBUTES"] = "team=platform";
    const cfg = loadConfig();
    expect(cfg.otlpHeaders).toBe("Authorization=Bearer token");
    expect(cfg.resourceAttributes).toBe("team=platform");
    expect(process.env["OTEL_EXPORTER_OTLP_HEADERS"]).toBe(
      "Authorization=Bearer token"
    );
    expect(process.env["OTEL_RESOURCE_ATTRIBUTES"]).toBe("team=platform");
  });

  test("reads span attributes and attribute limit", () => {
    process.env["OPENCODE_SPAN_ATTRIBUTES"] = "team=platform";
    process.env["OPENCODE_SPAN_ATTRIBUTE_COUNT_LIMIT"] = "8192";
    const cfg = loadConfig();
    expect(cfg.spanAttributes).toBe("team=platform");
    expect(cfg.spanAttributeCountLimit).toBe(8192);
  });

  test("parses trace propagation providers", () => {
    process.env["OPENCODE_TRACE_PROPAGATION_PROVIDERS"] =
      " gateway-a , gateway-b ";
    expect(loadConfig().tracePropagationProviders).toEqual(
      new Set(["gateway-a", "gateway-b"])
    );
  });

  test("uses user ID defaults", () => {
    const cfg = loadConfig();
    expect(cfg.userIDEnabled).toBe(true);
    expect(cfg.userIDEndpoint).toBe("queryUserByToken");
    expect(cfg.userIDAuthHeader).toBeUndefined();
    expect(cfg.userIDTimeout).toBe(3000);
    expect(cfg.userIDRetryCount).toBe(2);
    expect(cfg.userIDCooldown).toBe(300000);
    expect(cfg.userIDTracestateEnabled).toBe(true);
    expect(cfg.userIDTracestateKey).toBe("opencode_user_id");
  });

  test("reads user ID environment settings", () => {
    process.env["OPENCODE_USER_ID_ENABLED"] = "false";
    process.env["OPENCODE_USER_ID_ENDPOINT"] =
      "https://identity.example.com/query";
    process.env["OPENCODE_USER_ID_X-Blackbox-Auth"] = "secret";
    process.env["OPENCODE_USER_ID_TIMEOUT"] = "5000";
    process.env["OPENCODE_USER_ID_RETRY_COUNT"] = "4";
    process.env["OPENCODE_USER_ID_COOLDOWN"] = "0";
    process.env["OPENCODE_USER_ID_TRACESTATE_ENABLED"] = "false";
    process.env["OPENCODE_USER_ID_TRACESTATE_KEY"] = "acme_user";
    const cfg = loadConfig();
    expect(cfg.userIDEnabled).toBe(false);
    expect(cfg.userIDEndpoint).toBe("https://identity.example.com/query");
    expect(cfg.userIDAuthHeader).toBe("secret");
    expect(cfg.userIDTimeout).toBe(5000);
    expect(cfg.userIDRetryCount).toBe(4);
    expect(cfg.userIDCooldown).toBe(0);
    expect(cfg.userIDTracestateEnabled).toBe(false);
    expect(cfg.userIDTracestateKey).toBe("acme_user");
  });

  test("rejects user ID retry counts above the maximum", () => {
    process.env["OPENCODE_USER_ID_RETRY_COUNT"] = "11";
    expect(loadConfig().userIDRetryCount).toBe(2);
  });

  test("options override environment values", () => {
    process.env["OPENCODE_OTLP_ENDPOINT"] = "http://from-env:4317";
    process.env["OPENCODE_TRACE_PREFIX"] = "env.";
    const cfg = loadConfig({
      enabled: true,
      endpoint: "http://from-option:4317",
      protocol: "http/json",
      tracePrefix: "option.",
      tracePropagationProviders: ["gateway"],
      spanAttributeCountLimit: 2048,
      userIDEnabled: false,
      userIDRetryCount: 0,
      userIDCooldown: 0,
      userIDTracestateEnabled: false,
      userIDTracestateKey: "option_user",
    });
    expect(cfg.enabled).toBe(true);
    expect(cfg.endpoint).toBe("http://from-option:4317");
    expect(cfg.protocol).toBe("http/json");
    expect(cfg.tracePrefix).toBe("option.");
    expect(cfg.tracePropagationProviders).toEqual(new Set(["gateway"]));
    expect(cfg.spanAttributeCountLimit).toBe(2048);
    expect(cfg.userIDEnabled).toBe(false);
    expect(cfg.userIDRetryCount).toBe(0);
    expect(cfg.userIDCooldown).toBe(0);
    expect(cfg.userIDTracestateEnabled).toBe(false);
    expect(cfg.userIDTracestateKey).toBe("option_user");
  });

  test("invalid options fall back to environment values", () => {
    process.env["OPENCODE_OTLP_PROTOCOL"] = "http/protobuf";
    process.env["OPENCODE_SPAN_ATTRIBUTE_COUNT_LIMIT"] = "2048";
    const cfg = loadConfig({
      protocol: "udp" as never,
      spanAttributeCountLimit: 0,
    });
    expect(cfg.protocol).toBe("http/protobuf");
    expect(cfg.spanAttributeCountLimit).toBe(2048);
  });

  test("tolerates null options", () => {
    expect(() =>
      loadConfig(null as unknown as Parameters<typeof loadConfig>[0])
    ).not.toThrow();
  });
});

describe("resolveHelperPath", () => {
  test("replaces project placeholders", () => {
    expect(
      resolveHelperPath("${PROJECT_ROOT}/helper", "/directory", "/worktree")
    ).toBe("/worktree/helper");
    expect(
      resolveHelperPath("${WORKTREE}/helper", "/directory", undefined)
    ).toBe("/directory/helper");
    expect(
      resolveHelperPath("${DIRECTORY}/helper", "/directory", "/worktree")
    ).toBe("/directory/helper");
  });

  test("returns undefined unchanged", () => {
    expect(
      resolveHelperPath(undefined, "/directory", "/worktree")
    ).toBeUndefined();
  });
});

describe("resolveLogLevel", () => {
  test("normalizes known levels", () => {
    expect(resolveLogLevel("DEBUG", "info")).toBe("debug");
    expect(resolveLogLevel("WARN", "info")).toBe("warn");
    expect(resolveLogLevel("ERROR", "info")).toBe("error");
  });

  test("keeps the current level for unknown values", () => {
    expect(resolveLogLevel("verbose", "info")).toBe("info");
    expect(resolveLogLevel("", "warn")).toBe("warn");
  });
});
