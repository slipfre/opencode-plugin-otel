# opencode-plugin-otel

[![npm version](https://img.shields.io/npm/v/@devtheops/opencode-plugin-otel.svg)](https://www.npmjs.com/package/@devtheops/opencode-plugin-otel)
[![npm downloads](https://img.shields.io/npm/dm/@devtheops/opencode-plugin-otel.svg)](https://www.npmjs.com/package/@devtheops/opencode-plugin-otel)
[![GitHub stars](https://img.shields.io/github/stars/DEVtheOPS/opencode-plugin-otel.svg)](https://github.com/DEVtheOPS/opencode-plugin-otel/stargazers)
[![Build status](https://img.shields.io/github/actions/workflow/status/DEVtheOPS/opencode-plugin-otel/release-please.yml?branch=main)](https://github.com/DEVtheOPS/opencode-plugin-otel/actions/workflows/release-please.yml)
[![Discord notifications](https://img.shields.io/badge/discord-notifications-5865F2?logo=discord&logoColor=white)](https://discord.gg/zavuskz8xB)
[![License](https://img.shields.io/npm/l/@devtheops/opencode-plugin-otel.svg)](https://github.com/DEVtheOPS/opencode-plugin-otel/blob/main/LICENSE)

An [opencode](https://opencode.ai) plugin that exports traces through OpenTelemetry using OTLP over gRPC, HTTP/protobuf, or HTTP/JSON.

- [Trace spans](#trace-spans)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Plugin options](#plugin-options)
  - [Quick start](#quick-start)
  - [Headers and attributes](#headers-and-attributes)
  - [Dynamic headers](#dynamic-headers)
  - [LLM trace propagation](#llm-trace-propagation)
- [Local development](#local-development)

## Trace spans

The trace hierarchy separates one serialized opencode execution from the user messages incorporated into it:

```text
opencode.run (CHAIN, one session busy-to-idle execution)
├── opencode.interaction (AGENT, one user message)
│   ├── opencode.llm
│   ├── opencode.compaction (CHAIN)
│   │   └── opencode.llm (compaction summary)
│   ├── opencode.llm (post-compaction continuation)
│   └── opencode.tool.<name>
│       └── opencode.permission.check (GUARDRAIL)
└── opencode.interaction (AGENT, a queued user message in the same run)
```

A new `opencode.run` starts when a session begins working and ends on `session.idle` or a terminal `session.error`. A recoverable context-overflow error keeps the run open for compaction and continuation. Another prompt submitted while that session is working creates another `opencode.interaction` under the existing run. Different sessions can have independent runs at the same time.

Automatic compaction remains inside the interaction that triggered it. The summary LLM is a child of the `opencode.compaction` span, while post-compaction LLM and tool spans return to the originating interaction. Provider context overflow is recorded as an errored LLM attempt followed by a successful overflow compaction in the same interaction. Manual compaction started without an active interaction is parented directly to a new run.

Run spans include `opencode.run.id`, use the OpenInference `CHAIN` kind, expose interaction inputs as a JSON array, and use the final interaction output as the run output. Interaction spans include `opencode.interaction.id` and use the OpenInference `AGENT` kind. Compaction spans use the OpenInference `CHAIN` kind and include `opencode.compaction.id`, `opencode.compaction.auto`, and `opencode.compaction.overflow`. Overflow compactions also include `opencode.compaction.trigger_message.id`. Their summary LLM spans include `opencode.llm.purpose=compaction`. LLM and tool spans include model, token, cost, input, output, status, and timing attributes when available. Manual permission checks create OpenInference `GUARDRAIL` spans only when the request identifies an active tool span. They include the request, permission, tool, reply, grant result, and wait duration; uncorrelated checks are logged and do not fall back to interaction or run parents.

## Installation

Add the plugin to your opencode config at `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@devtheops/opencode-plugin-otel"]
}
```

For local development, point directly at the TypeScript entrypoint:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["/path/to/opencode-plugin-otel/src/index.ts"]
}
```

## Configuration

The plugin reads settings from `OPENCODE_*` environment variables and inline plugin options. An option takes precedence over the matching environment variable, which takes precedence over the built-in default.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCODE_ENABLE_TELEMETRY` | *(unset)* | Set to any non-empty value to enable trace export |
| `OPENCODE_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint with a URL scheme. HTTP transports append `/v1/traces` |
| `OPENCODE_OTLP_PROTOCOL` | `grpc` | `grpc`, `http/protobuf`, or `http/json` |
| `OPENCODE_TRACE_PREFIX` | `opencode.` | Prefix applied to emitted span names |
| `OPENCODE_OTLP_HEADERS` | *(unset)* | Comma-separated `key=value` headers added to exports |
| `OPENCODE_OTLP_HEADERS_HELPER` | *(unset)* | Executable that returns dynamic OTLP headers as JSON |
| `OPENCODE_RESOURCE_ATTRIBUTES` | *(unset)* | Comma-separated attributes merged into the OTel resource |
| `OPENCODE_SPAN_ATTRIBUTES` | *(unset)* | Comma-separated attributes attached to every span |
| `OPENCODE_SPAN_ATTRIBUTE_COUNT_LIMIT` | `4096` | Maximum attributes retained per span |
| `OPENCODE_TRACEPARENT` | *(unset)* | W3C `traceparent` used as the remote parent for all spans |
| `OPENCODE_TRACESTATE` | *(unset)* | W3C `tracestate` paired with `OPENCODE_TRACEPARENT` |
| `OPENCODE_TRACE_PROPAGATION_PROVIDERS` | *(unset)* | Provider IDs that receive W3C trace context on LLM requests; `*` enables all providers |
| `OPENCODE_USER_ID_ENABLED` | `true` | Enables resolving `user.id` from the first configured provider API key |
| `OPENCODE_USER_ID_ENDPOINT` | `queryUserByToken` | Endpoint used to resolve `user.id` |
| `OPENCODE_USER_ID-X-Blackbox-Auth` | *(unset)* | Value sent in the `X-Blackbox-Auth` header for user ID lookup |
| `OPENCODE_USER_ID_TIMEOUT` | `3000` | User ID request timeout in milliseconds |
| `OPENCODE_USER_ID_RETRY_COUNT` | `2` | Retries after the initial user ID request fails, from `0` to `10` |
| `OPENCODE_USER_ID_COOLDOWN` | `300000` | Cooldown after all user ID attempts fail; `0` disables the cooldown |

### Plugin options

Every setting can also be passed through opencode's plugin tuple form:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    ["@devtheops/opencode-plugin-otel", {
      "enabled": true,
      "endpoint": "http://localhost:4317",
      "protocol": "grpc",
      "tracePrefix": "opencode.",
      "resourceAttributes": "service.version=1.2.3,deployment.environment=production"
    }]
  ]
}
```

Option keys mirror the resolved config:

| Option | Environment variable |
|--------|----------------------|
| `enabled` | `OPENCODE_ENABLE_TELEMETRY` |
| `endpoint` | `OPENCODE_OTLP_ENDPOINT` |
| `protocol` | `OPENCODE_OTLP_PROTOCOL` |
| `tracePrefix` | `OPENCODE_TRACE_PREFIX` |
| `otlpHeaders` | `OPENCODE_OTLP_HEADERS` |
| `otlpHeadersHelper` | `OPENCODE_OTLP_HEADERS_HELPER` |
| `resourceAttributes` | `OPENCODE_RESOURCE_ATTRIBUTES` |
| `spanAttributes` | `OPENCODE_SPAN_ATTRIBUTES` |
| `spanAttributeCountLimit` | `OPENCODE_SPAN_ATTRIBUTE_COUNT_LIMIT` |
| `traceparent` | `OPENCODE_TRACEPARENT` |
| `tracestate` | `OPENCODE_TRACESTATE` |
| `tracePropagationProviders` | `OPENCODE_TRACE_PROPAGATION_PROVIDERS` |
| `userIDEnabled` | `OPENCODE_USER_ID_ENABLED` |
| `userIDEndpoint` | `OPENCODE_USER_ID_ENDPOINT` |
| `userIDAuthHeader` | `OPENCODE_USER_ID-X-Blackbox-Auth` |
| `userIDTimeout` | `OPENCODE_USER_ID_TIMEOUT` |
| `userIDRetryCount` | `OPENCODE_USER_ID_RETRY_COUNT` |
| `userIDCooldown` | `OPENCODE_USER_ID_COOLDOWN` |

Keep secrets such as `otlpHeaders` out of committed configuration. Prefer an environment variable or opencode `{env:VAR}` substitution.

### Quick start

```bash
export OPENCODE_ENABLE_TELEMETRY=1
export OPENCODE_OTLP_ENDPOINT=http://localhost:4317
export OPENCODE_OTLP_PROTOCOL=grpc
opencode
```

For `http/protobuf` and `http/json`, set the collector base URL. The plugin appends `/v1/traces` automatically.

### Headers and attributes

```bash
export OPENCODE_OTLP_HEADERS="Authorization=Bearer <token>"
export OPENCODE_RESOURCE_ATTRIBUTES="service.version=1.2.3,deployment.environment=production"
export OPENCODE_SPAN_ATTRIBUTES="team=platform"
```

`OPENCODE_RESOURCE_ATTRIBUTES` describes the producer resource. `OPENCODE_SPAN_ATTRIBUTES` adds filterable attributes to every emitted span.

### Dynamic headers

Use `OPENCODE_OTLP_HEADERS_HELPER` when the collector requires short-lived credentials. The helper is prewarmed during startup. After an authentication failure, the plugin refreshes the headers, rebuilds the trace exporter, and retries the failed export once.

```bash
export OPENCODE_OTLP_HEADERS_HELPER='${PROJECT_ROOT}/scripts/opencode-otel-headers.sh'
```

The helper must be executable and print a JSON object whose values are strings. `${PROJECT_ROOT}`, `${WORKTREE}`, and `${DIRECTORY}` placeholders are supported.

### LLM trace propagation

Use `OPENCODE_TRACE_PROPAGATION_PROVIDERS` to connect LLM spans to traces emitted by gateways such as LiteLLM or vLLM:

```bash
export OPENCODE_TRACE_PROPAGATION_PROVIDERS="company-litellm,vllm"
```

Only W3C `traceparent` and `tracestate` are injected. Propagation is disabled when the setting is unset.

## Local development

See [CONTRIBUTING.md](./CONTRIBUTING.md).
