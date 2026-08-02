# AGENTS.md

Instructions for AI agents working in this repository.

## Build & typecheck

Always run after making changes:

```bash
bun run typecheck
```

There is no build step. TypeScript source files are published directly and loaded natively by Bun.

## Testing

```bash
bun test
```

## Project layout

```text
src/
├── index.ts              — Plugin entrypoint
├── types.ts              — Shared types
├── config.ts             — Env config and log level
├── otel.ts               — OTel trace SDK setup
├── probe.ts              — OTLP endpoint TCP probe
├── util.ts               — errorSummary, setBoundedMap
└── handlers/
    ├── session.ts        — Session lifecycle events
    ├── message.ts        — LLM message and tool part events
    └── chat-headers.ts   — W3C trace-context propagation
```

## Key conventions

- **Bun over Node** — use `bun`, `bun test`, `bun run`. Never use `node`, `npx`, `jest`, or `vitest`.
- **No comments** unless explicitly requested.
- **No `sdk-node`** — the OTel Node SDK meta-package is intentionally excluded; use individual packages.
- **`HandlerContext`** — all event handlers receive a `HandlerContext` (defined in `src/types.ts`). Do not import `client` or OTel globals directly inside handlers; thread them through the context.
- **`setBoundedMap`** — always use this instead of `Map.set` for `pendingToolSpans` and other correlation maps to prevent unbounded growth.
- **Shutdown** — OTel providers are flushed via `SIGTERM`/`SIGINT`/`beforeExit`. Do not use `process.on("exit")` for async flushing.
- **All env vars are `OPENCODE_` prefixed** — `OPENCODE_ENABLE_TELEMETRY`, `OPENCODE_OTLP_ENDPOINT`, `OPENCODE_TRACE_PREFIX`, `OPENCODE_OTLP_HEADERS`, `OPENCODE_RESOURCE_ATTRIBUTES`, and `OPENCODE_SPAN_ATTRIBUTES`. Never use bare `OTEL_*` names for plugin config. `loadConfig` copies `OPENCODE_OTLP_HEADERS` → `OTEL_EXPORTER_OTLP_HEADERS` and `OPENCODE_RESOURCE_ATTRIBUTES` → `OTEL_RESOURCE_ATTRIBUTES` before the SDK initializes.
- **`OPENCODE_ENABLE_TELEMETRY`** — trace instrumentation is gated on this env var. The plugin always loads regardless; tracing is disabled when unset.
- **`OPENCODE_TRACE_PREFIX`** — defaults to `opencode.` and prefixes emitted span names.
- **Plugin options** — `loadConfig` also accepts an `OtelPluginOptions` object passed via opencode's plugin tuple form (`["opencode-plugin-otel", { ... }]`, threaded through `OtelPlugin`'s second argument). Precedence is option → `OPENCODE_*` env → default. Keep option keys 1:1 with `PluginConfig` field names.

## Commit message format

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>[optional scope]: <description>
```

Common types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `chore`, `build`.

Use `!` or a `BREAKING CHANGE:` footer for breaking changes.

Examples:

```text
feat(handlers): add support for file.edited event
fix(probe): handle malformed endpoint URL without throwing
chore(deps): bump @opentelemetry/api to 1.10.0
```
