import { expect } from "bun:test";
import type { E2EFixture } from "./fixture.ts";
import type { RunResult } from "./opencode-process.ts";
import type { ExportedSpan } from "./otlp-receiver.ts";

export function requireSuccess(result: RunResult) {
  if (result.exitCode !== 0) {
    throw new Error(
      `OpenCode exited ${result.exitCode}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`
    );
  }
}

export function spansNamed(spans: ExportedSpan[], name: string) {
  return spans.filter((span) => span.name === name);
}

export function oneSpan(spans: ExportedSpan[], name: string) {
  const matches = spansNamed(spans, name);
  expect(matches).toHaveLength(1);
  return matches[0]!;
}

export function expectOk(span: ExportedSpan) {
  const code = span.status["code"];
  expect(code === 1 || code === "STATUS_CODE_OK").toBe(true);
}

export function expectError(span: ExportedSpan) {
  const code = span.status["code"];
  expect(code === 2 || code === "STATUS_CODE_ERROR").toBe(true);
}

export async function requireSpans(
  fixture: E2EFixture,
  result: RunResult,
  count: number
) {
  const spans = await fixture.otlp.waitForSpanCount(count);
  if (process.env["OPENCODE_E2E_PRINT_SPANS"]) {
    console.log(`[e2e] exported spans:\n${JSON.stringify(spans, null, 2)}`);
  }
  if (spans.length !== count) {
    throw new Error(
      [
        `Expected ${count} spans, received ${spans.length}`,
        `stdout:\n${result.stdout}`,
        `stderr:\n${result.stderr}`,
        `llm hits:\n${JSON.stringify(
          fixture.llm.hits.map((hit) => hit.body),
          null,
          2
        )}`,
        `otlp errors:\n${JSON.stringify(fixture.otlp.errors)}`,
        `otlp payloads:\n${JSON.stringify(fixture.otlp.payloads, null, 2)}`,
      ].join("\n")
    );
  }
  return spans;
}
