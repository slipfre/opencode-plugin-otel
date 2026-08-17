import { describe, expect, test } from "bun:test";
import {
  expectError,
  expectOk,
  oneSpan,
  requireSpans,
  requireSuccess,
  spansNamed,
} from "./support/assertions.ts";
import { e2eAvailable, withE2EFixture } from "./support/fixture.ts";

const suite = e2eAvailable ? describe : describe.skip;

suite("OpenCode compaction E2E", () => {
  test(
    "keeps automatic compaction and continuation in one interaction trace",
    () =>
      withE2EFixture(
        {
          caseID: "automatic-compaction",
          autoCompact: true,
          replies: [
            {
              type: "tool",
              name: "bash",
              input: {
                command: "echo before-compact",
                description: "Trigger compact after a tool step",
              },
              usage: { input: 90_000, output: 2 },
            },
            {
              type: "text",
              text: "durable automatic compact summary",
              usage: { input: 5, output: 3 },
            },
            {
              type: "text",
              text: "continued after automatic compact",
              usage: { input: 4, output: 3 },
            },
          ],
        },
        async (fixture) => {
          const result = await fixture.run(
            "use a tool and continue after compact",
            ["--dangerously-skip-permissions"]
          );
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.llm.mainHits()).toHaveLength(3);
          expect(fixture.otlp.errors).toEqual([]);
          const spans = await requireSpans(fixture, result, 7);

          const run = oneSpan(spans, "e2e.run");
          const interaction = oneSpan(spans, "e2e.interaction");
          const tool = oneSpan(spans, "e2e.tool.bash");
          const compaction = oneSpan(spans, "e2e.compaction");
          const llms = spansNamed(spans, "e2e.llm");
          expect(llms).toHaveLength(3);

          const summary = llms.find(
            (span) => span.attributes["opencode.llm.purpose"] === "compaction"
          );
          const before = llms.find(
            (span) => span.attributes["llm.token_count.prompt"] === 90_000
          );
          const continuation = llms.find(
            (span) => span.attributes["llm.token_count.prompt"] === 4
          );
          expect(summary).toBeDefined();
          expect(before).toBeDefined();
          expect(continuation).toBeDefined();

          expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);
          expect(interaction.parentSpanId).toBe(run.spanId);
          expect(before!.parentSpanId).toBe(interaction.spanId);
          expect(tool.parentSpanId).toBe(interaction.spanId);
          expect(compaction.parentSpanId).toBe(interaction.spanId);
          expect(summary!.parentSpanId).toBe(compaction.spanId);
          expect(continuation!.parentSpanId).toBe(interaction.spanId);
          expect(continuation!.parentSpanId).not.toBe(compaction.spanId);

          expect(compaction.kind).toBe(1);
          expect(compaction.attributes["openinference.span.kind"]).toBe(
            "CHAIN"
          );
          expect(compaction.attributes["opencode.compaction.auto"]).toBe(true);
          expect(compaction.attributes["opencode.interaction.id"]).toBe(
            interaction.attributes["opencode.interaction.id"]
          );
          expect(typeof compaction.attributes["opencode.compaction.id"]).toBe(
            "string"
          );
          expect(summary!.kind).toBe(3);
          expect(summary!.attributes["openinference.span.kind"]).toBe("LLM");
          expect(summary!.attributes["opencode.compaction.id"]).toBe(
            compaction.attributes["opencode.compaction.id"]
          );
          expect(typeof summary!.attributes["opencode.message.id"]).toBe(
            "string"
          );
          expect(before!.attributes["opencode.llm.purpose"]).toBeUndefined();
          expect(
            continuation!.attributes["opencode.llm.purpose"]
          ).toBeUndefined();

          expect(
            BigInt(compaction.startTimeUnixNano) <=
              BigInt(summary!.startTimeUnixNano)
          ).toBe(true);
          expect(
            BigInt(summary!.endTimeUnixNano) <=
              BigInt(compaction.endTimeUnixNano)
          ).toBe(true);
          expect(
            BigInt(compaction.endTimeUnixNano) <=
              BigInt(continuation!.startTimeUnixNano)
          ).toBe(true);
          expect(run.attributes["run.total_messages"]).toBe(3);
          expect(run.attributes["run.total_tokens"]).toBe(90_017);
          expect(run.attributes["output.value"]).toBe(
            "continued after automatic compact"
          );
          expect(interaction.attributes["interaction.total_messages"]).toBe(3);
          expect(interaction.attributes["interaction.total_tokens"]).toBe(
            90_017
          );
          expect(interaction.attributes["output.value"]).toBe(
            "continued after automatic compact"
          );
          spans.forEach(expectOk);
        }
      ),
    60_000
  );

  test("reports the provider error when an automatic compaction summary overflows", () => {
    const summaryError = "Compaction summary exceeded the model context limit";
    return withE2EFixture(
      {
        caseID: "automatic-compaction-summary-overflow",
        autoCompact: true,
        replies: [
          {
            type: "tool",
            name: "bash",
            input: {
              command: "echo before-summary-overflow",
              description: "Trigger compact before summary overflow",
            },
            usage: { input: 90_000, output: 2 },
          },
          {
            type: "error",
            code: "context_length_exceeded",
            message: summaryError,
            status: 400,
          },
        ],
      },
      async (fixture) => {
        const result = await fixture.run(
          "use a tool and compact into an overflowing summary",
          ["--dangerously-skip-permissions"]
        );
        expect(result.exitCode).toBe(1);
        expect(fixture.llm.pending()).toBe(0);
        expect(fixture.llm.mainHits()).toHaveLength(2);
        expect(fixture.otlp.errors).toEqual([]);
        const spans = await requireSpans(fixture, result, 6);

        const run = oneSpan(spans, "e2e.run");
        const interaction = oneSpan(spans, "e2e.interaction");
        const tool = oneSpan(spans, "e2e.tool.bash");
        const compaction = oneSpan(spans, "e2e.compaction");
        const llms = spansNamed(spans, "e2e.llm");
        expect(llms).toHaveLength(2);

        const summary = llms.find(
          (span) => span.attributes["opencode.llm.purpose"] === "compaction"
        );
        const initial = llms.find((span) => span !== summary);
        expect(summary).toBeDefined();
        expect(initial).toBeDefined();

        expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);
        expect(interaction.parentSpanId).toBe(run.spanId);
        expect(initial!.parentSpanId).toBe(interaction.spanId);
        expect(tool.parentSpanId).toBe(interaction.spanId);
        expect(compaction.parentSpanId).toBe(interaction.spanId);
        expect(summary!.parentSpanId).toBe(compaction.spanId);

        expect(compaction.attributes["opencode.compaction.auto"]).toBe(true);
        expect(summary!.attributes["opencode.compaction.id"]).toBe(
          compaction.attributes["opencode.compaction.id"]
        );
        expect(summary!.attributes["opencode.llm.purpose"]).toBe("compaction");
        expect(summary!.attributes["error.type"]).toBe("ContextOverflowError");
        const summaryStatusMessage = String(summary!.status["message"] ?? "");
        expect(summaryStatusMessage).not.toContain(
          "session ended before message completed"
        );
        expect(summaryStatusMessage).toMatch(
          /ContextOverflowError|Compaction summary exceeded the model context limit/
        );

        expectOk(initial!);
        expectOk(tool);
        expectError(summary!);
        expectError(compaction);
        expectError(interaction);
        expectError(run);
      }
    );
  }, 60_000);

  test(
    "recovers a provider context overflow through automatic compaction",
    () =>
      withE2EFixture(
        {
          caseID: "context-overflow-compaction",
          autoCompact: true,
          replies: [
            {
              type: "error",
              code: "context_length_exceeded",
              message: "This model's maximum context length was exceeded",
            },
            {
              type: "text",
              text: "durable overflow compact summary",
              usage: { input: 5, output: 3 },
            },
            {
              type: "text",
              text: "continued after context overflow",
              usage: { input: 4, output: 3 },
            },
          ],
        },
        async (fixture) => {
          const result = await fixture.run("recover after context overflow");
          expect(result.exitCode).toBe(1);
          expect(result.stdout).toContain("continued after context overflow");
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.llm.mainHits()).toHaveLength(3);
          expect(fixture.otlp.errors).toEqual([]);
          const spans = await requireSpans(fixture, result, 6);

          const run = oneSpan(spans, "e2e.run");
          const interaction = oneSpan(spans, "e2e.interaction");
          const compaction = oneSpan(spans, "e2e.compaction");
          const llms = spansNamed(spans, "e2e.llm");
          expect(llms).toHaveLength(3);

          const summary = llms.find(
            (span) => span.attributes["opencode.llm.purpose"] === "compaction"
          );
          const failed = llms.find((span) => {
            const code = span.status["code"];
            return code === 2 || code === "STATUS_CODE_ERROR";
          });
          const continuation = llms.find(
            (span) => span.attributes["llm.token_count.prompt"] === 4
          );
          expect(summary).toBeDefined();
          expect(failed).toBeDefined();
          expect(continuation).toBeDefined();
          expect(new Set([summary, failed, continuation]).size).toBe(3);

          expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);
          expect(interaction.parentSpanId).toBe(run.spanId);
          expect(failed!.parentSpanId).toBe(interaction.spanId);
          expect(compaction.parentSpanId).toBe(interaction.spanId);
          expect(summary!.parentSpanId).toBe(compaction.spanId);
          expect(continuation!.parentSpanId).toBe(interaction.spanId);
          expect(continuation!.parentSpanId).not.toBe(compaction.spanId);

          expect(compaction.attributes["opencode.compaction.auto"]).toBe(true);
          expect(compaction.attributes["opencode.compaction.overflow"]).toBe(
            true
          );
          expect(
            typeof compaction.attributes[
              "opencode.compaction.trigger_message.id"
            ]
          ).toBe("string");
          expect(failed!.attributes["error.type"]).toBe("ContextOverflowError");
          expect(summary!.attributes["opencode.compaction.id"]).toBe(
            compaction.attributes["opencode.compaction.id"]
          );
          expect(summary!.attributes["opencode.llm.purpose"]).toBe(
            "compaction"
          );
          expect(summary!.attributes["opencode.compaction.overflow"]).toBe(
            true
          );
          expect(
            summary!.attributes["opencode.compaction.trigger_message.id"]
          ).toBe(
            compaction.attributes["opencode.compaction.trigger_message.id"]
          );
          expect(
            continuation!.attributes["opencode.llm.purpose"]
          ).toBeUndefined();

          expect(
            BigInt(failed!.endTimeUnixNano) <=
              BigInt(compaction.startTimeUnixNano)
          ).toBe(true);
          expect(
            BigInt(compaction.startTimeUnixNano) <=
              BigInt(summary!.startTimeUnixNano)
          ).toBe(true);
          expect(
            BigInt(summary!.endTimeUnixNano) <=
              BigInt(compaction.endTimeUnixNano)
          ).toBe(true);
          expect(
            BigInt(compaction.endTimeUnixNano) <=
              BigInt(continuation!.startTimeUnixNano)
          ).toBe(true);
          expect(run.attributes["run.total_messages"]).toBe(3);
          expect(run.attributes["run.total_tokens"]).toBe(15);
          expect(run.attributes["output.value"]).toBe(
            "continued after context overflow"
          );
          expect(interaction.attributes["interaction.total_messages"]).toBe(3);
          expect(interaction.attributes["interaction.total_tokens"]).toBe(15);
          expect(interaction.attributes["output.value"]).toBe(
            "continued after context overflow"
          );
          expectError(failed!);
          expectOk(summary!);
          expectOk(continuation!);
          expectOk(compaction);
          expectOk(interaction);
          expectOk(run);
        }
      ),
    60_000
  );

  test(
    "exports manual compaction as a run-scoped wrapper around the summary",
    () =>
      withE2EFixture(
        {
          caseID: "manual-compaction",
          replies: [
            {
              type: "text",
              text: "response before compact",
              usage: { input: 3, output: 2 },
              hold: true,
            },
            {
              type: "text",
              text: "durable compact summary",
              usage: { input: 5, output: 3 },
            },
          ],
        },
        async (fixture) => {
          const activeRun = await fixture.startRun(
            "remember this before compact"
          );
          expect(fixture.llm.held()).toBe(1);
          const result = await activeRun.manualCompact();
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.llm.mainHits()).toHaveLength(2);
          expect(fixture.otlp.errors).toEqual([]);
          const spans = await requireSpans(fixture, result, 6);

          const runs = spansNamed(spans, "e2e.run");
          const interactions = spansNamed(spans, "e2e.interaction");
          const llms = spansNamed(spans, "e2e.llm");
          const compaction = oneSpan(spans, "e2e.compaction");
          expect(runs).toHaveLength(2);
          expect(interactions).toHaveLength(1);
          expect(llms).toHaveLength(2);

          const summary = llms.find(
            (span) => span.attributes["opencode.llm.purpose"] === "compaction"
          );
          const promptLlm = llms.find((span) => span !== summary);
          const compactRun = runs.find(
            (span) => span.traceId === compaction.traceId
          );
          const promptRun = runs.find((span) => span !== compactRun);
          expect(summary).toBeDefined();
          expect(promptLlm).toBeDefined();
          expect(compactRun).toBeDefined();
          expect(promptRun).toBeDefined();

          expect(compaction.parentSpanId).toBe(compactRun!.spanId);
          expect(summary!.parentSpanId).toBe(compaction.spanId);
          expect(summary!.traceId).toBe(compaction.traceId);
          expect(compactRun!.traceId).toBe(compaction.traceId);
          expect(interactions[0]!.parentSpanId).toBe(promptRun!.spanId);
          expect(promptLlm!.parentSpanId).toBe(interactions[0]!.spanId);
          expect(promptLlm!.traceId).toBe(promptRun!.traceId);
          expect(promptRun!.traceId).not.toBe(compactRun!.traceId);

          expect(compaction.attributes["openinference.span.kind"]).toBe(
            "CHAIN"
          );
          expect(compaction.attributes["opencode.compaction.auto"]).toBe(false);
          expect(typeof compaction.attributes["opencode.compaction.id"]).toBe(
            "string"
          );
          expect(summary!.attributes["openinference.span.kind"]).toBe("LLM");
          expect(summary!.attributes["opencode.compaction.id"]).toBe(
            compaction.attributes["opencode.compaction.id"]
          );
          expect(compactRun!.attributes["run.total_messages"]).toBe(1);
          expect(compactRun!.attributes["run.total_tokens"]).toBe(8);
          spans.forEach(expectOk);
        }
      ),
    60_000
  );
});
