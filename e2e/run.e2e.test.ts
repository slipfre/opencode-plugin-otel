import { describe, expect, test } from "bun:test";
import {
  LLM_OUTPUT_MESSAGES,
  MESSAGE_CONTENTS,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
} from "@arizeai/openinference-semantic-conventions";
import pkg from "../package.json" with { type: "json" };
import {
  expectOk,
  oneSpan,
  requireSpans,
  requireSuccess,
  spansNamed,
} from "./support/assertions.ts";
import { e2eAvailable, withE2EFixture } from "./support/fixture.ts";

const suite = e2eAvailable ? describe : describe.skip;

suite("OpenCode run E2E", () => {
  test(
    "exports a complete text trace with token and reasoning attributes",
    () =>
      withE2EFixture(
        {
          caseID: "text-trace",
          replies: [
            {
              type: "text",
              text: "hello from e2e",
              reasoning: "reasoning from e2e",
              usage: { input: 11, output: 7, cacheRead: 4, reasoning: 2 },
            },
          ],
        },
        async (fixture) => {
          const result = await fixture.run("say hello");
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.otlp.errors).toEqual([]);
          const spans = await requireSpans(fixture, result, 3);

          const run = oneSpan(spans, "e2e.run");
          const interaction = oneSpan(spans, "e2e.interaction");
          const llm = oneSpan(spans, "e2e.llm");
          expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);
          expect(interaction.parentSpanId).toBe(run.spanId);
          expect(llm.parentSpanId).toBe(interaction.spanId);
          expect(run.attributes["openinference.span.kind"]).toBe("CHAIN");
          expect(interaction.attributes["openinference.span.kind"]).toBe(
            "AGENT"
          );
          expect(llm.attributes["openinference.span.kind"]).toBe("LLM");
          expect(llm.attributes["llm.model_name"]).toBe("test-model");
          expect(llm.attributes["llm.provider"]).toBe("test");
          expect(llm.attributes["llm.token_count.prompt"]).toBe(11);
          expect(
            llm.attributes["llm.token_count.prompt_details.cache_write"]
          ).toBe(0);
          expect(
            llm.attributes["llm.token_count.prompt_details.cache_read"]
          ).toBe(4);
          expect(llm.attributes["llm.token_count.completion"]).toBe(7);
          expect(
            llm.attributes["llm.token_count.completion_details.reasoning"]
          ).toBe(2);
          expect(llm.attributes["llm.token_count.total"]).toBe(18);
          expect(
            llm.attributes[
              `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`
            ]
          ).toBe("reasoning");
          expect(
            llm.attributes[
              `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`
            ]
          ).toBe("reasoning from e2e");
          expect(
            llm.attributes[
              `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`
            ]
          ).toBe("text");
          expect(String(llm.attributes["output.value"])).toContain(
            "hello from e2e"
          );
          expect(run.attributes["run.total_tokens"]).toBe(18);
          expect(run.attributes["run.total_messages"]).toBe(1);
          expect(run.attributes["e2e.case"]).toBe("text-trace");
          expect(run.resource["service.name"]).toBe("opencode");
          expect(run.resource["service.version"]).toBe("local");
          expect(run.scope).toEqual({
            name: "opencode-plugin-otel",
            version: pkg.version,
          });
          expect(run.attributes["opencode.plugin.version"]).toBeUndefined();
          expect(run.resource["opencode.plugin.version"]).toBeUndefined();
          expect(run.resource["app.version"]).toBeUndefined();
          expect(run.resource["e2e.resource"]).toBe("opencode-plugin-otel");
          expectOk(run);
          expectOk(interaction);
          expectOk(llm);
        }
      ),
    60_000
  );

  test(
    "keeps steered user input in one run with distinct interactions",
    () =>
      withE2EFixture(
        {
          caseID: "steer",
          replies: [
            {
              type: "text",
              text: "first response before steer",
              usage: { input: 3, output: 2 },
              hold: true,
            },
            {
              type: "text",
              text: "guided final response",
              usage: { input: 4, output: 3 },
            },
          ],
        },
        async (fixture) => {
          const activeRun = await fixture.startRun("initial request");
          expect(fixture.llm.held()).toBe(1);
          await activeRun.steer("steering guidance");
          const result = await activeRun.finish();
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.llm.mainHits()).toHaveLength(2);
          expect(JSON.stringify(fixture.llm.mainHits()[1]!.body)).toContain(
            "steering guidance"
          );
          expect(fixture.otlp.errors).toEqual([]);
          const spans = await requireSpans(fixture, result, 5);

          const run = oneSpan(spans, "e2e.run");
          const interactions = spansNamed(spans, "e2e.interaction");
          const llms = spansNamed(spans, "e2e.llm");
          expect(interactions).toHaveLength(2);
          expect(llms).toHaveLength(2);
          expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);
          expect(
            interactions.every((span) => span.parentSpanId === run.spanId)
          ).toBe(true);
          expect(
            interactions.every((interaction) =>
              llms.some((llm) => llm.parentSpanId === interaction.spanId)
            )
          ).toBe(true);

          const interactionByInput = new Map(
            interactions.map((span) => [span.attributes["input.value"], span])
          );
          expect(
            interactionByInput.get("initial request")?.attributes[
              "output.value"
            ]
          ).toBe("first response before steer");
          expect(
            interactionByInput.get("steering guidance")?.attributes[
              "output.value"
            ]
          ).toBe("guided final response");
          expect(JSON.parse(String(run.attributes["input.value"]))).toEqual([
            "initial request",
            "steering guidance",
          ]);
          expect(run.attributes["output.value"]).toBe("guided final response");
          expect(run.attributes["run.total_interactions"]).toBe(2);
          expect(run.attributes["run.total_messages"]).toBe(2);
          expect(run.attributes["run.total_tokens"]).toBe(12);
          spans.forEach(expectOk);
        }
      ),
    60_000
  );

  test(
    "loads without exporting when telemetry is disabled",
    () =>
      withE2EFixture(
        {
          caseID: "disabled",
          replies: [{ type: "text", text: "telemetry disabled" }],
          pluginOptions: { enabled: false },
        },
        async (fixture) => {
          const result = await fixture.run("do not trace this");
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.otlp.payloads).toHaveLength(0);
          expect(fixture.otlp.spans()).toHaveLength(0);
        }
      ),
    60_000
  );
});
