import { describe, expect, test } from "bun:test";
import { oneSpan, requireSpans, requireSuccess } from "./support/assertions.ts";
import { e2eAvailable, withE2EFixture } from "./support/fixture.ts";

const suite = e2eAvailable ? describe : describe.skip;

suite("OpenCode trace propagation E2E", () => {
  test("propagates W3C context to the trace and model request", () => {
    const traceId = "0af7651916cd43dd8448eb211c80319c";
    const remoteSpanId = "b7ad6b7169203331";
    return withE2EFixture(
      {
        caseID: "w3c-propagation",
        replies: [{ type: "text", text: "context propagated" }],
        pluginOptions: {
          traceparent: `00-${traceId}-${remoteSpanId}-01`,
          tracestate: "vendor=value",
          tracePropagationProviders: ["test"],
        },
      },
      async (fixture) => {
        const result = await fixture.run("propagate context");
        requireSuccess(result);
        expect(fixture.llm.pending()).toBe(0);
        expect(fixture.otlp.errors).toEqual([]);
        const spans = await requireSpans(fixture, result, 3);

        const run = oneSpan(spans, "e2e.run");
        const interaction = oneSpan(spans, "e2e.interaction");
        const llm = oneSpan(spans, "e2e.llm");
        expect(run.traceId).toBe(traceId);
        expect(run.parentSpanId).toBe(remoteSpanId);
        expect(interaction.traceId).toBe(traceId);
        expect(interaction.parentSpanId).toBe(run.spanId);
        expect(llm.traceId).toBe(traceId);
        expect(llm.parentSpanId).toBe(interaction.spanId);

        const hit = fixture.llm.mainHits()[0];
        expect(hit).toBeDefined();
        expect(hit!.headers.get("traceparent")).toBe(
          `00-${traceId}-${llm.spanId}-01`
        );
        expect(hit!.headers.get("tracestate")).toBe("vendor=value");
      }
    );
  }, 60_000);
});
