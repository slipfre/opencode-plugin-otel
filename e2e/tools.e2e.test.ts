import { describe, expect, test } from "bun:test";
import {
  expectOk,
  oneSpan,
  requireSpans,
  requireSuccess,
  spansNamed,
} from "./support/assertions.ts";
import { e2eAvailable, withE2EFixture } from "./support/fixture.ts";

const suite = e2eAvailable ? describe : describe.skip;

suite("OpenCode tools E2E", () => {
  test(
    "exports tool execution and continuation spans",
    () =>
      withE2EFixture(
        {
          caseID: "tool-continuation",
          replies: [
            {
              type: "tool",
              name: "bash",
              input: {
                command: "echo e2e-tool-output",
                description: "Print deterministic output",
              },
              usage: { input: 5, output: 2 },
            },
            {
              type: "text",
              text: "tool completed",
              usage: { input: 4, output: 3 },
            },
          ],
        },
        async (fixture) => {
          const result = await fixture.run("use the bash tool", [
            "--dangerously-skip-permissions",
          ]);
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.llm.mainHits()).toHaveLength(2);
          const spans = await requireSpans(fixture, result, 5);

          const run = oneSpan(spans, "e2e.run");
          const interaction = oneSpan(spans, "e2e.interaction");
          const tool = oneSpan(spans, "e2e.tool.bash");
          const llms = spansNamed(spans, "e2e.llm");
          expect(llms).toHaveLength(2);
          expect(interaction.parentSpanId).toBe(run.spanId);
          expect(tool.parentSpanId).toBe(interaction.spanId);
          expect(
            llms.every((span) => span.parentSpanId === interaction.spanId)
          ).toBe(true);
          expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);
          expect(tool.attributes["tool.name"]).toBe("bash");
          expect(tool.attributes["tool.success"]).toBe(true);
          expect(String(tool.attributes["input.value"])).toContain(
            "echo e2e-tool-output"
          );
          expect(String(tool.attributes["output.value"])).toContain(
            "e2e-tool-output"
          );
          expect(run.attributes["run.total_tokens"]).toBe(14);
          expect(run.attributes["run.total_messages"]).toBe(2);
          expect(String(run.attributes["output.value"])).toContain(
            "tool completed"
          );
          expectOk(run);
          expectOk(interaction);
          expectOk(tool);
          llms.forEach(expectOk);
        }
      ),
    60_000
  );

  test(
    "exports a foreground subtask trace with parent-child correlation",
    () =>
      withE2EFixture(
        {
          caseID: "foreground-subtask",
          replies: [
            {
              type: "tool",
              name: "task",
              input: {
                description: "Check delegated result",
                prompt: "Return the exact text: subtask completed",
                subagent_type: "explore",
              },
              usage: { input: 6, output: 2 },
            },
            {
              type: "text",
              text: "subtask completed",
              usage: { input: 4, output: 3 },
            },
            {
              type: "text",
              text: "parent received subtask",
              usage: { input: 5, output: 2 },
            },
          ],
        },
        async (fixture) => {
          const result = await fixture.run("delegate a subtask", [
            "--dangerously-skip-permissions",
          ]);
          requireSuccess(result);
          expect(fixture.llm.pending()).toBe(0);
          expect(fixture.llm.mainHits()).toHaveLength(3);
          const spans = await requireSpans(fixture, result, 8);

          const runs = spansNamed(spans, "e2e.run");
          const interactions = spansNamed(spans, "e2e.interaction");
          const llms = spansNamed(spans, "e2e.llm");
          const task = oneSpan(spans, "e2e.tool.task");
          expect(runs).toHaveLength(2);
          expect(interactions).toHaveLength(2);
          expect(llms).toHaveLength(3);

          const parentRun = runs.find(
            (span) => span.attributes["session.is_subagent"] === false
          );
          const childRun = runs.find(
            (span) => span.attributes["session.is_subagent"] === true
          );
          const parentInteraction = interactions.find(
            (span) => span.attributes["session.is_subagent"] === false
          );
          const childInteraction = interactions.find(
            (span) => span.attributes["session.is_subagent"] === true
          );
          expect(parentRun).toBeDefined();
          expect(childRun).toBeDefined();
          expect(parentInteraction).toBeDefined();
          expect(childInteraction).toBeDefined();

          expect(parentInteraction!.parentSpanId).toBe(parentRun!.spanId);
          expect(task.parentSpanId).toBe(parentInteraction!.spanId);
          expect(childRun!.parentSpanId).toBe(task.spanId);
          expect(childInteraction!.parentSpanId).toBe(childRun!.spanId);
          expect(
            llms.filter(
              (span) => span.parentSpanId === parentInteraction!.spanId
            )
          ).toHaveLength(2);
          expect(
            llms.filter(
              (span) => span.parentSpanId === childInteraction!.spanId
            )
          ).toHaveLength(1);
          expect(new Set(spans.map((span) => span.traceId)).size).toBe(1);

          expect(task.attributes["tool.name"]).toBe("task");
          expect(task.attributes["subagent.session.id"]).toBe(
            childRun!.attributes["session.id"]
          );
          expect(task.attributes["subagent.agent.name"]).toBe("explore");
          expect(childRun!.attributes["agent.name"]).toBe("explore");
          expect(childRun!.attributes["session.parent_id"]).toBe(
            parentRun!.attributes["session.id"]
          );
          expect(childRun!.attributes["task.call_id"]).toBe(
            task.attributes["tool.id"]
          );
          expect(childInteraction!.attributes["output.value"]).toBe(
            "subtask completed"
          );
          expect(parentInteraction!.attributes["output.value"]).toBe(
            "parent received subtask"
          );
          expect(parentRun!.attributes["run.total_tokens"]).toBe(15);
          expect(childRun!.attributes["run.total_tokens"]).toBe(7);
          spans.forEach(expectOk);
        }
      ),
    60_000
  );
});
