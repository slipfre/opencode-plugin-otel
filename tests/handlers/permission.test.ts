import { describe, expect, test } from "bun:test";
import { SpanKind, SpanStatusCode } from "@opentelemetry/api";
import {
  AGENT_NAME,
  OpenInferenceSpanKind,
  SemanticConventions,
  SESSION_ID,
  TOOL_ID,
  TOOL_NAME,
} from "@arizeai/openinference-semantic-conventions";
import type {
  EventMessagePartUpdated,
  EventSessionIdle,
} from "@opencode-ai/sdk";
import type {
  EventPermissionAsked,
  EventPermissionReplied,
} from "@opencode-ai/sdk/v2";
import { permissionHandlers } from "../../src/handlers/permission.ts";
import { handleMessagePartUpdated } from "../../src/handlers/message.ts";
import { handleSessionIdle } from "../../src/handlers/session.ts";
import { handleInteractionStarted } from "../../src/interaction.ts";
import { MAX_PENDING } from "../../src/types.ts";
import { makeCtx } from "../helpers.ts";

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND;

function makeToolRunning(
  sessionID = "ses_1",
  callID = "call_1"
): EventMessagePartUpdated {
  return {
    type: "message.part.updated",
    properties: {
      part: {
        type: "tool",
        sessionID,
        messageID: "msg_1",
        callID,
        tool: "bash",
        state: {
          status: "running",
          input: { command: "pwd" },
          time: { start: 1000 },
        },
      },
    },
  } as unknown as EventMessagePartUpdated;
}

function makeToolCompleted(
  sessionID = "ses_1",
  callID = "call_1"
): EventMessagePartUpdated {
  return {
    type: "message.part.updated",
    properties: {
      part: {
        type: "tool",
        sessionID,
        messageID: "msg_1",
        callID,
        tool: "bash",
        state: {
          status: "completed",
          input: { command: "pwd" },
          time: { start: 1000, end: 2000 },
          output: "ok",
          title: "done",
          metadata: {},
        },
      },
    },
  } as unknown as EventMessagePartUpdated;
}

function makeToolError(
  sessionID = "ses_1",
  callID = "call_1"
): EventMessagePartUpdated {
  return {
    type: "message.part.updated",
    properties: {
      part: {
        type: "tool",
        sessionID,
        messageID: "msg_1",
        callID,
        tool: "bash",
        state: {
          status: "error",
          input: { command: "pwd" },
          time: { start: 1000, end: 2000 },
          error: "The user rejected permission to use this specific tool call.",
          metadata: {},
        },
      },
    },
  } as unknown as EventMessagePartUpdated;
}

function makePermissionAsked(
  overrides: {
    id?: string;
    sessionID?: string;
    permission?: string;
    patterns?: string[];
    tool?: { messageID: string; callID: string } | null;
  } = {}
): EventPermissionAsked {
  return {
    type: "permission.asked",
    properties: {
      id: overrides.id ?? "per_1",
      sessionID: overrides.sessionID ?? "ses_1",
      permission: overrides.permission ?? "bash",
      patterns: overrides.patterns ?? ["pwd"],
      metadata: {},
      always: ["pwd"],
      ...(overrides.tool === null
        ? {}
        : { tool: overrides.tool ?? { messageID: "msg_1", callID: "call_1" } }),
    },
  } as EventPermissionAsked;
}

function makePermissionReplied(
  reply: "once" | "always" | "reject" = "once",
  overrides: { sessionID?: string; requestID?: string } = {}
): EventPermissionReplied {
  return {
    type: "permission.replied",
    properties: {
      sessionID: overrides.sessionID ?? "ses_1",
      requestID: overrides.requestID ?? "per_1",
      reply,
    },
  } as EventPermissionReplied;
}

function makeSessionIdle(sessionID = "ses_1"): EventSessionIdle {
  return {
    type: "session.idle",
    properties: { sessionID },
  } as EventSessionIdle;
}

describe("permission spans", () => {
  test("creates a guardrail child span for the matching active tool", async () => {
    const { ctx, tracer } = makeCtx("proj_test", {
      "deployment.environment": "test",
    });
    handleInteractionStarted(
      "user_1",
      "ses_1",
      "build",
      "prompt",
      "anthropic/claude",
      900,
      ctx
    );
    handleMessagePartUpdated(makeToolRunning(), ctx);
    const tool = tracer.spans.at(-1)!;

    await permissionHandlers.asked(makePermissionAsked(), ctx);

    const permission = tracer.spans.at(-1)!;
    expect(permission.name).toBe("opencode.permission.check");
    expect(permission.kind).toBe(SpanKind.INTERNAL);
    expect(permission.parentSpan).toBe(tool);
    expect(permission.attributes[OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.GUARDRAIL
    );
    expect(permission.attributes[SESSION_ID]).toBe("ses_1");
    expect(permission.attributes[TOOL_ID]).toBe("call_1");
    expect(permission.attributes[TOOL_NAME]).toBe("bash");
    expect(permission.attributes[AGENT_NAME]).toBe("build");
    expect(permission.attributes["agent.type"]).toBe("primary");
    expect(permission.attributes["permission.request.id"]).toBe("per_1");
    expect(permission.attributes["permission.name"]).toBe("bash");
    expect(permission.attributes["permission.patterns"]).toEqual(["pwd"]);
    expect(permission.attributes["permission.tool.call_id"]).toBe("call_1");
    expect(permission.attributes["permission.tool.message_id"]).toBe("msg_1");
    expect(permission.attributes["deployment.environment"]).toBe("test");
    expect(ctx.pendingPermissionSpans.has("ses_1:per_1")).toBe(true);
  });

  test("ends a granted permission span with its reply and wait duration", async () => {
    const { ctx, tracer } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);
    await permissionHandlers.asked(makePermissionAsked(), ctx);

    await permissionHandlers.replied(makePermissionReplied("always"), ctx);

    const permission = tracer.spans.at(-1)!;
    expect(permission.ended).toBe(true);
    expect(permission.status.code).toBe(SpanStatusCode.OK);
    expect(permission.attributes["permission.reply"]).toBe("always");
    expect(permission.attributes["permission.granted"]).toBe(true);
    expect(permission.attributes["permission.wait_ms"]).toBeGreaterThanOrEqual(
      0
    );
    expect(ctx.pendingPermissionSpans.size).toBe(0);
  });

  test("records rejection as a completed check rather than a span error", async () => {
    const { ctx, tracer } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);
    const tool = tracer.spans.at(-1)!;
    await permissionHandlers.asked(makePermissionAsked(), ctx);

    await permissionHandlers.replied(makePermissionReplied("reject"), ctx);

    const permission = tracer.spans.at(-1)!;
    expect(permission.status.code).toBe(SpanStatusCode.OK);
    expect(permission.attributes["permission.reply"]).toBe("reject");
    expect(permission.attributes["permission.granted"]).toBe(false);
    expect(tool.attributes["error.type"]).toBeUndefined();

    handleMessagePartUpdated(makeToolError(), ctx);

    expect(tool.attributes["error.type"]).toBe("PermissionRejectedError");
  });

  test("does not set an error type on the tool when permission is granted", async () => {
    const { ctx, tracer } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);
    const tool = tracer.spans.at(-1)!;
    await permissionHandlers.asked(makePermissionAsked(), ctx);

    await permissionHandlers.replied(makePermissionReplied("once"), ctx);
    handleMessagePartUpdated(makeToolError(), ctx);

    expect(tool.attributes["error.type"]).toBeUndefined();
  });

  test("does not fall back to an interaction when the request has no tool", async () => {
    const { ctx, tracer, pluginLog } = makeCtx();
    handleInteractionStarted(
      "user_1",
      "ses_1",
      "build",
      "prompt",
      "anthropic/claude",
      900,
      ctx
    );

    await permissionHandlers.asked(
      makePermissionAsked({ permission: "doom_loop", tool: null }),
      ctx
    );

    expect(tracer.spans.map((span) => span.name)).toEqual([
      "opencode.run",
      "opencode.interaction",
    ]);
    expect(ctx.pendingPermissionSpans.size).toBe(0);
    expect(pluginLog.calls.at(-1)).toMatchObject({
      level: "debug",
      message: "otel: permission span skipped without tool correlation",
    });
  });

  test("does not create a span when the referenced tool span is missing", async () => {
    const { ctx, tracer, pluginLog } = makeCtx();

    await permissionHandlers.asked(makePermissionAsked(), ctx);

    expect(tracer.spans).toHaveLength(0);
    expect(ctx.pendingPermissionSpans.size).toBe(0);
    expect(pluginLog.calls.at(-1)).toMatchObject({
      level: "warn",
      message: "otel: permission span skipped because tool span was not found",
      extra: {
        sessionID: "ses_1",
        requestID: "per_1",
        callID: "call_1",
        permission: "bash",
      },
    });
  });

  test("ignores duplicate requests without creating another span", async () => {
    const { ctx, tracer, pluginLog } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);
    await permissionHandlers.asked(makePermissionAsked(), ctx);

    await permissionHandlers.asked(makePermissionAsked(), ctx);

    expect(
      tracer.spans.filter((span) => span.name === "opencode.permission.check")
    ).toHaveLength(1);
    expect(pluginLog.calls.at(-1)).toMatchObject({
      level: "debug",
      message: "otel: duplicate permission request ignored",
    });
  });

  test("logs an unmatched reply without synthesizing a span", async () => {
    const { ctx, tracer, pluginLog } = makeCtx();

    await permissionHandlers.replied(makePermissionReplied(), ctx);

    expect(tracer.spans).toHaveLength(0);
    expect(pluginLog.calls.at(-1)).toMatchObject({
      level: "debug",
      message: "otel: permission reply has no pending span",
    });
  });

  test("ends an unanswered permission before sweeping its parent tool", async () => {
    const { ctx, tracer } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);
    await permissionHandlers.asked(makePermissionAsked(), ctx);
    const tool = tracer.spans[0]!;
    const permission = tracer.spans[1]!;

    handleSessionIdle(makeSessionIdle(), ctx);

    expect(permission.ended).toBe(true);
    expect(permission.status).toEqual({
      code: SpanStatusCode.ERROR,
      message: "session ended before permission reply",
    });
    expect(tool.ended).toBe(true);
    expect(tool.status.code).toBe(SpanStatusCode.ERROR);
    expect(ctx.pendingPermissionSpans.size).toBe(0);
  });

  test("ends an unanswered permission before its tool completes", async () => {
    const { ctx, tracer } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);
    await permissionHandlers.asked(makePermissionAsked(), ctx);
    const tool = tracer.spans[0]!;
    const permission = tracer.spans[1]!;

    handleMessagePartUpdated(makeToolCompleted(), ctx);

    expect(permission.ended).toBe(true);
    expect(permission.status).toEqual({
      code: SpanStatusCode.ERROR,
      message: "tool ended before permission reply",
    });
    expect(tool.ended).toBe(true);
    expect(tool.status.code).toBe(SpanStatusCode.OK);
    expect(ctx.pendingPermissionSpans.size).toBe(0);
  });

  test("ends the oldest active span when the correlation map reaches capacity", async () => {
    const { ctx, tracer, pluginLog } = makeCtx();
    handleMessagePartUpdated(makeToolRunning(), ctx);

    for (let index = 0; index <= MAX_PENDING; index += 1) {
      await permissionHandlers.asked(
        makePermissionAsked({ id: `per_${index}` }),
        ctx
      );
    }

    const permissions = tracer.spans.filter(
      (span) => span.name === "opencode.permission.check"
    );
    expect(permissions).toHaveLength(MAX_PENDING + 1);
    expect(permissions[0]!.ended).toBe(true);
    expect(permissions[0]!.status.code).toBe(SpanStatusCode.ERROR);
    expect(ctx.pendingPermissionSpans.size).toBe(MAX_PENDING);
    expect(pluginLog.calls.at(-1)).toMatchObject({
      level: "warn",
      message: "otel: pending permission span evicted",
    });
  });
});
