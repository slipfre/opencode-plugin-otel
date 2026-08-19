import { describe, expect, test } from "bun:test";
import { createTraceState } from "@opentelemetry/api";
import { handleChatHeaders } from "../src/handlers/chat-headers.ts";
import { makeCtx } from "./helpers.ts";

type ChatHeadersInput = Parameters<typeof handleChatHeaders>[0];

function makeInput(
  overrides: {
    agent?: string;
    modelID?: string;
    providerID?: string;
    messageID?: string;
  } = {}
) {
  const providerID = overrides.providerID ?? "company-litellm";
  return {
    sessionID: "ses_1",
    agent: overrides.agent ?? "build",
    model: {
      id: overrides.modelID ?? "claude",
      providerID,
    } as ChatHeadersInput["model"],
    provider: {
      source: "config",
      info: { id: providerID },
      options: {},
    } as ChatHeadersInput["provider"],
    message: {
      id: overrides.messageID ?? "user_1",
    } as ChatHeadersInput["message"],
  };
}

function seedRequest(
  ctx: ReturnType<typeof makeCtx>["ctx"],
  overrides: { agent?: string; modelID?: string; providerID?: string } = {}
) {
  ctx.llmRequestContexts.set("ses_1:user_1", [
    {
      messageID: "msg_1",
      agent: overrides.agent ?? "build",
      modelID: overrides.modelID ?? "claude",
      providerID: overrides.providerID ?? "company-litellm",
      spanContext: {
        traceId: "0af7651916cd43dd8448eb211c80319c",
        spanId: "b7ad6b7169203331",
        traceFlags: 1,
      },
    },
  ]);
}

describe("handleChatHeaders", () => {
  test("injects traceparent for an enabled provider", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.traceparent).toBe(
      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    );
  });

  test("injects tracestate when present", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    ctx.llmRequestContexts.get("ses_1:user_1")![0]!.spanContext.traceState =
      createTraceState("vendor=value");
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBe("vendor=value");
  });

  test("supports an explicit wildcard", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("*");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.traceparent).toBeDefined();
  });

  test("does not inject for an unconfigured provider", () => {
    const { ctx } = makeCtx();
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers).toEqual({});
  });

  test("does not inject without a matching live request", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("company-litellm");
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers).toEqual({});
  });

  test("does not attach a concurrent title request to the main span", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput({ agent: "title" }), output, ctx);

    expect(output.headers).toEqual({});
  });

  test("selects each concurrently live request by agent, model, and provider", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("*");
    seedRequest(ctx);
    ctx.llmRequestContexts.get("ses_1:user_1")!.push({
      messageID: "msg_2",
      agent: "review",
      modelID: "gpt-5",
      providerID: "company-openai",
      spanContext: {
        traceId: "4bf92f3577b34da6a3ce929d0e0e4736",
        spanId: "00f067aa0ba902b7",
        traceFlags: 1,
      },
    });
    const buildOutput = { headers: {} as Record<string, string> };
    const reviewOutput = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), buildOutput, ctx);
    handleChatHeaders(
      makeInput({
        agent: "review",
        modelID: "gpt-5",
        providerID: "company-openai",
      }),
      reviewOutput,
      ctx
    );

    expect(buildOutput.headers.traceparent).toContain(
      "0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331"
    );
    expect(reviewOutput.headers.traceparent).toContain(
      "4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7"
    );
  });

  test("selects the newest live request when metadata is identical", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    ctx.llmRequestContexts.get("ses_1:user_1")!.push({
      messageID: "msg_2",
      agent: "build",
      modelID: "claude",
      providerID: "company-litellm",
      spanContext: {
        traceId: "4bf92f3577b34da6a3ce929d0e0e4736",
        spanId: "00f067aa0ba902b7",
        traceFlags: 1,
      },
    });
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.traceparent).toBe(
      "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    );
  });

  test("requires matching model and provider metadata", () => {
    const { ctx } = makeCtx();
    ctx.tracePropagationProviders.add("*");
    seedRequest(ctx);
    const modelOutput = { headers: {} as Record<string, string> };
    const providerOutput = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput({ modelID: "other" }), modelOutput, ctx);
    handleChatHeaders(makeInput({ providerID: "other" }), providerOutput, ctx);

    expect(modelOutput.headers).toEqual({});
    expect(providerOutput.headers).toEqual({});
  });
  test("injects the resolved user ID as a tracestate member", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "u_42" });
    ctx.userIDTracestateKey = "opencode_user_id";
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.traceparent).toBe(
      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    );
    expect(output.headers.tracestate).toBe("opencode_user_id=u_42");
  });

  test("merges the user ID into a propagated tracestate", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "u_42" });
    ctx.userIDTracestateKey = "opencode_user_id";
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    ctx.llmRequestContexts.get("ses_1:user_1")![0]!.spanContext.traceState =
      createTraceState("vendor=value");
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBe(
      "opencode_user_id=u_42,vendor=value"
    );
  });

  test("honours a custom tracestate key", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "u_42" });
    ctx.userIDTracestateKey = "acme_user";
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBe("acme_user=u_42");
  });

  test("does not inject the user ID for a provider without trace propagation", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "u_42" });
    ctx.userIDTracestateKey = "opencode_user_id";
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBeUndefined();
  });

  test("does not inject an unresolved user ID", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "unknown" });
    ctx.userIDTracestateKey = "opencode_user_id";
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBeUndefined();
  });

  test("does not inject a user ID with invalid tracestate characters", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "u=42,x" });
    ctx.userIDTracestateKey = "opencode_user_id";
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBeUndefined();
  });

  test("does not inject the user ID when the key is unset", () => {
    const { ctx } = makeCtx("proj_test", { "user.id": "u_42" });
    ctx.tracePropagationProviders.add("company-litellm");
    seedRequest(ctx);
    const output = { headers: {} as Record<string, string> };

    handleChatHeaders(makeInput(), output, ctx);

    expect(output.headers.tracestate).toBeUndefined();
  });
});
