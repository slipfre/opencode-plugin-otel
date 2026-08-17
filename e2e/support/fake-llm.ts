type Usage = {
  input: number;
  output: number;
  cacheRead?: number;
  reasoning?: number;
};

export type TextReply = {
  type: "text";
  text: string;
  reasoning?: string;
  usage?: Usage;
  hold?: boolean;
};

export type ToolReply = {
  type: "tool";
  name: string;
  input: unknown;
  usage?: Usage;
  hold?: boolean;
};

export type ErrorReply = {
  type: "error";
  code: string;
  message: string;
  status?: number;
  hold?: boolean;
};

export type LlmReply = TextReply | ToolReply | ErrorReply;

export type LlmHit = {
  body: Record<string, unknown>;
  headers: Headers;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function chunk(input: {
  delta?: Record<string, unknown>;
  finish?: string;
  usage?: Usage;
}) {
  return {
    id: "chatcmpl-e2e",
    object: "chat.completion.chunk",
    choices: [
      {
        delta: input.delta ?? {},
        ...(input.finish ? { finish_reason: input.finish } : {}),
      },
    ],
    ...(input.usage
      ? {
          usage: {
            prompt_tokens: input.usage.input,
            completion_tokens: input.usage.output,
            total_tokens: input.usage.input + input.usage.output,
            ...(input.usage.cacheRead === undefined
              ? {}
              : {
                  prompt_tokens_details: {
                    cached_tokens: input.usage.cacheRead,
                  },
                }),
            ...(input.usage.reasoning === undefined
              ? {}
              : {
                  completion_tokens_details: {
                    reasoning_tokens: input.usage.reasoning,
                  },
                }),
          },
        }
      : {}),
  };
}

function sse(lines: unknown[]) {
  const body = [
    ...lines.map((line) => `data: ${JSON.stringify(line)}\n\n`),
    "data: [DONE]\n\n",
  ].join("");
  return new Response(body, {
    headers: { "content-type": "text/event-stream" },
  });
}

function titleRequest(body: Record<string, unknown>) {
  return JSON.stringify(body).includes(
    "Generate a title for this conversation"
  );
}

function response(reply: LlmReply) {
  if (reply.type === "error") {
    return Response.json(
      {
        error: {
          message: reply.message,
          type: "invalid_request_error",
          param: null,
          code: reply.code,
        },
      },
      { status: reply.status ?? 400 }
    );
  }
  const start = chunk({ delta: { role: "assistant" } });
  if (reply.type === "text") {
    return sse([
      start,
      ...(reply.reasoning
        ? [chunk({ delta: { reasoning_content: reply.reasoning } })]
        : []),
      chunk({ delta: { content: reply.text } }),
      chunk({ finish: "stop", usage: reply.usage }),
    ]);
  }
  const id = "call_e2e_1";
  return sse([
    start,
    chunk({
      delta: {
        tool_calls: [
          {
            index: 0,
            id,
            type: "function",
            function: { name: reply.name, arguments: "" },
          },
        ],
      },
    }),
    chunk({
      delta: {
        tool_calls: [
          {
            index: 0,
            function: { arguments: JSON.stringify(reply.input) },
          },
        ],
      },
    }),
    chunk({ finish: "tool_calls", usage: reply.usage }),
  ]);
}

export function startFakeLlm(replies: LlmReply[]) {
  const queue = [...replies];
  const hits: LlmHit[] = [];
  const held: Array<() => void> = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      if (new URL(request.url).pathname !== "/v1/chat/completions") {
        return new Response("not found", { status: 404 });
      }
      const raw = await request.json().catch(() => ({}));
      const body = isRecord(raw) ? raw : {};
      hits.push({ body, headers: new Headers(request.headers) });
      if (titleRequest(body)) {
        return response({ type: "text", text: "E2E title" });
      }
      const next = queue.shift();
      if (!next) {
        return response({ type: "text", text: "unexpected request" });
      }
      if (next.hold) {
        await new Promise<void>((resolve) => held.push(resolve));
      }
      return response(next);
    },
  });
  return {
    url: `http://${server.hostname}:${server.port}/v1`,
    hits,
    mainHits: () => hits.filter((hit) => !titleRequest(hit.body)),
    held: () => held.length,
    pending: () => queue.length,
    release: () => held.shift()?.(),
    stop: () => {
      held.splice(0).forEach((resolve) => resolve());
      server.stop(true);
    },
  };
}
