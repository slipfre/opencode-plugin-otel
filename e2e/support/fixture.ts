import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { OtelPluginOptions } from "../../src/config.ts";
import { startFakeLlm, type LlmReply } from "./fake-llm.ts";
import {
  createOpenCodeRunner,
  e2eAvailable,
  opencodeEntry,
} from "./opencode-process.ts";
import { startOtlpReceiver } from "./otlp-receiver.ts";

type FixtureOptions = {
  caseID: string;
  replies: LlmReply[];
  pluginOptions?: OtelPluginOptions;
  autoCompact?: boolean;
};

const tempRoot = process.env["OPENCODE_E2E_TMPDIR"] ?? tmpdir();

function providerConfig(baseURL: string) {
  return {
    name: "Test",
    id: "test",
    env: [],
    npm: "@ai-sdk/openai-compatible",
    models: {
      "test-model": {
        id: "test-model",
        name: "Test Model",
        attachment: false,
        reasoning: false,
        temperature: false,
        tool_call: true,
        release_date: "2025-01-01",
        limit: { context: 100_000, output: 10_000 },
        cost: { input: 0, output: 0 },
        options: {},
      },
    },
    options: { apiKey: "test-key", baseURL },
  };
}

export { e2eAvailable };
export type { RunResult } from "./opencode-process.ts";

export async function createE2EFixture(options: FixtureOptions) {
  if (!e2eAvailable) {
    throw new Error(`OpenCode entry not found: ${opencodeEntry}`);
  }
  const home = await mkdtemp(path.join(tempRoot, "opencode-otel-e2e-"));
  const configDir = path.join(home, ".config/opencode");
  await Promise.all([
    mkdir(path.join(configDir, "node_modules"), { recursive: true }),
    mkdir(path.join(home, ".local/share"), { recursive: true }),
    mkdir(path.join(home, ".local/state"), { recursive: true }),
    mkdir(path.join(home, ".cache"), { recursive: true }),
  ]);
  await Bun.write(
    path.join(configDir, "package-lock.json"),
    JSON.stringify({
      packages: { "": { dependencies: { "@opencode-ai/plugin": "0.0.0" } } },
    })
  );
  const llm = startFakeLlm(options.replies);
  const otlp = startOtlpReceiver();
  const pluginEntry = path.resolve(import.meta.dir, "../../src/index.ts");
  const config = {
    formatter: false,
    lsp: false,
    share: "disabled",
    plugin: [
      [
        pathToFileURL(pluginEntry).href,
        {
          enabled: true,
          endpoint: otlp.endpoint,
          protocol: "http/json",
          userIDEnabled: false,
          tracePrefix: "e2e.",
          spanAttributes: `e2e.case=${options.caseID}`,
          resourceAttributes: "e2e.resource=opencode-plugin-otel",
          ...options.pluginOptions,
        },
      ],
    ],
    provider: { test: providerConfig(llm.url) },
  };
  const runner = createOpenCodeRunner({
    home,
    config,
    autoCompact: options.autoCompact === true,
    model: llm,
    diagnostics: () =>
      [
        `llm hits:\n${JSON.stringify(
          llm.hits.map((hit) => hit.body),
          null,
          2
        )}`,
        `otlp errors:\n${JSON.stringify(otlp.errors)}`,
        `otlp payloads:\n${JSON.stringify(otlp.payloads, null, 2)}`,
      ].join("\n"),
  });

  return {
    home,
    llm,
    otlp,
    run: runner.run,
    startRun: runner.startRun,
    async close() {
      await runner.close();
      llm.stop();
      otlp.stop();
      await rm(home, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 50,
      });
    },
  };
}

export type E2EFixture = Awaited<ReturnType<typeof createE2EFixture>>;

export async function withE2EFixture<T>(
  options: FixtureOptions,
  callback: (fixture: E2EFixture) => Promise<T>
) {
  const fixture = await createE2EFixture(options);
  try {
    return await callback(fixture);
  } finally {
    await fixture.close();
  }
}
