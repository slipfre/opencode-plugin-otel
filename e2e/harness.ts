import { existsSync } from "node:fs"
import { mkdir, mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { pathToFileURL } from "node:url"
import { startFakeLlm, type LlmReply } from "./fake-llm.ts"
import { startOtlpReceiver } from "./otlp-receiver.ts"

type PluginOptions = {
  enabled?: boolean
  tracePrefix?: string
  traceparent?: string
  tracestate?: string
  tracePropagationProviders?: string[]
}

type FixtureOptions = {
  caseID: string
  replies: LlmReply[]
  pluginOptions?: PluginOptions
  autoCompact?: boolean
}

export type RunResult = {
  exitCode: number
  stdout: string
  stderr: string
  durationMs: number
}

const configuredEntry = process.env["OPENCODE_E2E_ENTRY"]
export const opencodeEntry = configuredEntry
  ? path.resolve(configuredEntry)
  : path.resolve(import.meta.dir, "../../opencode/packages/opencode/src/index.ts")

if (configuredEntry && !existsSync(opencodeEntry)) {
  throw new Error(`OPENCODE_E2E_ENTRY does not exist: ${opencodeEntry}`)
}

export const e2eAvailable = existsSync(opencodeEntry)

const tempRoot = process.env["OPENCODE_E2E_TMPDIR"] ?? tmpdir()

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
  }
}

function isolatedEnv(home: string, config: unknown, autoCompact: boolean) {
  const inherited: Record<string, string> = {}
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue
    const normalized = process.platform === "win32" ? key.toUpperCase() : key
    if (normalized.startsWith("OPENCODE_") || normalized.startsWith("OTEL_")) continue
    inherited[normalized] = value
  }
  const env: Record<string, string> = {
    ...inherited,
    PWD: home,
    HOME: home,
    XDG_CONFIG_HOME: path.join(home, ".config"),
    XDG_DATA_HOME: path.join(home, ".local/share"),
    XDG_STATE_HOME: path.join(home, ".local/state"),
    XDG_CACHE_HOME: path.join(home, ".cache"),
    OPENCODE_TEST_HOME: home,
    OPENCODE_CONFIG_CONTENT: JSON.stringify(config),
    OPENCODE_DISABLE_PROJECT_CONFIG: "1",
    OPENCODE_DISABLE_DEFAULT_PLUGINS: "1",
    OPENCODE_DISABLE_EXTERNAL_SKILLS: "1",
    OPENCODE_DISABLE_AUTOUPDATE: "1",
    OPENCODE_DISABLE_MODELS_FETCH: "1",
    OPENCODE_AUTH_CONTENT: "{}",
  }
  if (!autoCompact) env["OPENCODE_DISABLE_AUTOCOMPACT"] = "1"
  return env
}

async function terminate(proc: Bun.Subprocess) {
  if (proc.exitCode !== null) return
  proc.kill("SIGTERM")
  await Promise.race([proc.exited, Bun.sleep(2_000)])
  if (proc.exitCode === null) proc.kill("SIGKILL")
  await proc.exited
}

function availablePort() {
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: () => new Response(),
  })
  const port = server.port
  server.stop(true)
  return port
}

async function waitFor(description: string, timeoutMs: number, ready: () => boolean | Promise<boolean>) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await ready()) return
    await Bun.sleep(25)
  }
  throw new Error(`Timed out waiting for ${description} after ${timeoutMs}ms`)
}

export async function createE2EFixture(options: FixtureOptions) {
  if (!e2eAvailable) throw new Error(`OpenCode entry not found: ${opencodeEntry}`)
  const home = await mkdtemp(path.join(tempRoot, "opencode-otel-e2e-"))
  const configDir = path.join(home, ".config/opencode")
  await Promise.all([
    mkdir(path.join(configDir, "node_modules"), { recursive: true }),
    mkdir(path.join(home, ".local/share"), { recursive: true }),
    mkdir(path.join(home, ".local/state"), { recursive: true }),
    mkdir(path.join(home, ".cache"), { recursive: true }),
  ])
  await Bun.write(
    path.join(configDir, "package-lock.json"),
    JSON.stringify({ packages: { "": { dependencies: { "@opencode-ai/plugin": "0.0.0" } } } }),
  )
  const llm = startFakeLlm(options.replies)
  const otlp = startOtlpReceiver()
  const pluginEntry = path.resolve(import.meta.dir, "../src/index.ts")
  const config = {
    formatter: false,
    lsp: false,
    share: "disabled",
    plugin: [[pathToFileURL(pluginEntry).href, {
      enabled: true,
      endpoint: otlp.endpoint,
      protocol: "http/json",
      userIDEnabled: false,
      tracePrefix: "e2e.",
      spanAttributes: `e2e.case=${options.caseID}`,
      resourceAttributes: "e2e.resource=opencode-plugin-otel",
      ...options.pluginOptions,
    }]],
    provider: { test: providerConfig(llm.url) },
  }
  let activeServer: Bun.Subprocess | undefined

  return {
    home,
    llm,
    otlp,
    async run(prompt: string, extraArgs: string[] = [], timeoutMs = 45_000): Promise<RunResult> {
      const started = Date.now()
      const proc = Bun.spawn([
        process.execPath,
        "run",
        "--conditions=browser",
        opencodeEntry,
        "--print-logs",
        "--log-level",
        "DEBUG",
        "run",
        "--model",
        "test/test-model",
        "--format",
        "json",
        ...extraArgs,
        prompt,
      ], {
        cwd: home,
        env: isolatedEnv(home, config, options.autoCompact === true),
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
      })
      const stdout = new Response(proc.stdout).text()
      const stderr = new Response(proc.stderr).text()
      let timer: ReturnType<typeof setTimeout> | undefined
      try {
        const exitCode = await Promise.race([
          proc.exited,
          new Promise<never>((_, reject) => {
            timer = setTimeout(() => reject(new Error(`OpenCode timed out after ${timeoutMs}ms`)), timeoutMs)
          }),
        ])
        return {
          exitCode,
          stdout: await stdout,
          stderr: await stderr,
          durationMs: Date.now() - started,
        }
      } catch (error) {
        await terminate(proc)
        const output = await stdout
        const errors = await stderr
        throw new Error([
          error instanceof Error ? error.message : String(error),
          `stdout:\n${output}`,
          `stderr:\n${errors}`,
          `llm hits:\n${JSON.stringify(llm.hits.map(hit => hit.body), null, 2)}`,
          `otlp errors:\n${JSON.stringify(otlp.errors)}`,
          `otlp payloads:\n${JSON.stringify(otlp.payloads, null, 2)}`,
        ].join("\n"))
      } finally {
        if (timer) clearTimeout(timer)
      }
    },
    async startRun(initialPrompt: string, timeoutMs = 45_000) {
      if (activeServer) throw new Error("An OpenCode server run is already active")
      const started = Date.now()
      const port = availablePort()
      const baseURL = `http://127.0.0.1:${port}`
      const proc = Bun.spawn([
        process.execPath,
        "run",
        "--conditions=browser",
        opencodeEntry,
        "--print-logs",
        "--log-level",
        "DEBUG",
        "serve",
        "--hostname",
        "127.0.0.1",
        "--port",
        String(port),
      ], {
        cwd: home,
        env: isolatedEnv(home, config, options.autoCompact === true),
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
      })
      activeServer = proc
      const stdoutPromise = new Response(proc.stdout).text()
      const stderrPromise = new Response(proc.stderr).text()
      let finalized = false
      const finalize = async (failure?: unknown): Promise<RunResult> => {
        if (finalized) throw new Error("The active OpenCode run is already finished")
        finalized = true
        llm.release()
        await terminate(proc)
        if (activeServer === proc) activeServer = undefined
        const stdout = await stdoutPromise
        const stderr = await stderrPromise
        if (failure) {
          throw new Error([
            failure instanceof Error ? failure.message : String(failure),
            `stdout:\n${stdout}`,
            `stderr:\n${stderr}`,
            `llm hits:\n${JSON.stringify(llm.hits.map(hit => hit.body), null, 2)}`,
            `otlp errors:\n${JSON.stringify(otlp.errors)}`,
            `otlp payloads:\n${JSON.stringify(otlp.payloads, null, 2)}`,
          ].join("\n"))
        }
        return { exitCode: 0, stdout, stderr, durationMs: Date.now() - started }
      }

      let request: (pathname: string, init?: RequestInit) => Promise<Response>
      let sessionID: string
      let send: (text: string) => Promise<Response>
      try {
        await waitFor("OpenCode server startup", 15_000, async () => {
          if (proc.exitCode !== null) throw new Error(`OpenCode server exited ${proc.exitCode} during startup`)
          const url = new URL("/global/health", baseURL)
          url.searchParams.set("directory", home)
          return fetch(url, { signal: AbortSignal.timeout(1_000) }).then(response => response.ok).catch(() => false)
        })

        request = async (pathname: string, init?: RequestInit) => {
          const url = new URL(pathname, baseURL)
          url.searchParams.set("directory", home)
          let response: Response
          try {
            response = await fetch(url, {
              ...init,
              headers: { "content-type": "application/json", ...init?.headers },
              signal: init?.signal ?? AbortSignal.timeout(30_000),
            })
          } catch (error) {
            throw new Error(`${init?.method ?? "GET"} ${url.pathname} failed: ${error instanceof Error ? error.message : String(error)}`)
          }
          if (response.ok) return response
          throw new Error(`${init?.method ?? "GET"} ${url.pathname} returned ${response.status}: ${await response.text()}`)
        }

        const created = await request("/session", { method: "POST", body: "{}" })
        const session = await created.json() as { id?: unknown }
        if (typeof session.id !== "string") throw new Error(`Invalid session response: ${JSON.stringify(session)}`)
        sessionID = session.id
        send = (text: string) => request(`/session/${sessionID}/prompt_async`, {
          method: "POST",
          body: JSON.stringify({
            model: { providerID: "test", modelID: "test-model" },
            agent: "build",
            parts: [{ type: "text", text }],
          }),
        })

        await send(initialPrompt)
        await waitFor("the first held model request", 15_000, () => llm.held() === 1)
      } catch (error) {
        await finalize(error)
        throw error
      }

      let steered = false
      let compacted = false
      return {
        async steer(guidancePrompt: string) {
          if (steered || compacted) throw new Error("The active OpenCode run has already been advanced")
          try {
            await send(guidancePrompt)
            await waitFor("the steering message to be stored", 15_000, async () => {
              const response = await request(`/session/${sessionID}/message`)
              const messages = await response.json() as Array<{ info?: { role?: unknown } }>
              return messages.filter(message => message.info?.role === "user").length === 2
            })
            steered = true
            llm.release()
          } catch (error) {
            await finalize(error)
          }
        },
        async finish(): Promise<RunResult> {
          if (!steered) throw new Error("Steer the active OpenCode run before finishing it")
          try {
            await waitFor("the steered run to finish", timeoutMs, async () => {
              if (llm.pending() !== 0) return false
              const response = await request("/session/status")
              const statuses = await response.json() as Record<string, unknown>
              return statuses[sessionID] === undefined
            })
            return await finalize()
          } catch (error) {
            return await finalize(error)
          }
        },
        async manualCompact(): Promise<RunResult> {
          if (steered || compacted) throw new Error("The active OpenCode run has already been advanced")
          compacted = true
          try {
            llm.release()
            await waitFor("the initial run to finish", timeoutMs, async () => {
              const response = await request("/session/status")
              const statuses = await response.json() as Record<string, unknown>
              return statuses[sessionID] === undefined
            })
            const response = await request(`/session/${sessionID}/summarize`, {
              method: "POST",
              body: JSON.stringify({ providerID: "test", modelID: "test-model" }),
            })
            if (await response.json() !== true) throw new Error("OpenCode did not confirm manual compaction")
            await waitFor("manual compaction to finish", timeoutMs, async () => {
              if (llm.pending() !== 0) return false
              const statusResponse = await request("/session/status")
              const statuses = await statusResponse.json() as Record<string, unknown>
              return statuses[sessionID] === undefined
            })
            return await finalize()
          } catch (error) {
            return await finalize(error)
          }
        },
      }
    },
    async close() {
      if (activeServer) {
        llm.release()
        await terminate(activeServer)
        activeServer = undefined
      }
      llm.stop()
      otlp.stop()
      await rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 })
    },
  }
}
