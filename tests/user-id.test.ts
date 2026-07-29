import { afterEach, describe, expect, mock, setSystemTime, spyOn, test } from "bun:test"
import { USER_ID } from "@arizeai/openinference-semantic-conventions"
import type { HandlerContext } from "../src/types.ts"
import { createUserIDManager } from "../src/user-id.ts"

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>

type TestResolverOptions = {
  authHeader?: string
  endpoint?: string
  fetcher?: Fetcher
  log?: HandlerContext["log"]
  query?: (token: string) => Promise<unknown>
  requestTimeoutMs?: number
  retryCount?: number
  cooldownMs?: number
}

function mockFetch(fetcher: Fetcher) {
  return spyOn(globalThis, "fetch").mockImplementation(Object.assign(fetcher, {
    preconnect: () => {},
  }))
}

function createTestResolver(options: TestResolverOptions = {}) {
  const fetcher = options.fetcher ?? (async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as { token: string }
    const response = await options.query?.(body.token)
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  })
  mockFetch(fetcher)
  const ctx: Pick<HandlerContext, "commonAttrs" | "log"> = {
    commonAttrs: {},
    log: options.log ?? (async () => {}),
  }
  const manager = createUserIDManager({
    userIDEnabled: true,
    userIDEndpoint: options.endpoint ?? "queryUserByToken",
    userIDAuthHeader: options.authHeader,
    userIDTimeout: options.requestTimeoutMs ?? 3000,
    userIDRetryCount: options.retryCount ?? 2,
    userIDCooldown: options.cooldownMs ?? 300000,
  }, ctx, ctx.commonAttrs)

  return {
    resolve: async (providers: Parameters<typeof manager.configure>[0]) => {
      await manager.configure(providers)
      return ctx.commonAttrs[USER_ID] ?? "unknown"
    },
  }
}

afterEach(() => {
  mock.restore()
  setSystemTime()
})

describe("user ID resolution", () => {
  test("posts the API key token to the configured endpoint", async () => {
    const requests: Array<{ input: string; init?: RequestInit }> = []
    const logs: Array<{ level: string; message: string; extra?: Record<string, unknown> }> = []
    const resolver = createTestResolver({
      authHeader: "blackbox-secret",
      endpoint: "https://identity.example.com/queryUserByToken",
      fetcher: async (input, init) => {
        requests.push({ input: String(input), init })
        return new Response(JSON.stringify({
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }), {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      },
      log: async (level, message, extra) => {
        logs.push({ level, message, extra })
      },
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("user-1")
    expect(requests).toHaveLength(1)
    expect(requests[0]?.input).toBe("https://identity.example.com/queryUserByToken")
    expect(requests[0]?.init?.method).toBe("POST")
    expect(requests[0]?.init?.body).toBe(JSON.stringify({ token: "token" }))
    expect(requests[0]?.init?.headers).toEqual({
      "content-type": "application/json",
      "X-Blackbox-Auth": "blackbox-secret",
    })
    expect(logs.filter((entry) => entry.message.startsWith("user ID request"))).toEqual([
      {
        level: "debug",
        message: "user ID request sent",
        extra: {
          endpoint: "https://identity.example.com/queryUserByToken",
          attempt: 1,
          maxAttempts: 3,
        },
      },
      {
        level: "debug",
        message: "user ID request returned",
        extra: {
          endpoint: "https://identity.example.com/queryUserByToken",
          attempt: 1,
          maxAttempts: 3,
          code: 0,
          resolved: true,
        },
      },
    ])
    expect(JSON.stringify(logs)).not.toContain("token")
    expect(JSON.stringify(logs)).not.toContain("blackbox-secret")
  })

  test("omits X-Blackbox-Auth when its value is not configured", async () => {
    let headers: RequestInit["headers"]
    const resolver = createTestResolver({
      endpoint: "https://identity.example.com/queryUserByToken",
      fetcher: async (_input, init) => {
        headers = init?.headers
        return new Response(JSON.stringify({
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }))
      },
    })

    await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })
    expect(headers).toEqual({
      "content-type": "application/json",
    })
  })

  test("uses only the first provider with an API key", async () => {
    const calls: string[] = []
    const resolver = createTestResolver({
      query: async (token) => {
        calls.push(token)
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-anthropic" },
        }
      },
    })
    const providers = {
      missing: { options: {} },
      anthropic: { options: { apiKey: "anthropic-token" } },
      openai: { options: { apiKey: "openai-token" } },
    }

    expect(await resolver.resolve(providers)).toBe("user-anthropic")
    expect(calls).toEqual(["anthropic-token"])
  })

  test("returns unknown when no provider has a non-empty API key", async () => {
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }
      },
    })

    expect(await resolver.resolve({
      missing: { options: {} },
      empty: { options: { apiKey: "  " } },
    })).toBe("unknown")
    expect(calls).toBe(0)
  })

  test("retries with backoff before returning unknown", async () => {
    const logs: Array<{ message: string; extra?: Record<string, unknown> }> = []
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        throw new Error("unavailable")
      },
      retryCount: 1,
      log: async (_level, message, extra) => {
        logs.push({ message, extra })
      },
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("unknown")
    expect(calls).toBe(2)
    expect(logs.filter((entry) => entry.message === "user ID request sent")).toHaveLength(2)
    expect(logs.filter((entry) => entry.message === "user ID request failed").map((entry) => entry.extra)).toEqual([
      {
        endpoint: "queryUserByToken",
        attempt: 1,
        maxAttempts: 2,
        error: "unavailable",
        retrying: true,
        retryDelayMs: 250,
      },
      {
        endpoint: "queryUserByToken",
        attempt: 2,
        maxAttempts: 2,
        error: "unavailable",
        retrying: false,
      },
    ])
  })

  test("uses the configured retry count with exponential backoff", async () => {
    const delays: number[] = []
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        throw new Error("unavailable")
      },
      retryCount: 2,
      log: async (_level, message, extra) => {
        if (message === "user ID request failed" && typeof extra?.retryDelayMs === "number") {
          delays.push(extra.retryDelayMs)
        }
      },
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("unknown")
    expect(calls).toBe(3)
    expect(delays).toEqual([250, 500])
  })

  test("retries invalid responses and accepts a later valid response", async () => {
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        if (calls < 2) return { code: 1, msg: "unavailable" }
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }
      },
      retryCount: 1,
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("user-1")
    expect(calls).toBe(2)
  })

  test("does not cache unknown as a resolved user ID", async () => {
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: calls === 1 ? "unknown" : "user-1" },
        }
      },
      retryCount: 0,
      cooldownMs: 0,
    })
    const providers = {
      configured: { options: { apiKey: "token" } },
    }

    expect(await resolver.resolve(providers)).toBe("unknown")
    expect(await resolver.resolve(providers)).toBe("user-1")
    expect(calls).toBe(2)
  })

  test("waits for the cooldown and retries on the next resolve", async () => {
    setSystemTime(new Date(1000))
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        if (calls === 1) throw new Error("unavailable")
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }
      },
      retryCount: 0,
      cooldownMs: 5000,
    })
    const providers = {
      configured: { options: { apiKey: "token" } },
    }

    expect(await resolver.resolve(providers)).toBe("unknown")
    setSystemTime(new Date(5999))
    expect(await resolver.resolve(providers)).toBe("unknown")
    expect(calls).toBe(1)
    setSystemTime(new Date(6000))
    expect(await resolver.resolve(providers)).toBe("user-1")
    expect(calls).toBe(2)
  })

  test("shares one in-flight retry sequence", async () => {
    let release: (() => void) | undefined
    let calls = 0
    const resolver = createTestResolver({
      query: async () => {
        calls++
        await new Promise<void>((resolve) => {
          release = resolve
        })
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }
      },
      retryCount: 0,
    })
    const providers = {
      configured: { options: { apiKey: "token" } },
    }

    const first = resolver.resolve(providers)
    const second = resolver.resolve(providers)
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
    release?.()

    expect(await first).toBe("user-1")
    expect(await second).toBe("user-1")
    expect(calls).toBe(1)
  })

  test("does not fall back to another provider after lookup failure", async () => {
    const calls: string[] = []
    const resolver = createTestResolver({
      query: async (token) => {
        calls.push(token)
        throw new Error("unavailable")
      },
      retryCount: 0,
    })

    expect(await resolver.resolve({
      first: { options: { apiKey: "first-token" } },
      second: { options: { apiKey: "second-token" } },
    })).toBe("unknown")
    expect(calls).toEqual(["first-token"])
  })

  test("trims the API key and resolved user ID", async () => {
    const calls: string[] = []
    const resolver = createTestResolver({
      query: async (token) => {
        calls.push(token)
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: " user-1 " },
        }
      },
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: " token " } },
    })).toBe("user-1")
    expect(calls).toEqual(["token"])
  })
})

describe("createUserIDManager", () => {
  test("owns common attributes, provider state, and successful lookup caching", async () => {
    let calls = 0
    mockFetch(async () => {
      calls++
      return new Response(JSON.stringify({
        code: 0,
        msg: "ok",
        result: { ssicNo: "user-1" },
      }))
    })
    const ctx: Pick<HandlerContext, "commonAttrs" | "log"> = {
      commonAttrs: {
        "project.id": "project-1",
      },
      log: async () => {},
    }
    const manager = createUserIDManager({
      userIDEnabled: true,
      userIDEndpoint: "queryUserByToken",
      userIDAuthHeader: undefined,
      userIDTimeout: 3000,
      userIDRetryCount: 2,
      userIDCooldown: 300000,
    }, ctx, ctx.commonAttrs)
    const providers = {
      configured: { options: { apiKey: "token" } },
    }

    await manager.configure(providers)
    manager.refreshInBackground()
    await new Promise<void>((resolve) => setTimeout(resolve, 0))

    expect(ctx.commonAttrs).toEqual({
      "project.id": "project-1",
      [USER_ID]: "user-1",
    })
    expect(calls).toBe(1)
  })

  test("does not resolve again when common attributes already contain a user ID", async () => {
    const fetcher = spyOn(globalThis, "fetch")
    const ctx: Pick<HandlerContext, "commonAttrs" | "log"> = {
      commonAttrs: {
        [USER_ID]: "user-1",
      },
      log: async () => {},
    }
    const manager = createUserIDManager({
      userIDEnabled: true,
      userIDEndpoint: "queryUserByToken",
      userIDAuthHeader: undefined,
      userIDTimeout: 3000,
      userIDRetryCount: 2,
      userIDCooldown: 300000,
    }, ctx, ctx.commonAttrs)

    await manager.configure({
      configured: { options: { apiKey: "token" } },
    })
    manager.refreshInBackground()
    await new Promise<void>((resolve) => setTimeout(resolve, 0))

    expect(ctx.commonAttrs[USER_ID]).toBe("user-1")
    expect(fetcher).not.toHaveBeenCalled()
  })

  test("does nothing when user ID resolution is disabled", async () => {
    const fetcher = spyOn(globalThis, "fetch")
    const ctx: Pick<HandlerContext, "commonAttrs" | "log"> = {
      commonAttrs: {
        "project.id": "project-1",
      },
      log: async () => {},
    }
    const manager = createUserIDManager({
      userIDEnabled: false,
      userIDEndpoint: "queryUserByToken",
      userIDAuthHeader: undefined,
      userIDTimeout: 3000,
      userIDRetryCount: 2,
      userIDCooldown: 300000,
    }, ctx, ctx.commonAttrs)

    await manager.configure({
      configured: { options: { apiKey: "token" } },
    })
    manager.refreshInBackground()
    await new Promise<void>((resolve) => setTimeout(resolve, 0))

    expect(ctx.commonAttrs).toEqual({
      "project.id": "project-1",
    })
    expect(fetcher).not.toHaveBeenCalled()
  })
})
