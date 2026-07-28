import { describe, expect, test } from "bun:test"
import { createUserIDResolver } from "../src/user-id.ts"

describe("createUserIDResolver", () => {
  test("posts the API key token to the configured endpoint", async () => {
    const requests: Array<{ input: string; init?: RequestInit }> = []
    const logs: Array<{ level: string; message: string; extra?: Record<string, unknown> }> = []
    const resolver = createUserIDResolver({
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
    expect(logs).toEqual([
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
    const resolver = createUserIDResolver({
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
    const resolver = createUserIDResolver({
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
    const resolver = createUserIDResolver({
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
    const delays: number[] = []
    const logs: Array<{ message: string; extra?: Record<string, unknown> }> = []
    let calls = 0
    const resolver = createUserIDResolver({
      query: async () => {
        calls++
        throw new Error("unavailable")
      },
      retryDelaysMs: [100, 200],
      log: async (_level, message, extra) => {
        logs.push({ message, extra })
      },
      sleep: async (delayMs) => {
        delays.push(delayMs)
      },
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("unknown")
    expect(calls).toBe(3)
    expect(delays).toEqual([100, 200])
    expect(logs.filter((entry) => entry.message === "user ID request sent")).toHaveLength(3)
    expect(logs.filter((entry) => entry.message === "user ID request failed").map((entry) => entry.extra)).toEqual([
      {
        endpoint: "queryUserByToken",
        attempt: 1,
        maxAttempts: 3,
        error: "unavailable",
        retrying: true,
        retryDelayMs: 100,
      },
      {
        endpoint: "queryUserByToken",
        attempt: 2,
        maxAttempts: 3,
        error: "unavailable",
        retrying: true,
        retryDelayMs: 200,
      },
      {
        endpoint: "queryUserByToken",
        attempt: 3,
        maxAttempts: 3,
        error: "unavailable",
        retrying: false,
      },
    ])
  })

  test("uses the configured retry count with exponential backoff", async () => {
    const delays: number[] = []
    let calls = 0
    const resolver = createUserIDResolver({
      query: async () => {
        calls++
        throw new Error("unavailable")
      },
      retryCount: 3,
      sleep: async (delayMs) => {
        delays.push(delayMs)
      },
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("unknown")
    expect(calls).toBe(4)
    expect(delays).toEqual([250, 500, 1000])
  })

  test("retries invalid responses and accepts a later valid response", async () => {
    let calls = 0
    const resolver = createUserIDResolver({
      query: async () => {
        calls++
        if (calls < 3) return { code: 1, msg: "unavailable" }
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }
      },
      retryDelaysMs: [100, 200],
      sleep: async () => {},
    })

    expect(await resolver.resolve({
      configured: { options: { apiKey: "token" } },
    })).toBe("user-1")
    expect(calls).toBe(3)
  })

  test("waits for the cooldown and retries on the next resolve", async () => {
    let clock = 1000
    let calls = 0
    const resolver = createUserIDResolver({
      query: async () => {
        calls++
        if (calls === 1) throw new Error("unavailable")
        return {
          code: 0,
          msg: "ok",
          result: { ssicNo: "user-1" },
        }
      },
      retryDelaysMs: [],
      cooldownMs: 5000,
      now: () => clock,
    })
    const providers = {
      configured: { options: { apiKey: "token" } },
    }

    expect(await resolver.resolve(providers)).toBe("unknown")
    clock = 5999
    expect(await resolver.resolve(providers)).toBe("unknown")
    expect(calls).toBe(1)
    clock = 6000
    expect(await resolver.resolve(providers)).toBe("user-1")
    expect(calls).toBe(2)
  })

  test("shares one in-flight retry sequence", async () => {
    let release: (() => void) | undefined
    let calls = 0
    const resolver = createUserIDResolver({
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
      retryDelaysMs: [],
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
    const resolver = createUserIDResolver({
      query: async (token) => {
        calls.push(token)
        throw new Error("unavailable")
      },
      retryDelaysMs: [],
    })

    expect(await resolver.resolve({
      first: { options: { apiKey: "first-token" } },
      second: { options: { apiKey: "second-token" } },
    })).toBe("unknown")
    expect(calls).toEqual(["first-token"])
  })

  test("trims the API key and resolved user ID", async () => {
    const calls: string[] = []
    const resolver = createUserIDResolver({
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
