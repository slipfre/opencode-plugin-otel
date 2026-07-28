import type { Config } from "@opencode-ai/plugin"
import type { PluginLogger } from "./types.ts"

type QueryUserByTokenResponse = {
  code: number
  msg: string
  result?: {
    ssicNo?: string
    fullName?: string
    adminDepName?: string
    baseName?: string
  }
}

type QueryUserByToken = (token: string) => Promise<unknown>
type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>

type UserIDResolverOptions = {
  authHeader?: string
  endpoint?: string
  fetcher?: Fetcher
  log?: PluginLogger
  query?: QueryUserByToken
  requestTimeoutMs?: number
  retryCount?: number
  retryDelaysMs?: readonly number[]
  cooldownMs?: number
  now?: () => number
  sleep?: (delayMs: number) => Promise<void>
}

const UNKNOWN_USER_ID = "unknown"
const QUERY_USER_BY_TOKEN_ENDPOINT = "queryUserByToken"
const DEFAULT_QUERY_TIMEOUT_MS = 3000
const DEFAULT_RETRY_COUNT = 2
const RETRY_BASE_DELAY_MS = 250
const DEFAULT_COOLDOWN_MS = 5 * 60 * 1000

async function queryUserByToken(
  token: string,
  endpoint: string,
  fetcher: Fetcher,
  timeoutMs: number,
  authHeader: string | undefined,
): Promise<QueryUserByTokenResponse> {
  const response = await fetcher(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(authHeader ? { "X-Blackbox-Auth": authHeader } : {}),
    },
    body: JSON.stringify({ token }),
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (!response.ok) throw new Error(`queryUserByToken failed with HTTP ${response.status}`)
  return await response.json() as QueryUserByTokenResponse
}

function userIDFromResponse(response: unknown): string | undefined {
  if (!response || typeof response !== "object") return
  const candidate = response as QueryUserByTokenResponse
  if (candidate.code !== 0 || !candidate.result) return
  const userID = candidate.result.ssicNo
  return typeof userID === "string" && userID.trim().length > 0 ? userID.trim() : undefined
}

function responseCode(response: unknown): number | undefined {
  if (!response || typeof response !== "object") return
  const code = (response as { code?: unknown }).code
  return typeof code === "number" ? code : undefined
}

function apiKeyFromProviders(providers: Config["provider"]): string | undefined {
  const provider = Object.values(providers ?? {}).find((candidate) => {
    const apiKey = candidate.options?.apiKey
    return typeof apiKey === "string" && apiKey.trim().length > 0
  })
  const apiKey = provider?.options?.apiKey
  return typeof apiKey === "string" ? apiKey.trim() : undefined
}

function createUserIDResolver(options: UserIDResolverOptions = {}) {
  const endpoint = options.endpoint ?? QUERY_USER_BY_TOKEN_ENDPOINT
  const fetcher = options.fetcher ?? fetch
  const requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_QUERY_TIMEOUT_MS
  const query = options.query
    ?? ((token: string) => queryUserByToken(token, endpoint, fetcher, requestTimeoutMs, options.authHeader))
  const retryCount = options.retryCount ?? DEFAULT_RETRY_COUNT
  const retryDelaysMs = options.retryDelaysMs
    ?? Array.from({ length: retryCount }, (_, index) => RETRY_BASE_DELAY_MS * (2 ** index))
  const cooldownMs = options.cooldownMs ?? DEFAULT_COOLDOWN_MS
  const now = options.now ?? Date.now
  const sleep = options.sleep ?? ((delayMs: number) => new Promise<void>((resolve) => setTimeout(resolve, delayMs)))
  const writeLog = async (
    level: "debug" | "warn",
    message: string,
    extra: Record<string, unknown>,
  ) => {
    try {
      await options.log?.(level, message, extra)
    } catch {
    }
  }
  let currentToken: string | undefined
  let currentUserID = UNKNOWN_USER_ID
  let retryAfter = 0
  let generation = 0
  let inFlight: Promise<string> | undefined

  const queryWithRetry = async (token: string): Promise<string | undefined> => {
    for (let attempt = 0; attempt <= retryDelaysMs.length; attempt++) {
      const attemptNumber = attempt + 1
      const maxAttempts = retryDelaysMs.length + 1
      await writeLog("debug", "user ID request sent", {
        endpoint,
        attempt: attemptNumber,
        maxAttempts,
      })
      let failure: string
      try {
        const response = await query(token)
        const userID = userIDFromResponse(response)
        await writeLog("debug", "user ID request returned", {
          endpoint,
          attempt: attemptNumber,
          maxAttempts,
          code: responseCode(response),
          resolved: userID !== undefined,
        })
        if (userID) return userID
        failure = "response did not contain a valid user ID"
      } catch (error) {
        failure = error instanceof Error ? error.message : String(error)
      }
      const delayMs = retryDelaysMs[attempt]
      await writeLog("warn", "user ID request failed", {
        endpoint,
        attempt: attemptNumber,
        maxAttempts,
        error: failure,
        retrying: delayMs !== undefined,
        ...(delayMs === undefined ? {} : { retryDelayMs: delayMs }),
      })
      if (delayMs === undefined) return
      await sleep(delayMs)
    }
  }

  const resolve = (providers: Config["provider"]): Promise<string> => {
    const token = apiKeyFromProviders(providers)
    if (!token) {
      currentToken = undefined
      currentUserID = UNKNOWN_USER_ID
      retryAfter = 0
      generation++
      inFlight = undefined
      return Promise.resolve(currentUserID)
    }

    if (token !== currentToken) {
      currentToken = token
      currentUserID = UNKNOWN_USER_ID
      retryAfter = 0
      generation++
      inFlight = undefined
    }

    if (currentUserID !== UNKNOWN_USER_ID) return Promise.resolve(currentUserID)
    if (now() < retryAfter) return Promise.resolve(currentUserID)
    if (inFlight) return inFlight

    const requestGeneration = generation
    const request = queryWithRetry(token)
      .then((userID) => {
        if (requestGeneration !== generation || token !== currentToken) return currentUserID
        currentUserID = userID ?? UNKNOWN_USER_ID
        retryAfter = userID ? 0 : now() + cooldownMs
        return currentUserID
      })
      .catch(() => {
        if (requestGeneration !== generation || token !== currentToken) return currentUserID
        currentUserID = UNKNOWN_USER_ID
        retryAfter = now() + cooldownMs
        return currentUserID
      })

    inFlight = request
    void request.finally(() => {
      if (inFlight === request) inFlight = undefined
    })
    return request
  }

  return {
    current: () => currentUserID,
    resolve,
  }
}

export { createUserIDResolver }
