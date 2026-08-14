import type { Config } from "@opencode-ai/plugin";
import { USER_ID } from "@arizeai/openinference-semantic-conventions";
import type { PluginConfig } from "./config.ts";
import type { CommonAttrs, HandlerContext, PluginLogger } from "./types.ts";

type QueryUserByTokenResponse = {
  code?: unknown;
  result?: {
    ssicNo?: unknown;
  };
};

type QueryUserByTokenResult = {
  code?: number;
  userID?: string;
};

type UserIDRequestConfig = {
  authHeader?: string;
  endpoint: string;
  timeoutMs: number;
};

type UserIDResolverConfig = UserIDRequestConfig & {
  cooldownMs: number;
  log: PluginLogger;
  retryCount: number;
};

type UserIDManagerConfig = Pick<
  PluginConfig,
  | "userIDEnabled"
  | "userIDEndpoint"
  | "userIDAuthHeader"
  | "userIDTimeout"
  | "userIDRetryCount"
  | "userIDCooldown"
>;

type UserIDManagerContext = Pick<HandlerContext, "commonAttrs" | "log">;

const UNKNOWN_USER_ID = "unknown";
const RETRY_BASE_DELAY_MS = 250;

function isResolvedUserID(userID: unknown): userID is string {
  return (
    typeof userID === "string" &&
    userID.trim().length > 0 &&
    userID.trim() !== UNKNOWN_USER_ID
  );
}

async function queryUserByToken(
  token: string,
  config: UserIDRequestConfig
): Promise<QueryUserByTokenResult> {
  const response = await fetch(config.endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(config.authHeader ? { "X-Blackbox-Auth": config.authHeader } : {}),
    },
    body: JSON.stringify({ token }),
    signal: AbortSignal.timeout(config.timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`queryUserByToken failed with HTTP ${response.status}`);
  }
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== "object") {
    return {};
  }
  const candidate = payload as QueryUserByTokenResponse;
  const code = typeof candidate.code === "number" ? candidate.code : undefined;
  const rawUserID = code === 0 ? candidate.result?.ssicNo : undefined;
  const normalizedUserID =
    typeof rawUserID === "string" ? rawUserID.trim() : undefined;
  const userID = isResolvedUserID(normalizedUserID)
    ? normalizedUserID
    : undefined;
  return { code, userID };
}

function apiKeyFromProviders(
  providers: Config["provider"]
): string | undefined {
  for (const provider of Object.values(providers ?? {})) {
    const apiKey = provider.options?.apiKey;
    if (typeof apiKey !== "string") {
      continue;
    }
    const token = apiKey.trim();
    if (token) {
      return token;
    }
  }
}

function createUserIDResolver(config: UserIDResolverConfig) {
  const maxAttempts = config.retryCount + 1;
  const writeLog = async (
    level: "debug" | "warn",
    message: string,
    extra: Record<string, unknown>
  ) => {
    try {
      await config.log(level, message, extra);
    } catch {}
  };
  let currentToken: string | undefined;
  let resolvedUserID: string | undefined;
  let cooldownUntil = 0;
  let tokenVersion = 0;
  let pendingResolution: Promise<string> | undefined;

  const currentUserID = () => resolvedUserID ?? UNKNOWN_USER_ID;

  const resetForToken = (token?: string) => {
    currentToken = token;
    resolvedUserID = undefined;
    cooldownUntil = 0;
    tokenVersion++;
    pendingResolution = undefined;
  };

  const completeResolution = (
    requestToken: string,
    requestTokenVersion: number,
    userID: string | undefined
  ) => {
    if (requestTokenVersion !== tokenVersion || requestToken !== currentToken) {
      return currentUserID();
    }
    resolvedUserID = userID;
    cooldownUntil = userID === undefined ? Date.now() + config.cooldownMs : 0;
    return currentUserID();
  };

  const fetchUserIDWithRetry = async (
    token: string
  ): Promise<string | undefined> => {
    for (let attemptNumber = 1; attemptNumber <= maxAttempts; attemptNumber++) {
      await writeLog("debug", "user ID request sent", {
        endpoint: config.endpoint,
        attempt: attemptNumber,
        maxAttempts,
      });
      let failure: string;
      try {
        const result = await queryUserByToken(token, config);
        await writeLog("debug", "user ID request returned", {
          endpoint: config.endpoint,
          attempt: attemptNumber,
          maxAttempts,
          code: result.code,
          resolved: result.userID !== undefined,
        });
        if (result.userID) {
          return result.userID;
        }
        failure = "response did not contain a valid user ID";
      } catch (error) {
        failure = error instanceof Error ? error.message : String(error);
      }
      const delayMs =
        attemptNumber < maxAttempts
          ? RETRY_BASE_DELAY_MS * 2 ** (attemptNumber - 1)
          : undefined;
      await writeLog("warn", "user ID request failed", {
        endpoint: config.endpoint,
        attempt: attemptNumber,
        maxAttempts,
        error: failure,
        retrying: delayMs !== undefined,
        ...(delayMs === undefined ? {} : { retryDelayMs: delayMs }),
      });
      if (delayMs === undefined) {
        return;
      }
      await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
    }
  };

  const resolveUserID = (providers: Config["provider"]): Promise<string> => {
    const token = apiKeyFromProviders(providers);
    if (!token) {
      if (currentToken !== undefined) {
        resetForToken();
      }
      return Promise.resolve(UNKNOWN_USER_ID);
    }

    if (token !== currentToken) {
      resetForToken(token);
    }

    if (resolvedUserID !== undefined) {
      return Promise.resolve(resolvedUserID);
    }
    if (Date.now() < cooldownUntil) {
      return Promise.resolve(UNKNOWN_USER_ID);
    }
    if (pendingResolution) {
      return pendingResolution;
    }

    const requestTokenVersion = tokenVersion;
    const request = fetchUserIDWithRetry(token)
      .then((userID) => completeResolution(token, requestTokenVersion, userID))
      .catch(() => completeResolution(token, requestTokenVersion, undefined));

    pendingResolution = request;
    request
      .finally(() => {
        if (pendingResolution === request) {
          pendingResolution = undefined;
        }
      })
      .catch(() => {});
    return request;
  };

  return resolveUserID;
}

function createUserIDManager(
  config: UserIDManagerConfig,
  ctx: UserIDManagerContext,
  baseCommonAttrs: CommonAttrs
) {
  const resolveUserID = createUserIDResolver({
    authHeader: config.userIDAuthHeader,
    endpoint: config.userIDEndpoint,
    log: ctx.log,
    timeoutMs: config.userIDTimeout,
    retryCount: config.userIDRetryCount,
    cooldownMs: config.userIDCooldown,
  });
  let configuredProviders: Config["provider"];

  const updateCommonAttrs = async () => {
    if (isResolvedUserID(ctx.commonAttrs[USER_ID])) {
      return;
    }
    const userID = await resolveUserID(configuredProviders);
    if (ctx.commonAttrs[USER_ID] === userID) {
      return;
    }
    ctx.commonAttrs = {
      ...baseCommonAttrs,
      [USER_ID]: userID,
    };
    await ctx.log("debug", "user ID updated", {
      resolved: userID !== UNKNOWN_USER_ID,
    });
  };

  const configure = async (nextProviders: Config["provider"]) => {
    configuredProviders = nextProviders;
    if (config.userIDEnabled) {
      await updateCommonAttrs();
    }
  };

  const refreshInBackground = () => {
    if (!config.userIDEnabled) {
      return;
    }
    updateCommonAttrs().catch((error) =>
      ctx
        .log("warn", "user ID update failed", {
          error: error instanceof Error ? error.message : String(error),
        })
        .catch(() => {})
    );
  };

  return {
    configure,
    refreshInBackground,
  };
}

export { createUserIDManager };
