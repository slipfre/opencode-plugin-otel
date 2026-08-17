const UNKNOWN_VERSION = "unknown";

type ClientWithTransport = {
  _client?: {
    get(options: { url: string }): PromiseLike<{ data?: unknown }>;
  };
};

async function resolveOpenCodeVersion(client: unknown) {
  try {
    const transport = (client as ClientWithTransport)._client;
    if (!transport) {
      return UNKNOWN_VERSION;
    }
    const result = await transport.get({ url: "/global/health" });
    const data = result.data;
    if (typeof data !== "object" || data === null || !("version" in data)) {
      return UNKNOWN_VERSION;
    }
    const version = data.version;
    return typeof version === "string" && version.trim()
      ? version.trim()
      : UNKNOWN_VERSION;
  } catch {
    return UNKNOWN_VERSION;
  }
}

export { resolveOpenCodeVersion };
