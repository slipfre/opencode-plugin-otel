import { describe, expect, test } from "bun:test";
import { resolveOpenCodeVersion } from "../src/opencode-version.ts";

describe("resolveOpenCodeVersion", () => {
  test("returns the trimmed version from the health endpoint", async () => {
    const requests: Array<{ url: string }> = [];
    const client = {
      _client: {
        async get(options: { url: string }) {
          requests.push(options);
          return { data: { healthy: true, version: " 1.18.5 " } };
        },
      },
    };

    expect(await resolveOpenCodeVersion(client)).toBe("1.18.5");
    expect(requests).toEqual([{ url: "/global/health" }]);
  });

  test("returns unknown when the client transport is unavailable", async () => {
    expect(await resolveOpenCodeVersion({})).toBe("unknown");
  });

  test("returns unknown for an invalid health response", async () => {
    const client = {
      _client: {
        async get() {
          return { data: { healthy: true, version: " " } };
        },
      },
    };

    expect(await resolveOpenCodeVersion(client)).toBe("unknown");
  });

  test("returns unknown when the health request fails", async () => {
    const client = {
      _client: {
        async get(): Promise<never> {
          throw new Error("unavailable");
        },
      },
    };

    expect(await resolveOpenCodeVersion(client)).toBe("unknown");
  });
});
