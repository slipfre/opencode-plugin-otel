#!/usr/bin/env python3
"""Export OpenCode Langfuse sessions as post-training trajectories.

The script uses Langfuse's public REST API and has no third-party Python
dependencies. Configure it with LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, and
LANGFUSE_SECRET_KEY, or pass the equivalent command-line options.
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
LLM_OBSERVATION_TYPES = {"GENERATION", "SPAN"}


class LangfuseApiError(RuntimeError):
    """Raised when a Langfuse public API request fails."""


class SessionExportError(RuntimeError):
    """Raised when a session does not contain the requested export data."""


class LangfuseClient:
    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        *,
        timeout: float = 30.0,
        retries: int = 3,
    ) -> None:
        normalized_host = host.rstrip("/")
        if normalized_host.endswith("/api/public"):
            self.api_base = normalized_host
        else:
            self.api_base = f"{normalized_host}/api/public"

        credentials = f"{public_key}:{secret_key}".encode("utf-8")
        encoded_credentials = base64.b64encode(credentials).decode("ascii")
        self.authorization = f"Basic {encoded_credentials}"
        self.timeout = timeout
        self.retries = retries

    def _get(
        self,
        path: str,
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> dict[str, Any]:
        query = urlencode(params or {}, doseq=True)
        url = f"{self.api_base}{path}"
        if query:
            url = f"{url}?{query}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": self.authorization,
                "User-Agent": "langfuse-session-llm-io-export/1.0",
            },
            method="GET",
        )

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise LangfuseApiError(
                        f"GET {path} returned a non-object JSON response"
                    )
                return payload
            except HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if (
                    error.code in RETRYABLE_HTTP_STATUS_CODES
                    and attempt < self.retries
                ):
                    delay = _retry_delay(error.headers.get("Retry-After"), attempt)
                    time.sleep(delay)
                    continue
                detail = _api_error_detail(body)
                raise LangfuseApiError(
                    f"GET {path} failed with HTTP {error.code}: {detail}"
                ) from error
            except URLError as error:
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise LangfuseApiError(
                    f"GET {path} failed: {error.reason}"
                ) from error
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise LangfuseApiError(
                    f"GET {path} returned invalid JSON: {error}"
                ) from error

        raise AssertionError("unreachable")

    def list_sessions(
        self,
        *,
        page_size: int = 100,
        from_timestamp: str | None = None,
        to_timestamp: str | None = None,
        environments: Sequence[str] = (),
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        seen_session_ids: set[str] = set()
        page = 1

        while True:
            params: list[tuple[str, Any]] = [
                ("page", page),
                ("limit", page_size),
            ]
            if from_timestamp:
                params.append(("fromTimestamp", from_timestamp))
            if to_timestamp:
                params.append(("toTimestamp", to_timestamp))
            params.extend(("environment", value) for value in environments)

            payload = self._get("/sessions", params)
            page_sessions = _require_list(payload, "data", "/sessions")
            for session in page_sessions:
                if not isinstance(session, dict) or not isinstance(
                    session.get("id"), str
                ):
                    raise LangfuseApiError(
                        "GET /sessions returned a session without a string id"
                    )
                current_session_id = session["id"]
                if session_id is not None and current_session_id != session_id:
                    continue
                if current_session_id not in seen_session_ids:
                    seen_session_ids.add(current_session_id)
                    sessions.append(session)
                    if session_id is not None:
                        return sessions

            meta = payload.get("meta")
            total_pages = meta.get("totalPages") if isinstance(meta, dict) else None
            if isinstance(total_pages, int):
                if page >= total_pages:
                    break
            elif len(page_sessions) < page_size:
                break

            if not page_sessions:
                break
            page += 1

        return sessions

    def get_latest_trace(self, session_id: str) -> dict[str, Any] | None:
        payload = self._get(
            "/traces",
            {
                "sessionId": session_id,
                "orderBy": "timestamp.desc",
                "page": 1,
                "limit": 1,
            },
        )
        traces = _require_list(payload, "data", "/traces")
        if not traces:
            return None
        trace = traces[0]
        if not isinstance(trace, dict) or not isinstance(trace.get("id"), str):
            raise LangfuseApiError(
                "GET /traces returned a trace without a string id"
            )
        return trace

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        return self._get(f"/traces/{quote(trace_id, safe='')}")


@dataclass(frozen=True)
class ExportResult:
    total_sessions: int
    exported_sessions: int
    skipped_sessions: int
    failed_sessions: int


def export_all_sessions(
    client: LangfuseClient,
    output_dir: Path,
    *,
    observation_name: str = "opencode.llm",
    page_size: int = 100,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
    environments: Sequence[str] = (),
    session_id: str | None = None,
    user_id: str | None = None,
    fail_fast: bool = False,
) -> ExportResult:
    sessions = client.list_sessions(
        page_size=page_size,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        environments=environments,
        session_id=session_id,
    )
    print(f"Found {len(sessions)} session(s).")

    exported = 0
    skipped = 0
    failed = 0

    for index, session in enumerate(sessions, start=1):
        session_id = session["id"]
        prefix = f"[{index}/{len(sessions)}] {session_id}"
        try:
            latest_trace = client.get_latest_trace(session_id)
            if latest_trace is None:
                skipped += 1
                print(f"{prefix}: skipped (no trace)", file=sys.stderr)
                continue

            trace_id = latest_trace["id"]
            trace = client.get_trace(trace_id)
            observations = _require_list(
                trace, "observations", f"/traces/{trace_id}"
            )
            observation = find_latest_llm_observation(
                observations, observation_name=observation_name
            )
            if observation is None:
                skipped += 1
                print(
                    f"{prefix}: skipped (latest trace {trace_id} has no "
                    f"{observation_name!r} GENERATION/SPAN)",
                    file=sys.stderr,
                )
                continue

            if user_id is not None and not observation_matches_user_id(
                observation, user_id
            ):
                skipped += 1
                print(
                    f"{prefix}: skipped (selected {observation_name!r} "
                    "does not match --user-id)",
                    file=sys.stderr,
                )
                continue

            safe_session_name = safe_session_directory_name(session_id)
            legacy_session_dir = output_dir / safe_session_name
            session_history = merge_observation_input_output(
                observation.get("input"),
                observation.get("output"),
            )
            model, provider = extract_observation_model_and_provider(observation)
            trajectory = build_training_trajectory(
                session_id,
                session_history,
                model=model,
                provider=provider,
            )
            session_path = output_dir / f"{safe_session_name}.json"
            write_json_atomic(session_path, trajectory)
            remove_legacy_session_export(legacy_session_dir, safe_session_name)
            exported += 1
            print(
                f"{prefix}: exported trace={trace_id} "
                f"observation={observation.get('id')} -> {session_path}"
            )
        except (LangfuseApiError, SessionExportError, OSError) as error:
            failed += 1
            print(f"{prefix}: failed ({error})", file=sys.stderr)
            if fail_fast:
                raise

    return ExportResult(
        total_sessions=len(sessions),
        exported_sessions=exported,
        skipped_sessions=skipped,
        failed_sessions=failed,
    )


def find_latest_llm_observation(
    observations: Iterable[Any],
    *,
    observation_name: str = "opencode.llm",
) -> dict[str, Any] | None:
    candidates = [
        observation
        for observation in observations
        if isinstance(observation, dict)
        and observation.get("name") == observation_name
        and observation.get("type") in LLM_OBSERVATION_TYPES
    ]
    if not candidates:
        return None
    return max(candidates, key=_observation_order_key)


def observation_matches_user_id(
    observation: Mapping[str, Any], user_id: str
) -> bool:
    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        return False
    attributes = metadata.get("attributes")
    if not isinstance(attributes, dict):
        return False
    return attributes.get("langfuse.user.id") == user_id


def merge_observation_input_output(
    input_value: Any, output_value: Any
) -> dict[str, Any]:
    if isinstance(input_value, dict) and isinstance(
        input_value.get("messages"), list
    ):
        merged = dict(input_value)
        input_messages = list(input_value["messages"])
    elif isinstance(input_value, list):
        input_messages = list(input_value)
        merged = {"messages": input_messages}
    else:
        raise SessionExportError(
            "observation input must be an object containing a messages list"
        )

    if isinstance(output_value, dict):
        output_messages = output_value.get("messages")
    else:
        output_messages = output_value
    if not isinstance(output_messages, list):
        raise SessionExportError(
            "observation output must be a message list or contain a messages list"
        )

    messages = [*input_messages, *output_messages]
    if not all(isinstance(message, dict) for message in messages):
        raise SessionExportError("observation messages must be JSON objects")

    overlap = 0
    for size in range(min(len(input_messages), len(output_messages)), 0, -1):
        if input_messages[-size:] == output_messages[:size]:
            overlap = size
            break

    merged["messages"] = [*input_messages, *output_messages[overlap:]]
    return merged


def extract_observation_model_and_provider(
    observation: Mapping[str, Any],
) -> tuple[str, str]:
    model = observation.get("model")
    provider: Any = None
    metadata = observation.get("metadata")
    if isinstance(metadata, dict):
        attributes = metadata.get("attributes")
        if isinstance(attributes, dict):
            provider = attributes.get("llm.provider") or attributes.get(
                "llm.system"
            )
        if not provider:
            provider = (
                metadata.get("llm.provider")
                or metadata.get("llm.system")
                or metadata.get("provider")
            )

    if not isinstance(model, str) or not model:
        raise SessionExportError("observation model must not be empty")
    if not isinstance(provider, str) or not provider:
        raise SessionExportError(
            "observation metadata must contain llm.provider or llm.system"
        )
    return model, provider


def build_training_trajectory(
    session_id: str,
    session_history: Mapping[str, Any],
    *,
    model: str,
    provider: str,
) -> dict[str, Any]:
    if not session_id:
        raise SessionExportError("session id must not be empty")
    if not isinstance(model, str) or not model:
        raise SessionExportError("observation model must not be empty")
    if not isinstance(provider, str) or not provider:
        raise SessionExportError("observation provider must not be empty")

    messages = session_history.get("messages")
    tools = session_history.get("tools", [])
    if not isinstance(messages, list):
        raise SessionExportError("session history must contain a messages list")
    if not isinstance(tools, list):
        raise SessionExportError("session history tools must be a list")

    return {
        "session_id": session_id,
        "model": model,
        "provider": provider,
        "messages": [_normalize_training_message(message) for message in messages],
        "tools": tools,
    }


def _normalize_training_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise SessionExportError("session messages must be JSON objects")

    role = message.get("role")
    if role in {"system", "user"}:
        return {"role": role, "content": message.get("content")}
    if role == "tool":
        tool_call_id = message.get("tool_call_id", message.get("toolCallId"))
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise SessionExportError("tool message must contain a tool call id")
        return {
            "role": "tool",
            "content": message.get("content"),
            "tool_call_id": tool_call_id,
        }
    if role != "assistant":
        raise SessionExportError(f"unsupported message role: {role!r}")

    return _normalize_assistant_message(message)


def _normalize_assistant_message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    existing_reasoning = message.get("reasoning_content")
    if existing_reasoning is not None:
        if not isinstance(existing_reasoning, str):
            raise SessionExportError("assistant reasoning_content must be a string")
        if existing_reasoning:
            reasoning_parts.append(existing_reasoning)

    existing_tool_calls = message.get("tool_calls", [])
    if existing_tool_calls is None:
        existing_tool_calls = []
    if not isinstance(existing_tool_calls, list):
        raise SessionExportError("assistant tool_calls must be a list")
    tool_calls.extend(
        _normalize_existing_tool_call(tool_call) for tool_call in existing_tool_calls
    )

    if isinstance(content, list):
        visible_content: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                raise SessionExportError("assistant content parts must be JSON objects")
            part_type = part.get("type")
            if part_type == "reasoning":
                text = part.get("text")
                if not isinstance(text, str):
                    raise SessionExportError("reasoning content must contain text")
                if text:
                    reasoning_parts.append(text)
            elif part_type == "tool_use":
                tool_calls.append(
                    _build_training_tool_call(
                        part.get("id"),
                        part.get("name"),
                        part.get("arguments"),
                    )
                )
            else:
                visible_content.append(part)
        normalized_content: Any = visible_content or None
    elif content is None or isinstance(content, str):
        normalized_content = content
    else:
        raise SessionExportError(
            "assistant content must be a string, list, or null"
        )

    normalized: dict[str, Any] = {
        "role": "assistant",
        "content": normalized_content,
    }
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    if reasoning_parts:
        normalized["reasoning_content"] = "\n".join(reasoning_parts)
    return normalized


def _normalize_existing_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise SessionExportError("assistant tool calls must be JSON objects")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise SessionExportError("assistant tool call must contain function data")
    return _build_training_tool_call(
        tool_call.get("id"),
        function.get("name"),
        function.get("arguments"),
    )


def _build_training_tool_call(
    tool_call_id: Any,
    name: Any,
    arguments: Any,
) -> dict[str, Any]:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise SessionExportError("assistant tool call must contain an id")
    if not isinstance(name, str) or not name:
        raise SessionExportError("assistant tool call must contain a function name")
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise SessionExportError(
                "assistant tool call arguments must be JSON serializable"
            ) from error
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def safe_session_directory_name(session_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", session_id):
        return session_id

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id).strip("._")
    sanitized = sanitized[:100] or "session"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}--{digest}"


def remove_legacy_session_export(
    legacy_session_dir: Path,
    safe_session_name: str,
) -> None:
    if not legacy_session_dir.is_dir() or legacy_session_dir.is_symlink():
        return

    for filename in (
        "input.json",
        "output.json",
        f"{safe_session_name}.json",
    ):
        (legacy_session_dir / filename).unlink(missing_ok=True)

    try:
        legacy_session_dir.rmdir()
    except OSError as error:
        if error.errno not in {errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST}:
            raise


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            output_file.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _observation_order_key(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _parse_timestamp(observation.get("startTime")),
        _parse_timestamp(observation.get("createdAt")),
        str(observation.get("id", "")),
    )


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_list(
    payload: Mapping[str, Any], key: str, endpoint: str
) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise LangfuseApiError(
            f"GET {endpoint} returned an invalid {key!r} field"
        )
    return value


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return float(2**attempt)


def _api_error_detail(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:500] or "empty response body"

    if isinstance(payload, dict):
        for key in ("message", "error"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(payload, ensure_ascii=False)[:500]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one post-training trajectory from the last opencode.llm "
            "observation in every Langfuse session."
        )
    )
    parser.add_argument(
        "--host",
        default=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        help="Langfuse base URL (default: LANGFUSE_HOST or http://localhost:3000)",
    )
    parser.add_argument(
        "--public-key",
        default=os.getenv("LANGFUSE_PUBLIC_KEY"),
        help="Langfuse project public key (default: LANGFUSE_PUBLIC_KEY)",
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("LANGFUSE_SECRET_KEY"),
        help="Langfuse project secret key (default: LANGFUSE_SECRET_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("langfuse_session_exports"),
        help="Export root directory (default: ./langfuse_session_exports)",
    )
    parser.add_argument(
        "--observation-name",
        default="opencode.llm",
        help="Observation name to export (default: opencode.llm)",
    )
    parser.add_argument(
        "--session-id",
        help="Only export the session with this exact session ID",
    )
    parser.add_argument(
        "--user-id",
        help=(
            "Only export trajectories whose selected observation has this exact "
            "langfuse.user.id span attribute"
        ),
    )
    parser.add_argument(
        "--from-timestamp",
        help="Only include sessions created on/after this ISO 8601 timestamp",
    )
    parser.add_argument(
        "--to-timestamp",
        help="Only include sessions created before this ISO 8601 timestamp",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        help="Only include this environment; may be repeated",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        choices=range(1, 101),
        metavar="1..100",
        help="Number of sessions fetched per request (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for connection failures, HTTP 429, and HTTP 5xx (default: 3)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first session export failure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.public_key:
        parser.error("--public-key or LANGFUSE_PUBLIC_KEY is required")
    if not args.secret_key:
        parser.error("--secret-key or LANGFUSE_SECRET_KEY is required")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
    if args.session_id == "":
        parser.error("--session-id must not be empty")
    if args.user_id == "":
        parser.error("--user-id must not be empty")

    client = LangfuseClient(
        args.host,
        args.public_key,
        args.secret_key,
        timeout=args.timeout,
        retries=args.retries,
    )
    try:
        result = export_all_sessions(
            client,
            args.output_dir,
            observation_name=args.observation_name,
            page_size=args.page_size,
            from_timestamp=args.from_timestamp,
            to_timestamp=args.to_timestamp,
            environments=args.environment,
            session_id=args.session_id,
            user_id=args.user_id,
            fail_fast=args.fail_fast,
        )
    except (LangfuseApiError, SessionExportError, OSError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1

    print(
        "Done: "
        f"total={result.total_sessions}, "
        f"exported={result.exported_sessions}, "
        f"skipped={result.skipped_sessions}, "
        f"failed={result.failed_sessions}"
    )
    return 1 if result.failed_sessions else 0


if __name__ == "__main__":
    raise SystemExit(main())
