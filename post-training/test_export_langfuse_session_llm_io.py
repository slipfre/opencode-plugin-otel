from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).parents[1] / "export_langfuse_session_llm_io.py"
SPEC = importlib.util.spec_from_file_location("export_langfuse_session_llm_io", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


class FakeClient:
    def __init__(self) -> None:
        self.session_list_kwargs: dict[str, Any] | None = None

    def list_sessions(self, **kwargs: Any) -> list[dict[str, str]]:
        self.session_list_kwargs = kwargs
        sessions = [
            {"id": "session/a"},
            {"id": "session-no-trace"},
            {"id": "session-no-llm"},
        ]
        session_id = kwargs.get("session_id")
        return [
            session
            for session in sessions
            if session_id is None or session["id"] == session_id
        ]

    def get_latest_trace(self, session_id: str) -> dict[str, str] | None:
        if session_id == "session-no-trace":
            return None
        return {"id": f"trace-{session_id}"}

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        if trace_id == "trace-session-no-llm":
            return {"observations": [{"name": "opencode.tool.read", "type": "SPAN"}]}
        return {
            "observations": [
                {
                    "id": "old",
                    "name": "opencode.llm",
                    "type": "GENERATION",
                    "startTime": "2026-01-01T00:00:00Z",
                    "input": {"old": True},
                    "output": {"old": True},
                },
                {
                    "id": "new",
                    "name": "opencode.llm",
                    "type": "GENERATION",
                    "startTime": "2026-01-01T00:00:02Z",
                    "model": "glm-5",
                    "metadata": {
                        "attributes": {
                            "langfuse.user.id": "user-123",
                            "llm.provider": "alibaba-cn",
                        }
                    },
                    "input": {
                        "messages": [{"role": "user", "content": "你好"}],
                        "tools": [{"type": "function", "name": "bash"}],
                    },
                    "output": [{"role": "assistant", "content": "你好"}],
                },
            ]
        }


class PagingClient(exporter.LangfuseClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _get(self, path: str, params: Any = None) -> dict[str, Any]:
        self.calls.append((path, params))
        page = dict(params)["page"]
        payloads = {
            1: {
                "data": [{"id": "a"}, {"id": "b"}],
                "meta": {"totalPages": 2},
            },
            2: {
                "data": [{"id": "b"}, {"id": "c"}],
                "meta": {"totalPages": 2},
            },
        }
        return payloads[page]


class ExportSessionLlmIoTests(unittest.TestCase):
    def test_session_list_paginates_and_deduplicates(self) -> None:
        client = PagingClient()

        sessions = client.list_sessions(
            page_size=2,
            from_timestamp="2026-01-01T00:00:00Z",
            environments=["production", "staging"],
        )

        self.assertEqual([session["id"] for session in sessions], ["a", "b", "c"])
        self.assertEqual(len(client.calls), 2)
        first_params = client.calls[0][1]
        self.assertIn(("fromTimestamp", "2026-01-01T00:00:00Z"), first_params)
        self.assertIn(("environment", "production"), first_params)
        self.assertIn(("environment", "staging"), first_params)

    def test_session_list_filters_by_id_and_stops_after_match(self) -> None:
        client = PagingClient()

        sessions = client.list_sessions(page_size=2, session_id="b")

        self.assertEqual([session["id"] for session in sessions], ["b"])
        self.assertEqual(len(client.calls), 1)

    def test_latest_llm_observation_uses_start_time(self) -> None:
        observations = [
            {
                "id": "later-created-but-earlier-start",
                "name": "opencode.llm",
                "type": "SPAN",
                "startTime": "2026-01-01T00:00:01Z",
                "createdAt": "2026-01-01T00:00:05Z",
            },
            {
                "id": "latest-start",
                "name": "opencode.llm",
                "type": "GENERATION",
                "startTime": "2026-01-01T00:00:02Z",
                "createdAt": "2026-01-01T00:00:03Z",
            },
            {
                "id": "wrong-type",
                "name": "opencode.llm",
                "type": "EVENT",
                "startTime": "2026-01-01T00:00:04Z",
            },
        ]

        observation = exporter.find_latest_llm_observation(observations)

        self.assertIsNotNone(observation)
        self.assertEqual(observation["id"], "latest-start")

    def test_export_writes_complete_session_history_to_session_named_json(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            session_id = "session/a"
            safe_session_name = exporter.safe_session_directory_name(session_id)
            legacy_session_dir = output_dir / safe_session_name
            legacy_session_dir.mkdir()
            (legacy_session_dir / "input.json").write_text("legacy input")
            (legacy_session_dir / "output.json").write_text("legacy output")

            result = exporter.export_all_sessions(client, output_dir)

            self.assertEqual(result.total_sessions, 3)
            self.assertEqual(result.exported_sessions, 1)
            self.assertEqual(result.skipped_sessions, 2)
            self.assertEqual(result.failed_sessions, 0)
            with (output_dir / f"{safe_session_name}.json").open(encoding="utf-8") as session_file:
                self.assertEqual(
                    json.load(session_file),
                    {
                        "session_id": session_id,
                        "model": "glm-5",
                        "provider": "alibaba-cn",
                        "messages": [
                            {"role": "user", "content": "你好"},
                            {"role": "assistant", "content": "你好"},
                        ],
                        "tools": [{"type": "function", "name": "bash"}],
                    },
                )
            self.assertFalse(legacy_session_dir.exists())

    def test_export_filters_by_session_id_and_user_id(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            result = exporter.export_all_sessions(
                client,
                output_dir,
                session_id="session/a",
                user_id="user-123",
            )

            self.assertEqual(result.total_sessions, 1)
            self.assertEqual(result.exported_sessions, 1)
            self.assertEqual(result.skipped_sessions, 0)
            self.assertEqual(result.failed_sessions, 0)
            self.assertEqual(client.session_list_kwargs["session_id"], "session/a")

    def test_export_skips_selected_observation_with_different_user_id(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            result = exporter.export_all_sessions(
                client,
                output_dir,
                session_id="session/a",
                user_id="different-user",
            )

            self.assertEqual(result.total_sessions, 1)
            self.assertEqual(result.exported_sessions, 0)
            self.assertEqual(result.skipped_sessions, 1)
            self.assertEqual(result.failed_sessions, 0)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_observation_user_id_uses_nested_span_attribute(self) -> None:
        observation = {
            "metadata": {
                "attributes": {"langfuse.user.id": "user-123"},
            },
        }

        self.assertTrue(
            exporter.observation_matches_user_id(observation, "user-123")
        )
        self.assertFalse(
            exporter.observation_matches_user_id(observation, "USER-123")
        )
        self.assertFalse(exporter.observation_matches_user_id({}, "user-123"))

    def test_argument_parser_accepts_session_and_user_filters(self) -> None:
        args = exporter.build_argument_parser().parse_args(
            ["--session-id", "session/a", "--user-id", "user-123"]
        )

        self.assertEqual(args.session_id, "session/a")
        self.assertEqual(args.user_id, "user-123")

    def test_merge_messages_does_not_duplicate_existing_output_suffix(self) -> None:
        assistant = {"role": "assistant", "content": "done"}

        merged = exporter.merge_observation_input_output(
            {"messages": [{"role": "user", "content": "go"}, assistant]},
            [assistant],
        )

        self.assertEqual(
            merged["messages"],
            [{"role": "user", "content": "go"}, assistant],
        )

    def test_training_trajectory_uses_raw_session_id_and_normalizes_messages(self) -> None:
        history = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Think"},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "bash",
                            "arguments": {"command": "pwd"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": "/workspace\n",
                    "toolCallId": "tool-1",
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Done"},
                        {"type": "text", "text": "Finished"},
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "model": "must-not-be-exported",
            "provider": "must-not-be-exported",
        }

        trajectory = exporter.build_training_trajectory(
            "raw/session-id",
            history,
            model="glm-5",
            provider="alibaba-cn",
        )

        self.assertEqual(
            set(trajectory),
            {"session_id", "model", "provider", "messages", "tools"},
        )
        self.assertEqual(trajectory["session_id"], "raw/session-id")
        self.assertEqual(trajectory["model"], "glm-5")
        self.assertEqual(trajectory["provider"], "alibaba-cn")
        self.assertEqual(trajectory["tools"], history["tools"])
        self.assertEqual(
            trajectory["messages"],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": "{\"command\": \"pwd\"}",
                            },
                        }
                    ],
                    "reasoning_content": "Think",
                },
                {
                    "role": "tool",
                    "content": "/workspace\n",
                    "tool_call_id": "tool-1",
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Finished"}],
                    "reasoning_content": "Done",
                },
            ],
        )

    def test_unsafe_session_id_gets_stable_safe_directory(self) -> None:
        value = exporter.safe_session_directory_name("a/b:中文")

        self.assertRegex(value, r"^a_b--[0-9a-f]{12}$")
        self.assertEqual(value, exporter.safe_session_directory_name("a/b:中文"))


if __name__ == "__main__":
    unittest.main()
