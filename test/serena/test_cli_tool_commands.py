import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from serena.cli import AgentFriendlyToolCommands


class _FakeTool:
    def __init__(self, return_value: str):
        self.return_value = return_value
        self.last_kwargs = None

    def apply_ex(self, log_call: bool = True, catch_exceptions: bool = True, **kwargs) -> str:  # noqa: FBT001, FBT002
        self.last_kwargs = kwargs
        return self.return_value


class _FakeAgent:
    def __init__(self, tools: dict[str, _FakeTool], project_root: str | None = "E:/fake"):
        self._tools = tools
        self._project_root = project_root

    def get_tool_by_name(self, tool_name: str) -> _FakeTool:
        return self._tools[tool_name]

    def get_active_project(self):
        if self._project_root is None:
            return None
        return SimpleNamespace(project_root=self._project_root)

    def on_shutdown(self, timeout: float = 1.0) -> None:
        _ = timeout


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def _parse_single_json_output(output: str) -> dict:
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 1, output
    return json.loads(lines[0])


def test_tool_list_json(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(AgentFriendlyToolCommands.list, ["--json"])
    assert result.exit_code == 0, result.output
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is True
    assert envelope["tool"] is None
    assert isinstance(envelope["result"], list)
    assert any(item["name"] == "find_symbol" for item in envelope["result"])


def test_tool_schema_find_symbol(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(AgentFriendlyToolCommands.schema, ["find_symbol", "--json"])
    assert result.exit_code == 0, result.output
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is True
    assert envelope["tool"] == "find_symbol"
    assert envelope["result"]["name"] == "find_symbol"
    assert "parameters" in envelope["result"]
    assert "properties" in envelope["result"]["parameters"]


def test_tool_run_requires_allow_write(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "write_memory",
            "--json-args",
            '{"memory_name":"m","content":"x"}',
            "--json",
        ],
    )
    assert result.exit_code != 0
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is False
    assert "--allow-write" in envelope["error"]


def test_tool_run_requires_allow_shell(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "execute_shell_command",
            "--json-args",
            '{"command":"echo hello"}',
            "--json",
        ],
    )
    assert result.exit_code != 0
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is False
    assert "--allow-shell" in envelope["error"]


@pytest.mark.parametrize(
    ("tool_name", "args_dict", "fake_result"),
    [
        ("find_symbol", {"name_path_pattern": "User"}, '[{"name_path":"User"}]'),
        ("get_symbols_overview", {"relative_path": "foo.py"}, '[{"name":"Foo","kind":"Class"}]'),
        ("read_file", {"relative_path": "foo.py"}, "line1\nline2"),
    ],
)
def test_tool_run_success_with_fake_agent(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    args_dict: dict,
    fake_result: str,
) -> None:
    fake_tool = _FakeTool(fake_result)
    fake_agent = _FakeAgent({tool_name: fake_tool}, project_root="E:/project/mem0")
    monkeypatch.setattr("serena.cli._create_tool_cli_agent", lambda project: fake_agent)

    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            tool_name,
            "--project",
            "E:/project/mem0",
            "--json-args",
            json.dumps(args_dict),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is True
    assert envelope["tool"] == tool_name
    assert envelope["project"] == "E:/project/mem0"
    assert fake_tool.last_kwargs == args_dict


def test_tool_run_stdin_json_with_fake_agent(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tool = _FakeTool('{"ok":true}')
    fake_agent = _FakeAgent({"find_symbol": fake_tool}, project_root="E:/project/mem0")
    monkeypatch.setattr("serena.cli._create_tool_cli_agent", lambda project: fake_agent)

    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "find_symbol",
            "--project",
            "E:/project/mem0",
            "--stdin-json",
            "--json",
        ],
        input='{"name_path_pattern":"User"}',
    )
    assert result.exit_code == 0, result.output
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is True
    assert fake_tool.last_kwargs == {"name_path_pattern": "User"}


def test_tool_run_reports_tool_error_as_non_zero(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tool = _FakeTool("Error: No active project")
    fake_agent = _FakeAgent({"read_file": fake_tool}, project_root=None)
    monkeypatch.setattr("serena.cli._create_tool_cli_agent", lambda project: fake_agent)

    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "read_file",
            "--json-args",
            '{"relative_path":"README.md"}',
            "--json",
        ],
    )
    assert result.exit_code != 0
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is False
    assert envelope["error"].startswith("Error:")
