import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from solidlsp.ls_config import Language

from serena.cli import AgentFriendlyToolCommands, _tool_parse_language_values


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
        self.execute_task_calls = 0

        class _Backend:
            @staticmethod
            def is_lsp() -> bool:
                return True

        self._backend = _Backend()

        class _ProjectConfig:
            def __init__(self):
                self.languages = [Language.PYTHON]

        class _Project:
            def __init__(self, root: str):
                self.project_root = root
                self.project_config = _ProjectConfig()

        self._project = _Project(project_root) if project_root is not None else None

    def get_tool_by_name(self, tool_name: str) -> _FakeTool:
        return self._tools[tool_name]

    def get_active_project(self):
        return self._project

    def get_language_backend(self):
        return self._backend

    def reset_language_server_manager(self) -> None:
        return None

    def execute_task(self, task, name: str | None = None, logged: bool = True, timeout: float | None = None):
        _ = name
        _ = logged
        _ = timeout
        self.execute_task_calls += 1
        return task()

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


def test_tool_parse_language_values_supports_alias_and_csv() -> None:
    parsed = _tool_parse_language_values(language="javascript", languages="python,typescript")
    assert parsed == [Language.TYPESCRIPT, Language.PYTHON]


def test_tool_run_language_override_is_runtime_only(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tool = _FakeTool('[{"name":"App","kind":"Class"}]')
    fake_agent = _FakeAgent({"get_symbols_overview": fake_tool}, project_root="E:/project/immich")
    monkeypatch.setattr("serena.cli._create_tool_cli_agent", lambda project: fake_agent)

    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "get_symbols_overview",
            "--project",
            "E:/project/immich",
            "--json-args",
            '{"relative_path":"server/src/main.ts"}',
            "--auto-language-from-path",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is True
    # 自动推断 ts -> typescript，且只在内存配置生效（通过 fake project_config 断言）
    active_project = fake_agent.get_active_project()
    assert active_project is not None
    assert active_project.project_config.languages == [Language.PYTHON, Language.TYPESCRIPT]
    assert fake_agent.execute_task_calls == 1


def test_tool_run_language_override_requires_active_project(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tool = _FakeTool("[]")
    fake_agent = _FakeAgent({"get_symbols_overview": fake_tool}, project_root=None)
    monkeypatch.setattr("serena.cli._create_tool_cli_agent", lambda project: fake_agent)

    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "get_symbols_overview",
            "--json-args",
            '{"relative_path":"server/src/main.ts"}',
            "--auto-language-from-path",
            "--json",
        ],
    )

    assert result.exit_code != 0
    output_lines = [line for line in result.output.splitlines() if line.strip()]
    assert output_lines, result.output
    envelope = json.loads(output_lines[-1])
    assert envelope["ok"] is False
    assert "requires an active project" in envelope["error"]


def test_tool_run_auto_language_from_path_real_immich_project(cli_runner: CliRunner) -> None:
    immich_root = Path("E:/project/immich")
    immich_cfg = immich_root / ".serena" / "project.yml"
    main_ts = immich_root / "server" / "src" / "main.ts"

    if not immich_cfg.is_file() or not main_ts.is_file():
        pytest.skip("Local immich project fixture not available")

    before_cfg = immich_cfg.read_text(encoding="utf-8")

    result = cli_runner.invoke(
        AgentFriendlyToolCommands.run,
        [
            "get_symbols_overview",
            "--project",
            str(immich_root),
            "--json-args",
            '{"relative_path":"server/src/main.ts"}',
            "--auto-language-from-path",
            "--json",
            "--max-chars",
            "12000",
        ],
    )

    after_cfg = immich_cfg.read_text(encoding="utf-8")

    assert before_cfg == after_cfg, "project.yml must remain unchanged (runtime-only language override)"
    assert result.exit_code == 0, result.output
    envelope = _parse_single_json_output(result.output)
    assert envelope["ok"] is True
    assert envelope["tool"] == "get_symbols_overview"
    assert envelope["project"] == str(immich_root)
    assert envelope["result"]
