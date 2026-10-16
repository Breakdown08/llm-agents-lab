"""Проверки настоящего графа и middleware без сетевых запросов к модели."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from llm_agents_lab.archaeologist import (
    ModuleReports,
    create_archaeologist,
    print_result,
)


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


class ArchaeologistTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_agent(self, messages, *, reverse=False):
        agent, reports = create_archaeologist(
            self.root, ScriptedModel(messages=iter(messages)),
            reverse_middleware=reverse,
        )
        return agent.invoke({"messages": [("user", "Исследуйте каталог")]}), reports

    def test_retry_small_large_and_grep(self):
        (self.root / "small.py").write_text("def small():\n    return 1\n")
        (self.root / "large.py").write_text("# padding\n" * 1000 + "def large(): pass\n")
        result, reports = self.run_agent([
            call("write_todos", {"todos": [{"content": "Разобрать модули", "status": "in_progress"}]}, "plan"),
            call("glob_search", {"pattern": "**/*.py"}, "find"),
            call("module_report", {"path": "/small.py"}, "small"),
            call("module_report", {"path": "/large.py"}, "large"),
            call("grep_search", {"pattern": "^def ", "path": "/", "include": "large.py", "output_mode": "content"}, "grep"),
            call("write_todos", {"todos": [{"content": "Разобрать модули", "status": "completed"}]}, "done"),
            AIMessage(content="Модули исследованы."),
        ])
        messages = {m.tool_call_id: m for m in result["messages"] if m.type == "tool"}
        self.assertEqual(reports.attempts, {"/small.py": 2, "/large.py": 2})
        self.assertEqual(messages["small"].status, "success")
        self.assertIn("2 строк", messages["small"].text)
        self.assertEqual(messages["large"].status, "error")
        self.assertIn("1001 строк", messages["large"].text)
        self.assertIn("grep_search", messages["large"].text)
        self.assertIn("def large", messages["grep"].text)
        self.assertEqual(result["todos"][0]["status"], "completed")
        output = StringIO()
        with redirect_stdout(output):
            print_result(result, reports)
        self.assertIn("glob_search → module_report → module_report → grep_search", output.getvalue())
        self.assertIn("status=error", output.getvalue())

    def test_reverse_order_swallows_error_before_retry(self):
        (self.root / "small.py").write_text("pass\n")
        (self.root / "large.py").write_text("pass\n" * 1001)
        result, reports = self.run_agent([
            call("module_report", {"path": "/small.py"}, "small"),
            call("module_report", {"path": "/large.py"}, "large"),
            AIMessage(content="Хранилище недоступно."),
        ], reverse=True)
        self.assertEqual(reports.attempts, {"/small.py": 1, "/large.py": 1})
        errors = [m for m in result["messages"] if m.type == "tool"]
        self.assertEqual(len(errors), 2)
        for message in errors:
            self.assertEqual(message.status, "error")
            self.assertIn("Повторять вызов не нужно", message.text)

    def test_empty_directory_and_empty_todos(self):
        result, reports = self.run_agent([
            call("glob_search", {"pattern": "**/*"}, "empty"),
            AIMessage(content="Каталог пуст, искать нечего."),
        ])
        self.assertIn("No files found", result["messages"][2].text)
        self.assertFalse(reports.attempts)
        output = StringIO()
        with redirect_stdout(output):
            print_result(result, reports)
        self.assertIn("Список задач пуст", output.getvalue())
        self.assertIn("искать нечего", output.getvalue())

    def test_line_count_boundary_and_independent_file_state(self):
        reports = ModuleReports(self.root)
        for name, content, count in [("empty.py", "", 0), ("limit.py", "pass\n" * 999 + "pass", 1000)]:
            (self.root / name).write_text(content)
            with self.assertRaises(ConnectionError):
                reports.module_report(name)
            self.assertIn(f"{count} строк", reports.module_report("/" + name))
            self.assertEqual(reports.attempts["/" + name], 2)

    def test_cannot_read_outside_root_or_via_symlink(self):
        reports = ModuleReports(self.root)
        with self.assertRaises(ValueError):
            reports.module_report("../outside.py")
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            reports.module_report("escape/outside.py")
        self.assertFalse(reports.attempts)


if __name__ == "__main__":
    unittest.main()
