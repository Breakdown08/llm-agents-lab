"""Проверки графа без сети и ключей модели: python -m unittest discover -s tests."""

import unittest
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from llm_agents_lab import main as lab_main
from llm_agents_lab.main import (
    AuditContext,
    AuditMiddleware,
    append_audit_log,
)


class ScriptedModel(BaseChatModel):
    """Заказывает заданные пакеты инструментов, затем выдаёт итог."""

    batches: list[list[str]]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-audit-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = self.calls
        self.calls += 1
        paths = self.batches[index] if index < len(self.batches) else []
        message = AIMessage(
            content="" if paths else "Все модули проверены.",
            tool_calls=[
                {
                    "name": "module_tests",
                    "args": {"path": path},
                    "id": f"call-{index}-{number}",
                    "type": "tool_call",
                }
                for number, path in enumerate(paths)
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class NoJumpAuditMiddleware(AuditMiddleware):
    def before_model(self, state, runtime):
        return super().before_model(state, runtime)


class UnnamedAuditMiddleware(AuditMiddleware):
    name = AgentMiddleware.name


class AuditMiddlewareTests(unittest.TestCase):
    def build_agent(self, model, layers):
        return create_agent(
            model=model,
            tools=[lab_main.module_tests],
            middleware=layers,
            context_schema=AuditContext,
        )

    def run_agent(self, batches, layer=None) -> tuple[ScriptedModel, dict[str, Any]]:
        model = ScriptedModel(batches=batches)
        agent = self.build_agent(model, [layer or AuditMiddleware(4)])
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Сколько тестов в модулях?"}]},
            context=AuditContext(user_id="test-user"),
        )
        self.assertTrue(
            all("user_id=test-user" in line for line in result["audit_log"])
        )
        return model, result

    def hooks(self, result):
        return [line.split("] ", 1)[1].split(":", 1)[0] for line in result["audit_log"]]

    def test_two_modules_finish_and_keep_hook_order(self):
        paths = list(lab_main.TESTS)[:2]
        model, result = self.run_agent([paths])
        self.assertEqual((model.calls, result["model_calls"], result["tool_calls"]), (2, 2, 2))
        self.assertEqual(result["messages"][-1].text, "Все модули проверены.")
        self.assertEqual(
            self.hooks(result),
            ["before_agent", "before_model", "after_model", "wrap_tool_call",
             "wrap_tool_call", "before_model", "after_model", "after_agent"],
        )
        for path in paths:
            response = lab_main.module_tests(path)
            self.assertTrue(any(
                f'args={{"path": "{path}"}}, response_length={len(response)}' in line
                for line in result["audit_log"]
            ))
        self.assertIn("model_calls=2, tool_calls=2", result["audit_log"][-1])

    def test_six_parallel_calls_all_complete_before_budget_check(self):
        model, result = self.run_agent([list(lab_main.TESTS)])
        self.assertEqual((model.calls, result["model_calls"], result["tool_calls"]), (1, 1, 6))
        self.assertEqual(sum(isinstance(m, ToolMessage) for m in result["messages"]), 6)
        self.assertIn("Бюджет", result["messages"][-1].text)
        self.assertEqual(self.hooks(result)[-2:], ["before_model", "after_agent"])

    def test_budget_at_exact_limit_stops_before_next_batch(self):
        paths = list(lab_main.TESTS)
        model, result = self.run_agent([paths[:2], paths[2:4], paths[4:]])
        self.assertEqual((model.calls, result["tool_calls"]), (2, 4))
        self.assertIn("Бюджет", result["messages"][-1].text)

    def test_without_hook_config_model_gets_final_turn(self):
        model, result = self.run_agent([list(lab_main.TESTS)], NoJumpAuditMiddleware(4))
        self.assertEqual((model.calls, result["model_calls"], result["tool_calls"]), (2, 2, 6))
        self.assertTrue(any("Бюджет" in m.text for m in result["messages"]))
        self.assertEqual(result["messages"][-1].text, "Все модули проверены.")

    def test_distinct_names_allow_building_two_instances(self):
        self.build_agent(ScriptedModel(batches=[]), [AuditMiddleware(4), AuditMiddleware(8)])
        with self.assertRaises(AssertionError):
            self.build_agent(
                ScriptedModel(batches=[]),
                [UnnamedAuditMiddleware(4), UnnamedAuditMiddleware(8)],
            )

    def test_reducer_appends_without_mutation(self):
        old, new = ["начало"], ["конец"]
        self.assertEqual(append_audit_log(old, new), ["начало", "конец"])
        self.assertEqual(old, ["начало"])
        self.assertEqual(new, ["конец"])


if __name__ == "__main__":
    unittest.main()
