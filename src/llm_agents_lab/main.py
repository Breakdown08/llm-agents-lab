"""Практическое задание: аудит агента и бюджет вызовов инструментов.

Запуск: python -m llm_agents_lab.main (или project).
Для проверки без условного перехода уберите @hook_config у before_model.
Свойство name позволяет собрать агента с двумя разными бюджетами, но запускать
вложенные обёртки нельзя: внутренняя возвращает Command вместо ToolMessage.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ToolCallRequest,
    hook_config,
)
from langchain.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from llm_agents_lab.course_model import build_model


def append_audit_log(old: list[str], new: list[str]) -> list[str]:
    """На каждом обновлении добавляет только новые записи журнала."""
    return old + new


@dataclass
class AuditContext:
    user_id: str


class AuditState(AgentState):
    model_calls: NotRequired[int]
    tool_calls: NotRequired[int]
    audit_log: Annotated[list[str], append_audit_log]


class AuditMiddleware(AgentMiddleware[AuditState, AuditContext]):
    """Проверяет бюджет перед моделью; уже заказанные инструменты выполняются все."""

    state_schema = AuditState

    def __init__(self, max_tool_calls: int) -> None:
        super().__init__()
        self.max_tool_calls = max_tool_calls

    @property
    def name(self) -> str:
        return f"{type(self).__name__}(max_tool_calls={self.max_tool_calls})"

    def before_agent(
        self, state: AuditState, runtime: Runtime[AuditContext]
    ) -> dict[str, Any]:
        return {
            "audit_log": [f"[user_id={runtime.context.user_id}] before_agent: начало запуска"],
        }

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AuditState, runtime: Runtime[AuditContext]
    ) -> dict[str, Any]:
        exhausted = state.get("tool_calls", 0) >= self.max_tool_calls
        entry = (
            f"[user_id={runtime.context.user_id}] before_model: "
            f"tool_calls={state.get('tool_calls', 0)}, "
            f"max_tool_calls={self.max_tool_calls}"
        )
        if exhausted:
            return {
                "audit_log": [entry + "; бюджет исчерпан"],
                "messages": [AIMessage("Бюджет вызовов инструментов исчерпан.")],
                "jump_to": "end",
            }
        return {"audit_log": [entry]}

    def after_model(
        self, state: AuditState, runtime: Runtime[AuditContext]
    ) -> dict[str, Any]:
        last = state["messages"][-1]
        model_calls = state.get("model_calls", 0) + 1
        tool_calls = state.get("tool_calls", 0) + len(
            getattr(last, "tool_calls", []) or []
        )
        return {
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "audit_log": [
                f"[user_id={runtime.context.user_id}] after_model: "
                f"model_calls={model_calls}, tool_calls={tool_calls}"
            ],
        }

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> Command:
        response = handler(request)
        # В задании используется один экземпляр обёртки и обычный инструмент.
        if not isinstance(response, ToolMessage):
            raise TypeError("AuditMiddleware ожидает ToolMessage от handler")
        args = json.dumps(request.tool_call["args"], ensure_ascii=False, sort_keys=True)
        return Command(
            update={
                "messages": [response],
                "audit_log": [
                    f"[user_id={request.runtime.context.user_id}] wrap_tool_call: "
                    f"{request.tool_call['name']} args={args}, "
                    f"response_length={len(response.text)}"
                ],
            }
        )

    def after_agent(
        self, state: AuditState, runtime: Runtime[AuditContext]
    ) -> dict[str, Any]:
        return {
            "audit_log": [
                f"[user_id={runtime.context.user_id}] after_agent: "
                f"model_calls={state.get('model_calls', 0)}, "
                f"tool_calls={state.get('tool_calls', 0)}"
            ],
        }


TESTS = {
    "/src/reports": 0,
    "/src/billing": 41,
    "/src/auth": 12,
    "/src/gateway": 8,
    "/src/notifications": 23,
    "/src/search": 17,
}


def module_tests(path: str) -> str:
    """Возвращает число тестов модуля репозитория по пути."""
    count = TESTS.get(path)
    if count is None:
        return f"модуля {path} в репозитории нет"
    return f"{path}: тестов {count}"


def main() -> None:
    agent = create_agent(
        model=build_model(temperature=0, max_tokens=1024),
        tools=[module_tests],
        system_prompt=(
            "Вы помощник по разбору легаси. Число тестов узнавайте инструментом, "
            "по одному модулю за вызов. Узнайте число тестов каждого модуля из "
            "вопроса, затем отвечайте по-русски одной фразой."
        ),
        middleware=[AuditMiddleware(max_tool_calls=4)],
        context_schema=AuditContext,
    )
    for paths in (list(TESTS)[:2], list(TESTS)):
        question = f"Сколько тестов в модулях {', '.join(paths)}?"
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "model_calls": 0,
                "tool_calls": 0,
                "audit_log": [],
            },
            context=AuditContext(user_id="u-101"),
        )
        print(question)
        print(*result["audit_log"], sep="\n")
        print("Ответ:", result["messages"][-1].text)
        print()


if __name__ == "__main__":
    main()
