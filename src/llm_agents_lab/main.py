from langchain.messages import SystemMessage
from langchain.tools import tool
from asyncio import wait
from langchain.chat_models import BaseChatModel
from llm_agents_lab.course_model import build_model
from dataclasses import dataclass
from enum import Enum
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from typing import Any, Callable
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT
from langchain_core.messages import content as types
from langchain_core.messages.content import TextContentBlock
from datetime import date


RATES = {"input": 0.15, "output": 0.60}
CALLS = {"n": 0}


class UserRole(Enum):
    VIEWER = 0
    EDITOR = 1


@dataclass
class Context:
    user_role: UserRole
    project: str


BASE_PROMPT = "Ты интеллектуальный помощник интерфейса микросервиса. Напомни пользователю куда он попал, сегодняшнее число"


def show(message) -> str:
    """Одна строка на сообщение: тип, начало текста, вызовы инструментов."""
    text = message.text.replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."

    # Поле tool_calls есть только у AIMessage.
    calls = getattr(message, "tool_calls", None)
    suffix = f"  tool_calls={[call['name'] for call in calls]}" if calls else ""

    return f"{type(message).__name__:<14} {text!r}{suffix}"


@tool
def get_user_permissions(role: UserRole) -> list[str]:
    """Возвращает права пользователя"""
    match role:
        case UserRole.EDITOR:
            return ["read", "post", "update"]
        case _:
            return ["read"]


@dynamic_prompt
def role_prompt(request: ModelRequest) -> str:
    context: Context | None = getattr(request.runtime, "context", None)
    print(f"[role_prompt] context={context!r}, role={getattr(context, 'user_role', None)}")
    role: UserRole | None = context.user_role if isinstance(context, Context) else None
    match role:
        case UserRole.EDITOR:
            return f"{BASE_PROMPT} Роль:{role.__str__()} Перечисли действия в системе, доступные пользователю"
        case UserRole.VIEWER:
            return f"{BASE_PROMPT} Роль:{role.__str__()} Предложи перейти по ссылке https://tech.company.com/request_access"
        case _:
            return BASE_PROMPT


@wrap_model_call
def add_meta(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Добавляет блок с текущей датой и проектом."""
    if request.system_message is not None:
        context: Context | None = getattr(request.runtime, "context", None)
        project: str | None = context.project if isinstance(context, Context) else None
        today: str = date.today().isoformat()
        system_message: SystemMessage = request.system_message
        content_block: TextContentBlock = TextContentBlock(
            type="text",
            text=(
                    f"Проект: {project}. "
                    f"Дата: {today}."
            ),
        )
        new_content: list = [
            *system_message.content_blocks,
            content_block,
        ]
        return handler(request.override(system_message=SystemMessage(content=new_content)))
    return handler(request)


@wrap_model_call
def observer(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Печатает то, что уйдёт в модель, и пропускает вызов дальше."""
    CALLS["n"] += 1
    print(f"--- шаг {CALLS['n']}, в модель уходит ---")

    outgoing = list(request.messages)
    if request.system_message is not None:
        outgoing = [request.system_message, *outgoing]

    for number, message in enumerate(outgoing, start=1):
        print(f"  {number}. {show(message)}")

    print(f"  сообщений в запросе: {len(outgoing)}")
    print(f"  сообщений в состоянии: {len(request.state['messages'])}")
    names = [getattr(tool, "name", tool) for tool in request.tools]
    print(f"  инструменты в запросе: {names}")

    return handler(request)


def main():
    agent: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=build_model(temperature=0, max_tokens=300),
        tools=[get_user_permissions],
        system_prompt=BASE_PROMPT,
        middleware=[role_prompt, add_meta, observer],
        context_schema=Context,
    )


    QUESTION = "Я успешно авторизовался, что дальше?"

    for role in (UserRole.VIEWER, UserRole.EDITOR):
        print(f"РОЛЬ: {role}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": QUESTION}]},
            context=Context(user_role=role, project="Блог"),
        )
        print(f"  ответ: {result['messages'][-1].text}")
        print()

if __name__ == "__main__":
    main()
