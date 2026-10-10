from llm_agents_lab.course_model import build_model
from dataclasses import dataclass
from typing import Literal, Annotated
from langchain.agents import create_agent, AgentState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.runtime import Runtime
from typing import Any
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langchain.agents.middleware import before_model
import operator


SYSTEM_PROMPT: str = "Вы агент службы доставки" 
QUESTION: str = "Сперва перечислите мои заказы, затем отмените два моих последних заказа, затем снова покажите список моих заказов"

USERS: dict[str, dict] = {
    "user_001": {"name": "Кирилл"},
    "user_002": {"name": "Евгения"},
    "user_003": {"name": "Петр"},
}
USER_ORDERS: dict[str, list[str]] = {
    "user_001": ["order_101", "order_102", "order_303"],
    "user_002": ["order_201"],
    "user_003": ["order_301"],
}
ORDERS: dict[str, dict] = {
    "order_101": {"user_id": "user_001", "amount": 1500, "status": "active"},
    "order_102": {"user_id": "user_001", "amount": 3200, "status": "active"},
    "order_201": {"user_id": "user_002", "amount": 800,  "status": "active"},
    "order_302": {"user_id": "user_003", "amount": 700,  "status": "active"},
    "order_303": {"user_id": "user_001", "amount": 2200, "status": "active"},
}


class SupportState(AgentState):
    """Состояние агента поддержки: к messages добавлено поле отмененных заказов."""

    cancelled: Annotated[list[str], operator.add]


@dataclass
class SessionContext:
    """Конфигурация запуска: кто спрашивает."""

    user_id: str
    city: str
    role: Literal["client", "operator"]


@tool
def my_orders(runtime: ToolRuntime[SessionContext]) -> str:
    """Вернуть список активных заказов текущего пользователя."""
    user_id: str | None = getattr(runtime.context, "user_id", None)
    if user_id is None:
        return "Пользователь не передан в контекст"
    orders: list[str] = USER_ORDERS.get(user_id, [])
    if not orders:
        return "Активных заказов нет."
    return "\n".join(orders)


@tool
def cancel_order(order_id: str, runtime: ToolRuntime[SessionContext, SupportState]) -> Command | str:
    """Отменяет заказ по его идентификатору"""

    user_id: str | None = getattr(runtime.context, "user_id", None)
    if user_id is None:
        return "Пользователь не передан в контекст"

    user_role: str | None = getattr(runtime.context, "role", None)
    if user_role is None:
        return "Роль пользователя отсутствует"

    if user_role != "operator":
        return (
            f"В операции отказано: роль {runtime.context.role}. "
            "Отмена разрешена только оператору."
        )

    cancelled: list[str] = []
    order: dict | None = ORDERS.get(order_id)

    operation_result: str = f"Заказ {order_id} отсутствует"

    if order is not None:
        user_order_list: list[str] = USER_ORDERS.get(user_id, [])

        if order_id in user_order_list:
            ORDERS.pop(order_id, None)
            user_order_list.remove(order_id)

            cancelled.append(order_id)
            operation_result = f"Заказ {order_id} успешно отменен"

    return Command(
    update={
            "cancelled": cancelled,
            "messages": [
                ToolMessage(
                    content=operation_result,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@before_model
def show_runtime(state: AgentState, runtime: Runtime[SessionContext]) -> None:
    """Печатает поля Runtime перед вызовом модели."""
    print("MIDDLEWARE, объект Runtime")
    user_role: str | None = getattr(runtime.context, "role", None)
    if user_role is None:
        print("  конфигурация запуска не передана!")
    print("  context.user_role:", user_role)
    print("  сообщений в состоянии:", len(state["messages"]))
    return None


def run(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    question: str,
    **invoke_kwargs: Any,
):
    print(question)
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ],
            "cancelled": [],
        },
        **invoke_kwargs,
    )
    print("ответ агента:", result["messages"][-1].text.replace("\n", " "))
    print("ключи состояния:", list(result.keys()))
    print("отмененные заказы:", result["cancelled"])


def main():
    agent: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=build_model(temperature=0),
        system_prompt=SYSTEM_PROMPT,
        context_schema=SessionContext,
        state_schema=SupportState,
        tools=[my_orders, cancel_order],
        middleware=[show_runtime]
    )
    run(
        agent,
        QUESTION,
        context=SessionContext(
            user_id="user_001",
            city="Нячанг",
            role="client",
        ),
    )

    run(
        agent,
        QUESTION,
        context=SessionContext(
            user_id="user_001",
            city="Нячанг",
            role="operator",
        ),
    )


if __name__ == "__main__":
    main()
