from llm_agents_lab.course_model import build_model
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from collections.abc import Callable


TEST_HUMAN_MESSAGE_1: str = "Что с заявкой T-77?"
TEST_HUMAN_MESSAGE_2: str = "Что с заявкой T-999?"
TEST_HUMAN_MESSAGE_3: str = "Заявка T-77 не решается, передайте разработчикам"
TEST_HUMAN_MESSAGE_4: str = "Проверьте заявки T-77, T-78 и T-79"


class TicketEscalation(BaseModel):
    """Параметры заявки на эскалацию."""

    ticket_id: str = Field(
        description="Идентификатор заявки в формате T-XX"
    )
    reason: Literal["bug", "billing", "other"] = Field(
        description=(
            "Причина эскалации: "
            "bug — техническая ошибка; "
            "billing — проблемы с оплатой; "
            "other — прочие причины."
        )
    )


@tool(return_direct=True, args_schema=TicketEscalation)
def escalate(ticket_id: str, reason: str) -> str:
    """Возвращает строку содержащую информацию для эскалации конкретной заявки"""
    return (
        f"Заявка {ticket_id} передана специалистам. "
        f"Причина обращения: {reason}."
    )


@tool(parse_docstring=True)
def find_ticket(ticket_id: str) -> str:
    """Возвращает статус заявки по её идентификатору

    Args:
        ticket_id: идентификатор заявки в формате T-XX
    """
    database: dict = {
        "T-77": "Формирование пакета документов завершено, данные направлены на электронную почту клиента.",
        "T-78": "В процессе... приблизительная готовность - 3 рабочих дня.",
        "T-79": "В процессе... приблизительная готовность - 1 рабочий день.",
        "T-80": "Формирование пакета документов завершено, данные направлены на электронную почту клиента.",
        "T-81": "Отменен",
    }
    result: str | None = database.get(ticket_id)
    if result is not None:
        return result
    raise ValueError(f"заявка {ticket_id} не найдена в реестре")


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Превращает исключение инструмента в сообщение, понятное модели."""
    try:
        return handler(request)
    except Exception as error:
        return ToolMessage(
            content=f"Инструмент не отработал: {error}. Не повторяйте вызов, "
            "скажите пользователю, что такой заявки нет в системе.",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )


def report(title, agent, question):
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    messages = result["messages"]

    print(title)
    print("  типы сообщений:", [
        type(m).__name__ for m in messages
    ])

    ai_messages = [
        m for m in messages
        if type(m).__name__ == "AIMessage"
    ]

    print("  вызовов модели:", len(ai_messages))

    for index, message in enumerate(ai_messages, start=1):
        print(f"  ответ модели {index}:")

        for call in message.tool_calls:
            print("    инструмент:", call["name"])
            print("    аргументы:", call["args"])
            print("    ID вызова:", call["id"])

    for message in messages:
        if isinstance(message, ToolMessage):
            print("  результат инструмента:")
            print("    имя:", message.name)
            print("    статус:", message.status)
            print("    содержимое:", message.content)

    last = messages[-1]

    print("  последнее сообщение:", type(last).__name__)
    print("  его содержимое:", last.content)
    print()


def main():
    for current_tool in [find_ticket, escalate]:
        print(current_tool.name)
        print(current_tool.tool_call_schema.model_json_schema())


    agent = create_agent(
        middleware=[handle_tool_errors],
        model=build_model(temperature=0),
        tools=[find_ticket, escalate],
        system_prompt="Вы помощник магазина.",
    )
    report(
        f"ПРОГОН 1: {TEST_HUMAN_MESSAGE_1}",
        agent,
        TEST_HUMAN_MESSAGE_1,
    )
    report(
        f"ПРОГОН 2: {TEST_HUMAN_MESSAGE_2}",
        agent,
        TEST_HUMAN_MESSAGE_2,
    )
    report(
        f"ПРОГОН 3: {TEST_HUMAN_MESSAGE_3}",
        agent,
        TEST_HUMAN_MESSAGE_3,
    )
    report(
        f"ПРОГОН 4: {TEST_HUMAN_MESSAGE_4}",
        agent,
        TEST_HUMAN_MESSAGE_4,
    )


if __name__ == "__main__":
    main()
