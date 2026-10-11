from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.tools import BaseTool, ToolRuntime
from langchain.chat_models import BaseChatModel
from llm_agents_lab.course_model import build_model
from langchain.agents import create_agent, AgentState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
)
from langchain.messages import ToolMessage
from langchain.messages import AIMessage
from typing import Annotated, Any
import operator
import json


SYSTEM_PROMPT: str = "Вы оператор отдела кадров по вопросам командировок. Отвечайте по-русски и коротко."
RUN_LIMIT: int = 4


class SupportState(AgentState):
    """Состояние агента поддержки: сотрудник и вызванные инструменты"""

    employee: str
    looked_up: Annotated[list[str], operator.add]


def load_file(file: str) -> dict[str, Any]:
    """Загружает JSON-файл и возвращает словарь"""
    with open(file, encoding="utf-8") as data:
        parsed: Any = json.load(data)
    return parsed if isinstance(parsed, dict) else {}


def command_with_looked_up(looked_up: str, content: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "looked_up": [looked_up],
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def find_trip(code: str, runtime: ToolRuntime) -> Command:
    """Возвращает город и даты поездки по коду поездки вида TR-XX"""
    trip: dict[str, Any] | None = load_file("src/llm_agents_lab/trips.json").get(code)
    result: str = f"Поездка с кодом {code} не найдена"
    employee: str = runtime.state.get("employee", "Аноним")
    if trip:
        result = (
            f"Сотрудник: {employee} "
            f"Код поездки: {code}; Город: {trip['city']}; "
            f"Даты: с {trip['date_from']} по {trip['date_to']} включительно."
        )
    return command_with_looked_up(find_trip.name, result, runtime)


@tool
def hotel_price(city: str, runtime: ToolRuntime) -> Command:
    """Возвращает стоимость ночи в городе в рублях, если данных нет, возвращается -1"""
    hotels: list[dict[str, Any]] = load_file("src/llm_agents_lab/hotels.json").get(city, [])
    cheapest_hotel_price: int = -1
    employee: str = runtime.state.get("employee", "Аноним")
    if hotels:
        cheapest_hotel = min(hotels, key=lambda hotel: hotel["price"])
        cheapest_hotel_price = cheapest_hotel["price"]
    return command_with_looked_up(hotel_price.name, str(cheapest_hotel_price), runtime)


def run(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    question: str,
    **invoke_kwargs: Any,
):
    print(question)

    step: int = 0

    messages_history: list[BaseMessage] = []
    looked_up: list[str] = []

    for chunk in agent.stream(
        {
            "messages": [
                {"role": "user", "content": question}
            ],
            "employee": "Кирилл",
            "looked_up": [],
        },
        **invoke_kwargs,
    ):
        if chunk["type"] != "updates":
            continue

        for node, update in chunk["data"].items():
            step += 1

            if not isinstance(update, dict):
                continue

            if "looked_up" in update:
                looked_up.extend(update["looked_up"])

            messages = update.get("messages", [])

            for message in messages:
                messages_history.append(message)

                last_message = message
                calls = getattr(message, "tool_calls", None)
                if calls:
                    names = [call["name"] for call in calls]
                    print(f"шаг {step}: узел {node}, запрошены инструменты {names}")
                else:
                    text = message.text.replace("\n", " ")
                    if len(text) > 50:
                        text = text[:47] + "..."
                    print(f"шаг {step}: узел {node}, {type(message).__name__} {text!r}")

    print()
    print("ШАГОВ ВСЕГО:", step)
    print(
        "ПОСЛЕДНЕЕ СООБЩЕНИЕ ЗАПРОСИЛО ИНСТРУМЕНТЫ:",
        bool(getattr(last_message, "tool_calls", None)),
    )

    ai_messages: list[AIMessage] = [
        message
        for message in messages_history
        if isinstance(message, AIMessage)
    ]
    budget_exhausted: bool = len(ai_messages) > RUN_LIMIT

    last_ai_message: AIMessage | None = (
        ai_messages[-1] if ai_messages else None
    )

    answer: str = (
        last_ai_message.text
        if last_ai_message is not None
        else "Ответ отсутствует"
    )

    print()
    print("Ответ пользователю:", answer.replace("\n", " "))
    print("looked_up:", looked_up)
    print("Бюджет исчерпан:", budget_exhausted)


def main():
    model: BaseChatModel = build_model(temperature=0, max_tokens=512)
    tools: list[BaseTool] = [find_trip, hotel_price]

    agent: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=SupportState,
        middleware=[
            ModelCallLimitMiddleware(run_limit=RUN_LIMIT, exit_behavior="end"),
        ],
    )

    run(
        agent,
        "Найди командировку TR-22 и сообщи стоимость ночи в отеле в городе назначения.",
        config={"recursion_limit": 25},
        stream_mode="updates",
        version="v2",
    )


if __name__ == "__main__":
    main()

