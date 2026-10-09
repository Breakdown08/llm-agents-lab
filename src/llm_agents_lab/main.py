from langchain.chat_models import BaseChatModel
from llm_agents_lab.course_model import build_model
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


ISSUES: list[str] = [
    "Здравствуйте, когда доставят ORD-1234?",
    "Здравствуйте, когда доставят ID-1234?",
    "Здравствуйте, когда доставят мой заказ?",
]


class Issue(BaseModel):
    """Обращение в службу поддержки."""

    topic: Literal["оплата", "доставка", "возврат", "прочее"] = Field(
        description="Тема обращения"
    )
    urgency: Literal["низкая", "средняя", "высокая"] = Field(
        description="Срочность обращения"
    )
    summary: str = Field(description="Суть обращения одним предложением, по-русски")
    order_id: str | None = Field(description="Идентификатор заказа")


    @field_validator("order_id")
    @classmethod
    def check_order_id(cls, value: str) -> str:
        """Внутренний формат номера, которого нет в JSON Schema."""
        if not value.startswith("ORD-"):
            message = "Номер заказа записывается с префиксом ORD-, например ORD-1234"
            raise ValueError(message)
        return value


def run(model: BaseChatModel, title: str, content: str, handle_errors) -> None:
    """Прогоняет одно обращение с заданным режимом обработки ошибок."""
    print(title)
    agent = create_agent(
        model=model,
        tools=[], response_format=ToolStrategy(schema=Issue, handle_errors=handle_errors),
    )

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    except Exception as error:  # noqa: BLE001
        print("  исключение:", type(error).__name__)
        print("  текст:", str(error)[:160])
        print()
        return

    calls = sum(1 for m in result["messages"] if type(m).__name__ == "AIMessage")
    print("  вызовов модели:", calls)
    for message in result["messages"]:
        if type(message).__name__ == "ToolMessage":
            print("  сообщение инструмента:", repr(message.text[:120]))
    print("  разобранный ответ:", result.get("structured_response"))
    print()


def main():
    model: BaseChatModel = build_model()
    for issue in ISSUES:
        run(
            model,
            "РЕЖИМ 1: handle_errors=ValueError, как показано в документации",
            issue,
            ValueError
        )

if __name__ == "__main__":
    main()
