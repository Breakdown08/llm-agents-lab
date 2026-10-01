import json
import warnings
from typing import Any
from pathlib import Path
from langchain.tools import tool
from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain.chat_models import BaseChatModel
from llm_agents_lab.course_model import build_model

HISTORY_FILE: Path = Path(__file__).with_name("dialogue.json")
SAVE_HISTORY: bool = False


def restore(items, **kwargs) -> tuple[list[BaseMessage], list[warnings.WarningMessage]]:
    """Собирает сообщения из словарей и возвращает их вместе с предупреждениями."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        messages: list[BaseMessage] = [load(item, **kwargs) for item in items]
    return messages, caught


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    return f"В городе {city} сейчас +17 и дождь."


def get_message_passport(message: BaseMessage) -> str:
    passport: str = ""
    passport += "Message passport:\n"
    passport += f"    ClassName: {type(message).__name__}\n"
    passport += f"    Field type: {type(message.content).__name__}\n"
    passport += f"    Total characters: {len(message.text)}\n"
    passport += (
        f"    List blocks types: "
        f"{[block["type"] for block in message.content_blocks]}\n"
    )
    passport += (
        f"    Tool calls count: "
        f"{len(getattr(message, "tool_calls", []))}\n"
    )
    return passport


def main():
    model_with_tools: Runnable[Any, AIMessage] = build_model(temperature=0).bind_tools([get_weather])
    history: list[BaseMessage] = [HumanMessage("Какая погода в Казани?")]

    ai_message: AIMessage = model_with_tools.invoke(history)

    if not ai_message.tool_calls:
        print("Модель не попросила инструмент, продолжать цикл не с чем.")
        print("Так бывает: вызов инструментов держат не все модели и не все шлюзы.")
        raise SystemExit(0)

    history.append(ai_message)

    for tool_call in ai_message.tool_calls:
        tool_message = get_weather.invoke(tool_call)
        history.append(tool_message)

    final: AIMessage = model_with_tools.invoke(history)
    history.append(final)

    if SAVE_HISTORY:
        HISTORY_FILE.write_text(
            json.dumps([dumpd(message) for message in history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        raw: str = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        data: tuple[list[BaseMessage], list[warnings.WarningMessage]] = restore(raw, allowed_objects="messages")
        restored: list[BaseMessage] = data[0]
        print("ПОСЛЕ ВОССТАНОВЛЕНИЯ:", [type(message).__name__ for message in restored])
        print(
            "СОДЕРЖИМОЕ СОВПАЛО:",
            [get_message_passport(message) for message in restored] == [get_message_passport(message) for message in history],
        )

    for message in history:
        print(get_message_passport(message))


if __name__ == "__main__":
    main()
