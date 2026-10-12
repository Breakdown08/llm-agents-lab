from langchain.agents import create_agent
from uuid import uuid4
from langgraph.store.base import BaseStore, SearchItem
from typing import Literal
from dataclasses import dataclass
from llm_agents_lab.course_model import build_model
from langgraph.store.sqlite import SqliteStore
from pathlib import Path
from langchain.tools import ToolRuntime, tool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from collections.abc import Callable
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import time


DB_PATH = Path(__file__).with_name("memory.db")
CHECKPOINT_PATH = Path(__file__).with_name("checkpoints.db")
# Для практики
DB_PATH.unlink(missing_ok=True)
CHECKPOINT_PATH.unlink(missing_ok=True)

ALLOWED_FIELDS = ("city", "product", "contact_time")


@dataclass
class Session:
    """Конфигурация запуска: чьи записи править."""

    user_id: str


def print_memory(
    store: BaseStore,
    user_id: str,
    stage: str,
) -> None:
    print(f"\n{'=' * 60}")
    print(stage)
    print("=" * 60)

    for name in ("profile", "notes"):
        namespace: tuple[str, str] = init_namespace(user_id, name)
        items: list[SearchItem] = store.search(namespace, limit=10)

        print(f"\nNamespace: {namespace}")
        print(f"Количество записей: {len(items)}")

        for item in items:
            print(f"  {item.key}: {item.value}")


def print_result(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, HumanMessage):
            print(f"Пользователь: {message.content}")

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            print(f"Ответ модели: {message.content}")
            break


@tool
def remember_fact(
    field: Literal["city", "product", "contact_time"],
    value: str,
    runtime: ToolRuntime[Session],
) -> tuple[bool, str]:
    """Запомнить устойчивый факт о пользователе.

    Поля: city это город работы; product это какой информационный продукт обсуждается с клиентом; contact_time это
    рабочие часы. Остальные состояния этим инструментом не сохраняются.
    Возвращает пару: принято ли и почему.
    """

    store: BaseStore | None = runtime.store
    if store is None:
        return False, "хранилище недоступно"

    namespace_profile: tuple[str, str] = init_namespace(runtime.context.user_id, "profile")

    if field not in ALLOWED_FIELDS:
        return False, f"Поле {field} не сохраняется, допустимы: {', '.join(ALLOWED_FIELDS)}."

    store.put(namespace_profile, field, {"value": value})

    return True, f"Запомнил {field}."


@tool
def add_note(
    text: str,
    days: float,
    runtime: ToolRuntime[Session],
) -> tuple[bool, str]:
    """Записывает разовую заметку.

    Поля: text это заметка; days срок годности в днях
    Возвращает пару: принято ли и почему.
    """

    store: BaseStore | None = runtime.store
    if store is None:
        return False, "хранилище недоступно"

    namespace_notes: tuple[str, str] = init_namespace(runtime.context.user_id, "notes")

    ttl_minutes: float = days * 24 * 60

    store.put(
        namespace_notes,
        str(uuid4()),
        {"text": text},
        ttl=ttl_minutes,
    )

    return True, f"Запомнил {text}."


@wrap_model_call
def inject_memory(
    request: ModelRequest[Session],
    handler: Callable[[ModelRequest[Session]], ModelResponse],
) -> ModelResponse:
    """Добавляет профиль и заметки пользователя в системный промпт."""

    store: BaseStore | None = request.runtime.store
    if store is None:
        return handler(request)

    context: Session | None = request.runtime.context
    if context is None:
        return handler(request)

    user_id: str = context.user_id

    namespace_profile: tuple[str, str] = init_namespace(user_id, "profile")
    namespace_notes: tuple[str, str] = init_namespace(user_id, "notes")

    profile_items: list[SearchItem] = store.search(namespace_profile)
    note_items: list[SearchItem] = store.search(namespace_notes)

    profile_text: str = "\n".join(
        f"{item.key}: {item.value['value']}"
        for item in profile_items
    )

    notes_text: str = "\n".join(
        item.value["text"]
        for item in note_items
    )

    memory_text: str = (
        f"\n\nПрофиль клиента:\n{profile_text or 'Нет данных'}"
        f"\n\nЗаметки:\n{notes_text or 'Нет заметок'}"
    )

    system_text: str = (
        request.system_message.text + memory_text
        if request.system_message is not None
        else memory_text
    )

    return handler(
        request.override(
            system_message=SystemMessage(content=system_text)
        )
    )


def init_namespace(user_id: str, namespace: str) -> tuple[str, str]:
    return (user_id, namespace)


def main():
    user_id: str = "test_user"

    with (
        SqliteStore.from_conn_string(str(DB_PATH)) as store,
        SqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as checkpointer,
    ):
        store.setup()
        agent = create_agent(
            model=build_model(),
            tools=[remember_fact, add_note],
            middleware=[inject_memory],
            context_schema=Session,
            store=store,
            checkpointer=checkpointer
        )

        print_memory(store, user_id, "ДО ВЫЗОВОВ АГЕНТА")

        context = Session(user_id=user_id)

        first_result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Я работаю в Нячанге, обсуждаю продукт LangChain, "
                            "со мной удобно связываться после 18:00. "
                            "Напомни мне про заявку T-77. "
                            "Сохрани напоминание на 0.001 дня."
                        ),
                    }
                ]
            },
            config={"configurable": {"thread_id": "thread_1"}},
            context=context,
        )

        print("Первый ответ:")
        print_result(first_result["messages"])

        second_result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Теперь я работаю в Ханое. "
                            "Что записано в моём профиле?"
                        ),
                    }
                ]
            },
            config={"configurable": {"thread_id": "thread_2"}},
            context=context,
        )

        print()
        print_memory(store, user_id, "ПОСЛЕ ВЫЗОВОВ АГЕНТА")
        print()

        print("Второй ответ:")
        print_result(second_result["messages"])

        print()
        print("\nОжидание 120 секунд...")
        time.sleep(120)

        deleted: int = store.sweep_ttl()
        print(f"Удалено просроченных записей: {deleted}")

        print_memory(store, user_id, "ПОСЛЕ SWEEP_TTL")


if __name__ == "__main__":
    main()
