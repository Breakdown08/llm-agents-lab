import argparse
from dataclasses import dataclass
from typing import Callable
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from typing import Any
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langgraph.checkpoint.memory import InMemorySaver
from llm_agents_lab.course_model import build_model


REPORTS: dict[str, str] = {
    "Казань": "бой 9, подмочка 5, пересорт 3",
    "Омск": "бой 2, подмочка 1, пересорт 14",
    "Пермь": "бой 4, подмочка 12, пересорт 2",
}

FILLER: str = (
    "Приёмка идёт по электронной очереди, талон берут на въезде. Паллеты "
    "принимают по счёту мест и по весу брутто. Упаковку вскрывают при видимом "
    "повреждении обрешётки и снимают вскрытие на камеру. Температурный груз "
    "замеряют дважды. Документы возвращают водителю с отметкой о приёмке, копию "
    "подшивают в журнал склада и дублируют в учётную систему до 20:00 того же "
    "дня. Расхождение больше пяти мест останавливает приёмку до прихода "
    "старшего смены."
)

LONG_PROMPT: str = (
    "Вы помощник логистического отдела. Отвечайте по-русски, одной фразой. "
    "Данные берите инструментами, по памяти не отвечайте. "
    "Инструмент read_report отдаёт недельный отчёт по складу города вместе с "
    "регламентом приёмки. Инструмент read_stock отдаёт остаток по артикулу. "
    "Инструмент read_routes отдаёт маршруты через город. Инструмент "
    "write_order меняет статус заказа, его разрешено звать только оператору "
    "склада, и только после того, как оператор назвал номер заказа целиком. "
    "Инструмент cancel_order отменяет заказ, его разрешено звать только "
    "оператору склада и только после подтверждения суммы возврата. "
    "Наблюдателю запрещено менять данные: если наблюдатель просит отменить "
    "заказ или сменить статус, откажите и предложите обратиться к оператору."
)

TEMPERATURE: int = 0
MAX_TOKENS: int = 384

@dataclass
class Access:
    """Конфигурация запуска: под какой ролью работает собеседник."""

    role: str


def read_report(city: str) -> str:
    """Возвращает полный недельный отчёт по складу города."""
    defects = REPORTS.get(city)

    if defects is None:
        return f"Отчёта по складу {city} нет."

    return f"Отчёт по складу {city}. Регламент: {FILLER} Брак за неделю: {defects}."


def read_stock(sku: str) -> str:
    """Возвращает остаток товара по артикулу на всех складах."""
    return f"Артикул {sku}: Казань 7 шт., Омск 0 шт."


def read_routes(city: str) -> str:
    """Возвращает список маршрутов, которые проходят через город."""
    return f"Через город {city} идут маршруты M-1 и M-4."


def write_order(order_id: str, status: str) -> str:
    """Меняет статус заказа на указанный."""
    return f"Заказ {order_id} переведён в статус {status}."


def cancel_order(order_id: str) -> str:
    """Отменяет заказ по номеру."""
    return f"Заказ {order_id} отменён."


ALL_TOOLS = [read_report, read_stock, read_routes, write_order, cancel_order]


@dynamic_prompt
def prompt_for_role(request: ModelRequest) -> str:
    """Собирает короткую инструкцию под роль собеседника."""
    base = (
        "Вы помощник логистического отдела. Отвечайте по-русски, одной фразой. "
        "Данные берите инструментами, по памяти не отвечайте."
    )

    if request.runtime.context.role == "оператор":
        return base + " Менять статус и отменять заказ разрешено."

    return base + " Менять данные вам нельзя, доступно только чтение."


@wrap_model_call
def tools_by_role(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Оставляет в запросе инструменты, разрешённые роли собеседника."""
    if request.runtime.context.role != "оператор":
        request = request.override(
            tools=[t for t in request.tools if t.name.startswith("read_")]
        )

    return handler(request)


def spent(messages) -> int:
    """Складывает входные токены по всем ответам модели в истории thread."""
    total = 0

    for message in messages:
        # Расход возвращает не каждый провайдер, поэтому пустой словарь как запасной путь.
        usage = getattr(message, "usage_metadata", None) or {}
        total += usage.get("input_tokens", 0)

    return total


QUESTIONS = [
    "Сколько брака за неделю на складе Казань?",
    "А на складе Омск?",
    "А на складе Пермь?",
]

def run_basic():
    print(f"{'агент':<10}  {'вызовов':>7}  {'вход всего':>10}  последний ответ")
    model_raw: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=build_model(temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
        tools=ALL_TOOLS,
        system_prompt=LONG_PROMPT,
        context_schema=Access,
        checkpointer=InMemorySaver(),
    )
    model_with_dynamic_prompt: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=build_model(temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
        tools=ALL_TOOLS,
        system_prompt=LONG_PROMPT,
        context_schema=Access,
        checkpointer=InMemorySaver(),
        middleware=[prompt_for_role]
    )
    model_with_dynamic_prompt_and_tools: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=build_model(temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
        tools=ALL_TOOLS,
        system_prompt=LONG_PROMPT,
        context_schema=Access,
        checkpointer=InMemorySaver(),
        middleware=[prompt_for_role, tools_by_role]
    )
    model_with_dynamic_prompt_and_tools_and_context: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        model=build_model(temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
        tools=ALL_TOOLS,
        system_prompt=LONG_PROMPT,
        context_schema=Access,
        checkpointer=InMemorySaver(),
        middleware=[
            prompt_for_role,
            tools_by_role,
            ContextEditingMiddleware(
                edits=[ClearToolUsesEdit(trigger=300, keep=1)],
            ),
        ]
    )


    for label, agent in (
        ("голый", model_raw),
        ("с промптом", model_with_dynamic_prompt),
        ("с промптом и инструментами", model_with_dynamic_prompt_and_tools),
        ("с промптом, инструментами и контекстом", model_with_dynamic_prompt_and_tools_and_context)
    ):
        config = {"configurable": {"thread_id": f"together-{label}"}}

        for question in QUESTIONS:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config,
                context=Access(role="наблюдатель"),
            )

        messages = result["messages"]
        calls = sum(1 for message in messages if message.type == "ai")
        answer = messages[-1].text.replace("\n", " ")
        print(f"{label:<10}  {calls:>7}  {spent(messages):>10}  {answer}")


@wrap_model_call
def observer(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Показывает итоговый запрос, ничего в нём не меняя."""
    instruction = request.system_message.text if request.system_message else ""
    tools = ", ".join(t.name for t in request.tools)
    print(f"  Инструкция: {instruction}")
    print(f"  Инструменты: {tools}")
    return handler(request)


def run_advanced():
    results = []

    for label, tuned in (("голый", False), ("четвёртый", True)):
        middleware = []
        if tuned:
            middleware.extend([
                prompt_for_role,
                tools_by_role,
                ContextEditingMiddleware(
                    edits=[ClearToolUsesEdit(trigger=300, keep=1)],
                ),
            ])
        # Последний middleware видит уже подставленную инструкцию и отбор tools.
        middleware.append(observer)
        agent = create_agent(
            model=build_model(temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
            tools=ALL_TOOLS,
            system_prompt=None if tuned else LONG_PROMPT,
            context_schema=Access,
            checkpointer=InMemorySaver(),
            middleware=middleware,
        )

        for role in ("оператор", "наблюдатель"):
            config = {"configurable": {"thread_id": f"advanced-{label}-{role}"}}
            print(f"\nСборка: {label}; роль: {role}; thread: {config['configurable']['thread_id']}")
            history_size = 0

            for question in [*QUESTIONS, "Отмените заказ Z-771"]:
                print(f"\nВопрос: {question}")
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config,
                    context=Access(role=role),
                )
                messages = result["messages"]
                turn_messages = messages[history_size:]
                history_size = len(messages)
                answer = messages[-1].text.replace("\n", " ")
                print(f"Ответ: {answer}")

            # Смотрим реальные вызовы инструментов именно на просьбу об отмене.
            tool_calls = [
                call["name"]
                for message in turn_messages
                if message.type == "ai"
                for call in message.tool_calls
            ]
            results.append((
                label, role,
                sum(message.type == "ai" for message in messages),
                spent(messages), ", ".join(tool_calls) or "—", answer,
            ))

    print("\nРЕЗУЛЬТАТЫ УСЛОЖНЕНИЯ")
    print(f"{'агент':<12} {'роль':<12} {'вызовов':>7} {'вход всего':>10}  инструменты на отмене; последний ответ")
    for label, role, calls, tokens, tool_names, answer in results:
        print(f"{label:<12} {role:<12} {calls:>7} {tokens:>10}  {tool_names}; {answer}")

    print("\nСРАВНЕНИЕ ОТВЕТОВ ОПЕРАТОРУ")
    for label, role, _, _, tool_names, answer in results:
        if role == "оператор":
            print(f"{label}: {answer} (вызванные инструменты: {tool_names})")

    print(
        "\nПричина: голая сборка требует подтверждения суммы возврата перед отменой, "
        "но роль из Access не включается в запрос к модели. Инструкция и набор "
        "инструментов у неё одинаковы для обеих ролей. В четвёртой сборке роль "
        "определяет текст инструкции и набор инструментов: оператору разрешена "
        "отмена, а наблюдателю доступны только read_* инструменты."
    )
    print(
        "Выигрыш: короткая инструкция, отбор инструментов для наблюдателя и очистка "
        "старых результатов. Потерянное условие: в короткой инструкции оператору "
        "нет требования подтвердить сумму возврата (и назвать полный номер перед "
        "сменой статуса). Экономия токенов не означает сохранения всех правил."
    )


def main():
    parser = argparse.ArgumentParser(description="Сравнение сборок агента")
    parser.add_argument(
        "--advanced", action="store_true", help="запустить только усложнение",
    )
    args = parser.parse_args()
    if not args.advanced:
        run_basic()
    run_advanced()


if __name__ == "__main__":
    main()
