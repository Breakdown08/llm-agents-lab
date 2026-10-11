from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from llm_agents_lab.course_model import build_model
from pathlib import Path
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from argparse import ArgumentParser, Namespace
from langgraph.graph.state import CompiledStateGraph
from typing import Any
from langgraph.types import StateSnapshot
from langchain.tools import tool
from collections import Counter
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from collections.abc import Callable
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage



DB_PATH = Path(__file__).with_name("session.sqlite")
CONSOLE_ARG_THREAD_ID: str = "thread_id"

SYSTEM_PROMPT: str = (
    "Вы мой домашний агент. Отвечайте по-русски, одной короткой фразой. "
    "Если нужного факта в диалоге не было, так и скажите."
)
SUMMARY_PROMPT: str = (
    "Кратко перескажите по-русски историю переписки ниже. Сохраните все "
    "факты, которые пользователь просил запомнить, и принятые в диалоге "
    "решения.\n\n"
    "Переписка:\n{messages}"
)
SUMMARY_PREFIX: str = "Here is a summary of the conversation to date:"

SHIFT_RULES: dict[int, str] = {
    1: "Не упоминать о Бойцовском клубе",
    2: "Нигде не упоминать о Бойцовском клубе",
    3: "Если кто-то крикнул «стоп», выдохся или отключился — бой окончен",
    4: "В бою участвуют лишь двое",
    5: "Только один бой за вечер (по очереди)",
    6: "Снимать обувь и рубашки (драться без рубашек и обуви)",
    7: "Бой продолжается столько, сколько потребуется",
    8: "Если новичок пришел первый раз — он обязан принять бой",
}

tool_stats: Counter[str] = Counter()


@wrap_tool_call
def collect_tool_stats(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    tool_name: str = request.tool_call["name"]

    tool_stats[tool_name] += 1

    return handler(request)


@tool
def shift_rules(topic: int) -> str:
    """Возвращает правило бойцовского клуба (их 8)"""
    return SHIFT_RULES.get(topic, "Такого правила нет, их всего 8")


def get_config_with_thread_id(thread_id: str) -> RunnableConfig:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return config


def print_summary_status(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    runnable_config: RunnableConfig,
) -> None:
    snapshot: StateSnapshot = agent.get_state(runnable_config)
    messages: list[BaseMessage] = snapshot.values.get("messages", [])

    summary: str | None = None

    for message in messages:
        if isinstance(message, HumanMessage):
            if message.text.startswith(SUMMARY_PREFIX):
                summary = message.text[len(SUMMARY_PREFIX):].lstrip()
                break

    print(f"summary_exists: {summary is not None}")
    print(f"summary_preview: {summary[:200] if summary is not None else 'N/A'}")


def request_with_config(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    runnable_config: RunnableConfig,
    question: str,
    call_number: int,
) -> None:
    """Пользовательский запрос агенту с конфигом конкретного диалога"""
    result: dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        runnable_config,
    )
    checkpoints: int = len(list(agent.get_state_history(runnable_config)))
    snapshot: StateSnapshot = agent.get_state(runnable_config)
    messages: list[BaseMessage] = snapshot.values.get("messages", [])
    messages_count: int = len(messages)
    answer: str = result["messages"][-1].text.replace("\n", " ")
    last_message: BaseMessage = result["messages"][-1]

    input_tokens: int = 0
    if isinstance(last_message, AIMessage):
        if last_message.usage_metadata is not None:
            input_tokens = last_message.usage_metadata.get("input_tokens", 0)

    print(f"вопрос:      {question}")
    print(f"ответ:       {answer}")
    print(f"файл thread: {DB_PATH.name}, {DB_PATH.stat().st_size} байт")
    print(f"tool calls:  {dict(tool_stats)}")
    print(f"call={call_number} | messages={messages_count} | checkpoints={checkpoints} | input_tokens={input_tokens}")


def dialog_loop(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    runnable_config: RunnableConfig,
) -> None:
    call_number: int = 0
    print("Dialog session started! Text your message to agent...")
    user_input_prefix = "> "
    while True:
        try:
            user_input: str = input(user_input_prefix)
        except EOFError:
            break
        if user_input.strip().lower() == "exit":
            break
        if not user_input.strip(): continue

        call_number += 1

        request_with_config(agent, runnable_config, user_input, call_number)

    print_summary_status(agent, runnable_config)
    print("Dialog session successfully executed.")


def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--thread", dest=CONSOLE_ARG_THREAD_ID, type=str, help="Запуск с нужным диалогом")
    args: Namespace = parser.parse_args()

    thread_id: str = getattr(args, CONSOLE_ARG_THREAD_ID) or "daily"
    runnable_config: RunnableConfig = get_config_with_thread_id(thread_id)
    model: BaseChatModel = build_model(temperature=0, max_tokens=256)

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = create_agent(
            model=model,
            tools=[shift_rules],
            middleware=[
                collect_tool_stats,
                SummarizationMiddleware(
                    model=model,
                    trigger=("messages", 10),
                    keep=("messages", 4),
                    summary_prompt=SUMMARY_PROMPT,
                ),
            ],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        dialog_loop(agent, runnable_config)


if __name__ == "__main__":
    main()
