from langchain.chat_models import BaseChatModel
from langchain_anthropic.chat_models import ModelProfile
from langchain_core.messages import BaseMessage
from langchain.messages import HumanMessage, AIMessage, UsageMetadata
from collections.abc import Callable

from langchain_core.messages.utils import count_tokens_approximately
from llm_agents_lab.course_model import build_model

PRICE_INPUT_MISS = 1.00
PRICE_INPUT_HIT = 0.10
PRICE_OUTPUT = 5.00


Checker = Callable[
    [BaseChatModel, AIMessage, list[BaseMessage]],
    None,
]


def make_checker() -> Checker:
    turn: int = 0


    def check(
        model: BaseChatModel,
        ai_message: AIMessage,
        history: list[BaseMessage],
    ) -> None:
        nonlocal turn
        turn += 1
        print(f"Ход диалога: {turn}")

        profile: ModelProfile | None = model.profile
        max_input_tokens: int = 0
        input_tokens_approximate: int = count_tokens_approximately(history)
        input_tokens_actual: int = 0
        usage_metadata: UsageMetadata | None = ai_message.usage_metadata
        if usage_metadata is not None:
            input_tokens_actual = usage_metadata["input_tokens"]
        input_tokens_error_percent:int = _get_input_tokens_error_percent(input_tokens_approximate, input_tokens_actual)
        if profile is not None:
            max_input_tokens = profile.get('max_input_tokens', 0)
        usage_ratio: float = input_tokens_actual / max_input_tokens
        if usage_ratio >= 0.2:
            print("\n" + "=" * 60)
            print("⚠️  ПРЕДУПРЕЖДЕНИЕ: КОНТЕКСТНОЕ ОКНО ЗАПОЛНЕНО НА 20%")
            print("=" * 60 + "\n")
        characters:int = sum(len(message.content) for message in history)
        print(f"Символы: {characters}    Приблизительный вход: {input_tokens_approximate}    Реальный вход: {input_tokens_actual}    Расхождение: {input_tokens_error_percent}%")
        
    return check


def _get_input_tokens_error_percent(
        input_tokens_approximate: int,
        input_tokens_actual: int
    ) -> int:
    return round(
        abs(input_tokens_approximate - input_tokens_actual)
        / input_tokens_actual
        * 100
    )


def _calculate_total_cost(output_tokens: int, cached: int, fresh: int) -> float:
    return (
            fresh * PRICE_INPUT_MISS
            + cached * PRICE_INPUT_HIT
            + output_tokens * PRICE_OUTPUT
        ) / 1_000_000



def show_total_cost(model: BaseChatModel, last_ai_message: AIMessage | None):
    if last_ai_message is None:
        return
    usage_metadata: UsageMetadata | None = last_ai_message.usage_metadata
    if usage_metadata is not None:
        cached: int = (usage_metadata.get("input_token_details") or {}).get("cache_read", 0)
        fresh: int = usage_metadata["input_tokens"] - cached
        money: float = _calculate_total_cost(usage_metadata["total_tokens"], cached, fresh)         
        profile: ModelProfile | None = model.profile
        model_name: str = profile.get("name", "unknown") if profile is not None else "unknown"
        print(f"  модель: {model_name}")
        print(f"  вход:   {usage_metadata['input_tokens']} токенов, из них из кеша {cached}")
        print(f"  выход:  {usage_metadata['output_tokens']} токенов")
        print(f"  всего:  {usage_metadata['total_tokens']} токенов")
        print(f"  деньги: ${money:.6f}")

