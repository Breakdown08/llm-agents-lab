import os
import time
from collections import defaultdict

from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.exceptions import ModelError
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import UsageMetadataCallbackHandler

from llm_agents_lab.course_model import gateway_kwargs
from llm_agents_lab.tools import show_profile


FAST_MODEL = os.environ["MODEL_NAME"]
STRONG_MODEL = os.getenv("MODEL_NAME_STRONG") or FAST_MODEL


router = init_chat_model(
    configurable_fields=("model", "max_tokens"),
    config_prefix="task_llm",
    temperature=0,
    **gateway_kwargs(),
)

TASKS = [
    (
        "классификация",
        "Отнесите обращение к одной категории: оплата, доставка, возврат. "
        "Обращение: деньги списали дважды. Ответьте одним словом.",
        {"task_llm_model": FAST_MODEL, "task_llm_max_tokens": 16},
    ),
    (
        "короткая справка",
        "Что такое очередь задач? Одно предложение.",
        {"task_llm_model": FAST_MODEL, "task_llm_max_tokens": 120},
    ),
    (
        "разбор с доводами",
        "Сравните очередь задач и стек по трём признакам, с доводами.",
        {"task_llm_model": STRONG_MODEL, "task_llm_max_tokens": 400},
    ),
]

rate_limiter: InMemoryRateLimiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

router = init_chat_model(
    configurable_fields=("model", "max_tokens"),
    config_prefix="task_llm",
    temperature=0,
    rate_limiter=rate_limiter,
    **gateway_kwargs(),
)

usage_callback = UsageMetadataCallbackHandler()

model_usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0})

PRICE_INPUT = 0.15  # $ за 1M входных токенов
PRICE_OUTPUT = 0.60  # $ за 1M выходных токенов


def main():
    print()

    if STRONG_MODEL == FAST_MODEL:
        print("MODEL_NAME_STRONG не задана: обе роли идут на одну модель.")
        print()

    start_time = time.time()

    for idx, (title, prompt, settings) in enumerate(TASKS):
        print(f"{title.upper()}")
        print(f"  настройки: {settings}")

        try:
            response = router.invoke(
                prompt,
                config={
                    "configurable": settings,
                    "callbacks": [usage_callback],
                    "run_name": f"task_{idx}",
                    "metadata": {"task_label": title},
                },
            )

            usage = response.usage_metadata or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            model_name = response.response_metadata.get("model_name", "unknown")
            model_usage[model_name]["input_tokens"] += input_tokens
            model_usage[model_name]["output_tokens"] += output_tokens

            print(f"  модель в ответе: {model_name}")
            print(f"  вход {input_tokens}, выход {output_tokens}")
            print(f"  ответ: {response.text.strip()[:100]!r}")

        except ModelError as e:
            print(f"  ОШИБКА: {type(e).__name__}, is_retryable={e.is_retryable}")
        except Exception as e:
            print(f"  НЕОЖИДАННАЯ ОШИБКА: {type(e).__name__}, is_retryable=unknown")

        print()

    elapsed = time.time() - start_time
    print(f"Общее время прогона: {elapsed:.2f} секунд")
    print()

    print("ТАБЛИЦА РАСХОДА")
    print(f"{'Модель':<30} {'Вход':>10} {'Выход':>10} {'Стоимость':>12}")
    total_cost = 0.0
    for model_name, tokens in model_usage.items():
        cost_in = (tokens["input_tokens"] / 1_000_000) * PRICE_INPUT
        cost_out = (tokens["output_tokens"] / 1_000_000) * PRICE_OUTPUT
        cost = cost_in + cost_out
        total_cost += cost
        print(f"{model_name:<30} {tokens['input_tokens']:>10} {tokens['output_tokens']:>10} ${cost:>11.6f}")
    print(f"{'ИТОГО':<30} {'':>10} {'':>10} ${total_cost:>11.6f}")


if __name__ == "__main__":
    main()
