from langchain.messages import AIMessage, HumanMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

from llm_agents_lab.course_model import build_model

# ПОДСТАВЬТЕ СВОИ СТАВКИ. Здесь цены модели курса на 22.09.2026, доллары за
# миллион токенов, ночной тариф. У вашего провайдера они другие и меняются
# часто.
PRICE_INPUT_MISS = 0.15
PRICE_INPUT_HIT = 0.003
PRICE_OUTPUT = 0.60

QUESTIONS = [
    "Назовите три задачи, где агент с инструментами выигрывает у одного запроса к модели.",
    "Возьмите первую из них и опишите, какие инструменты понадобятся.",
    "А теперь оцените, сколько вызовов модели уйдёт на один прогон такой задачи.",
]

callback = UsageMetadataCallbackHandler()
model = build_model(temperature=0)

history = []

for number, question in enumerate(QUESTIONS, start=1):
    history.append(HumanMessage(question))
    response = model.invoke(history, config={"callbacks": [callback]})
    history.append(AIMessage(response.text))

    usage = response.usage_metadata
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    print(
        f"ход {number}: вход {usage['input_tokens']} "
        f"(из кеша {cached}), выход {usage['output_tokens']}"
    )

print()
print("ИТОГ ПО ВСЕМ ВЫЗОВАМ")

for model_name, usage in callback.usage_metadata.items():
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    fresh = usage["input_tokens"] - cached
    money = (
        fresh * PRICE_INPUT_MISS
        + cached * PRICE_INPUT_HIT
        + usage["output_tokens"] * PRICE_OUTPUT
    ) / 1_000_000

    print(f"  модель: {model_name}")
    print(f"  вход:   {usage['input_tokens']} токенов, из них из кеша {cached}")
    print(f"  выход:  {usage['output_tokens']} токенов")
    print(f"  всего:  {usage['total_tokens']} токенов")
    print(f"  деньги: ${money:.6f}")

# Вывод:
# ход 1: вход 29 (из кеша 0), выход 1273
# ход 2: вход 899 (из кеша 0), выход 1587
# ход 3: вход 2129 (из кеша 0), выход 2395
#
# ИТОГ ПО ВСЕМ ВЫЗОВАМ
#   модель: deepseek/deepseek-v4-flash
#   вход:   3057 токенов, из них из кеша 0
#   выход:  5255 токенов
#   всего:  8312 токенов
#   деньги: $0.003612
