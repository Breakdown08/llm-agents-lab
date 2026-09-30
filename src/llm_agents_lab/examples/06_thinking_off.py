from llm_agents_lab.course_model import build_model

PROMPT = "Придумайте название для кофейни рядом с университетом. Ответьте только названием."

model = build_model(temperature=0)

print("БЕЗ РЫЧАГОВ, ДЛЯ СРАВНЕНИЯ")
base = model.invoke(PROMPT)
print(f"  {base.text.strip()!r}, выход {base.usage_metadata['output_tokens']} токенов")

print()
print("РЫЧАГ 1: ВЫКЛЮЧИТЬ РЕЖИМ РАССУЖДЕНИЯ")
try:
    off = model.invoke(PROMPT, extra_body={"thinking": {"type": "disabled"}})
    print(f"  {off.text.strip()!r}, выход {off.usage_metadata['output_tokens']} токенов")
except Exception as error:  # noqa: BLE001
    print(f"  отказ: {type(error).__name__}")
    print(f"  текст: {error}")

print()
print("РЫЧАГ 2: СНИЗИТЬ УСИЛИЕ НА РАССУЖДЕНИЕ")
try:
    low = model.invoke(PROMPT, reasoning_effort="low")
    print(f"  {low.text.strip()!r}, выход {low.usage_metadata['output_tokens']} токенов")
except Exception as error:  # noqa: BLE001
    print(f"  отказ: {type(error).__name__}")
    print(f"  текст: {error}")

# Вывод:
# БЕЗ РЫЧАГОВ, ДЛЯ СРАВНЕНИЯ
#   'СтудКофейня', выход 237 токенов
#
# РЫЧАГ 1: ВЫКЛЮЧИТЬ РЕЖИМ РАССУЖДЕНИЯ
#   'Кофе-Конспект', выход 357 токенов
#
# РЫЧАГ 2: СНИЗИТЬ УСИЛИЕ НА РАССУЖДЕНИЕ
#   'Кофе-конспект', выход 434 токенов
