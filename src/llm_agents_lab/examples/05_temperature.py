from llm_agents_lab.course_model import build_model

PROMPT = "Придумайте название для кофейни рядом с университетом. Ответьте только названием."


def show(title, model):
    response = model.invoke(PROMPT)
    # Поле output_token_details заполняет провайдер. Если он не разделяет
    # выход на рассуждение и ответ, словарь придёт пустым, и это тоже факт.
    details = response.usage_metadata.get("output_token_details") or {}
    reasoning = details.get("reasoning", "провайдер не разделил")
    print(f"{title:<24} {response.text.strip()!r}")
    print(f"{'':<24} выход {response.usage_metadata['output_tokens']} токенов, "
          f"из них на рассуждение: {reasoning}")
    return response.text.strip()


cold = build_model(temperature=0)
hot = build_model(temperature=1.5)

print("ТЕМПЕРАТУРА 0")
cold_1 = show("прогон 1", cold)
cold_2 = show("прогон 2", cold)

print()
print("ТЕМПЕРАТУРА 1.5")
hot_1 = show("прогон 1", hot)
hot_2 = show("прогон 2", hot)

print()
print(f"пара при 0 совпала:   {cold_1 == cold_2}")
print(f"пара при 1.5 совпала: {hot_1 == hot_2}")

print()
print("КОНТРОЛЬ: ОДНОСЛОВНЫЙ ОТВЕТ И СЧЁТЧИК ВЫХОДА")
control = cold.invoke("Столица Франции? Ответьте одним словом.")
print(f"  видимый текст:   {control.text.strip()!r}")
print(f"  счётчик выхода:  {control.usage_metadata['output_tokens']} токенов")

# Вывод:
# ТЕМПЕРАТУРА 0
# прогон 1                 'КофеСтудент'
#                          выход 192 токенов, из них на рассуждение: провайдер не разделил
# прогон 2                 'Кофейная пауза'
#                          выход 160 токенов, из них на рассуждение: провайдер не разделил
#
# ТЕМПЕРАТУРА 1.5
# прогон 1                 'Переменка'
#                          выход 230 токенов, из них на рассуждение: провайдер не разделил
# прогон 2                 'Кофе-Сессия'
#                          выход 161 токенов, из них на рассуждение: провайдер не разделил
#
# пара при 0 совпала:   False
# пара при 1.5 совпала: False
#
# КОНТРОЛЬ: ОДНОСЛОВНЫЙ ОТВЕТ И СЧЁТЧИК ВЫХОДА
#   видимый текст:   'Париж'
#   счётчик выхода:  67 токенов
