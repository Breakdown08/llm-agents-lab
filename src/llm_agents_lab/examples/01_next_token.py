import math

from llm_agents_lab.course_model import build_model

# temperature=0 здесь не ради воспроизводимости, а чтобы модель брала самого
# вероятного кандидата и картинка распределения читалась однозначно.
model = build_model(temperature=0).bind(logprobs=True, top_logprobs=5)

response = model.invoke("Продолжите фразу тремя словами: столица Франции, это")

print("ОТВЕТ:", response.text)
print()

logprobs = response.response_metadata.get("logprobs")

if not logprobs:
    print("Провайдер не вернул logprobs.")
    print("Это не ошибка примера: поле необязательное, и часть шлюзов его режет.")
    print("Механика от этого не меняется, но увидеть её на своём ключе не выйдет.")
    raise SystemExit(0)

print("Первые пять позиций и кандидаты на каждую:")
for position in logprobs["content"][:5]:
    print(f"выбран: {position['token']!r}")
    for candidate in position.get("top_logprobs", []):
        # Провайдер отдаёт натуральный логарифм вероятности, exp возвращает
        # обратно долю от единицы.
        probability = math.exp(candidate["logprob"])
        print(f"    {candidate['token']!r:<20} {probability:.4f}")

# Вывод:
# ОТВЕТ: прекрасный город Париж
#
# Провайдер не вернул logprobs.
# Это не ошибка примера: поле необязательное, и часть шлюзов его режет.
# Механика от этого не меняется, но увидеть её на своём ключе не выйдет.
