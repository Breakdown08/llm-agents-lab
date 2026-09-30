from langchain.messages import AIMessage, HumanMessage

from llm_agents_lab.course_model import build_model

model = build_model(temperature=0)

print("ДВА ОТДЕЛЬНЫХ ВЫЗОВА")

first = model.invoke("Меня зовут Михаил. Запомните это.")
print("  вопрос 1: Меня зовут Михаил. Запомните это.")
print(f"  ответ 1:  {first.text}")
print(f"  вход:     {first.usage_metadata['input_tokens']} токенов")

second = model.invoke("Как меня зовут? Ответьте одним словом.")
print("  вопрос 2: Как меня зовут? Ответьте одним словом.")
print(f"  ответ 2:  {second.text}")
print(f"  вход:     {second.usage_metadata['input_tokens']} токенов")

print()
print("ТОТ ЖЕ ВТОРОЙ ВОПРОС, НО СО СПИСКОМ СООБЩЕНИЙ")

history = [
    HumanMessage("Меня зовут Михаил. Запомните это."),
    AIMessage(first.text),
    HumanMessage("Как меня зовут? Ответьте одним словом."),
]

third = model.invoke(history)
print(f"  ответ:    {third.text}")
print(f"  вход:     {third.usage_metadata['input_tokens']} токенов")

# Вывод:
# ДВА ОТДЕЛЬНЫХ ВЫЗОВА
#   вопрос 1: Меня зовут Михаил. Запомните это.
#   ответ 1:  Приятно познакомиться, Михаил. Я запомнил ваше имя. Чем могу быть полезен?
#   вход:     18 токенов
#   вопрос 2: Как меня зовут? Ответьте одним словом.
#   ответ 2:  Неизвестно
#   вход:     16 токенов
#
# ТОТ ЖЕ ВТОРОЙ ВОПРОС, НО СО СПИСКОМ СООБЩЕНИЙ
#   ответ:    Михаил.
#   вход:     61 токенов
