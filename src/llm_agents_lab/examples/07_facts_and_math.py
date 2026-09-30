import re

from llm_agents_lab.course_model import build_model

A = 972_345_618_407
B = 851_209_366_773

model = build_model(temperature=0)

print("ПРОВЕРКА ПЕРВАЯ: АРИФМЕТИКА")
question = f"Посчитайте {A} * {B}. В ответе дайте только число, без пояснений."
answer = model.invoke(question)

truth = A * B
digits = re.sub(r"\D", "", answer.text)

print(f"  вопрос:        {A} * {B}")
print(f"  ответ модели:  {answer.text.strip()}")
print(f"  ответ Python:  {truth}")
print(f"  совпало:       {digits == str(truth)}")

print()
print("ПРОВЕРКА ВТОРАЯ: НЕСУЩЕСТВУЮЩАЯ ФУНКЦИЯ")
fake = model.invoke(
    "Опишите параметры функции create_agent_pool из LangChain "
    "и приведите пример вызова."
)
print(f"  {fake.text.strip()}")

print()
print("ТОТ ЖЕ ВОПРОС, НО С РАЗРЕШЕНИЕМ НЕ ЗНАТЬ")
honest = model.invoke(
    "Опишите параметры функции create_agent_pool из LangChain "
    "и приведите пример вызова. Если такой функции нет или вы не уверены, "
    "ответьте ровно одной фразой: не знаю."
)
print(f"  {honest.text.strip()}")

# Вывод:
# ПРОВЕРКА ПЕРВАЯ: АРИФМЕТИКА
#   вопрос:        972345618407 * 851209366773
#   ответ модели:  827669698128723562990611
#   ответ Python:  827669698128723562990611
#   совпало:       True
#
# ПРОВЕРКА ВТОРАЯ: НЕСУЩЕСТВУЮЩАЯ ФУНКЦИЯ
#   Функция `create_agent_pool` является **экспериментальной** и находится
#   в пакете `langchain_experimental.agent_pool`. Она предназначена для
#   создания пула агентов, которые могут обрабатывать запросы параллельно
#   или целенаправленно маршрутизировать их к конкретным агентам.
#
#   [ПОДРЕЗАНО. Дальше в ответе шли: таблица "Основные параметры" из шести
#    строк (agents, routing_strategy, reducer, max_concurrency, verbose,
#    kwargs) с типами и значениями по умолчанию, раздел про возвращаемый
#    объект AgentPool с интерфейсом Runnable, пример вызова на сорок строк
#    с импортом from langchain_experimental.agent_pool import create_agent_pool
#    и раздел "Примечания" из трёх пунктов, где названа версия пакета,
#    с которой функция доступна, и дана ссылка на страницу документации]
#
# ТОТ ЖЕ ВОПРОС, НО С РАЗРЕШЕНИЕМ НЕ ЗНАТЬ
#   не знаю
