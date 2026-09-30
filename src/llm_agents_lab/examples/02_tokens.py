from langchain.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from llm_agents_lab.course_model import build_model

SAMPLES = [
    ".",
    "cat",
    "кот",
    "The quick brown fox jumps over the lazy dog",
    "Быстрая бурая лиса прыгает через ленивого пса",
    "1234567890",
    "достопримечательность",
]

# max_tokens режет ответ: платить за длинную генерацию здесь не за что,
# нас интересует только вход.
model = build_model(temperature=0, max_tokens=16)

print(f"{'текст':<48}{'симв.':>7}{'оценка':>8}{'реально':>9}")
print("-" * 72)

for text in SAMPLES:
    response = model.invoke(text)
    approximate = count_tokens_approximately([HumanMessage(text)])
    real = response.usage_metadata["input_tokens"]
    shown = text if len(text) <= 45 else text[:42] + "..."
    print(f"{shown:<48}{len(text):>7}{approximate:>8}{real:>9}")

# Вывод:
# текст                                             симв.  оценка  реально
# ------------------------------------------------------------------------
# .                                                     1       5        5
# cat                                                   3       5        5
# кот                                                   3       5        6
# The quick brown fox jumps over the lazy dog          43      15       13
# Быстрая бурая лиса прыгает через ленивого пса        45      16       22
# 1234567890                                           10       7        8
# достопримечательность                                21      10        9
