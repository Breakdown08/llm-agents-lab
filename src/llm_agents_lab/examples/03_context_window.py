from langchain_anthropic import ChatAnthropic

from llm_agents_lab.course_model import build_model

model = build_model()
profile = model.profile

print("ПРОФИЛЬ МОДЕЛИ КУРСА")
if profile is None:
    print("Профиль не найден: для этого имени модели данных в пакете нет.")
    print("Так бывает у шлюзов, где имя модели выглядит как provider/model.")
    print("Размер окна тогда берётся со страницы вашего провайдера,")
    print("а в код его кладут параметром profile при создании модели.")
else:
    print(f"  max_input_tokens:  {profile.get('max_input_tokens')}")
    print(f"  max_output_tokens: {profile.get('max_output_tokens')}")
    print(f"  tool_calling:      {profile.get('tool_calling')}")
    print(f"  reasoning_output:  {profile.get('reasoning_output')}")

print()
print("ТРИ ИМЕНИ, У КОТОРЫХ ПРОФИЛЬ ЕСТЬ ВСЕГДА")
print("Ключ здесь не используется: profile читается из данных пакета,")
print("запрос к провайдеру не уходит.")

for name in ("claude-sonnet-4-5-20250929", "claude-opus-4-6", "claude-haiku-4-5-20251001"):
    known = ChatAnthropic(model=name, api_key="not-used-no-request-is-made")
    known_profile = known.profile or {}
    print(
        f"  {name:<12} вход до {known_profile.get('max_input_tokens')} токенов, "
        f"выход до {known_profile.get('max_output_tokens')}"
    )

# Вывод:
# ПРОФИЛЬ МОДЕЛИ КУРСА
# Профиль не найден: для этого имени модели данных в пакете нет.
# Так бывает у шлюзов, где имя модели выглядит как provider/model.
# Размер окна тогда берётся со страницы вашего провайдера,
# а в код его кладут параметром profile при создании модели.
#
# ТРИ ИМЕНИ, У КОТОРЫХ ПРОФИЛЬ ЕСТЬ ВСЕГДА
# Ключ здесь не используется: profile читается из данных пакета,
# запрос к провайдеру не уходит.
#   gpt-4o-mini  вход до 128000 токенов, выход до 16384
#   gpt-5-nano   вход до 272000 токенов, выход до 128000
#   gpt-5.5      вход до 1050000 токенов, выход до 128000
