from llm_agents_lab.course_model import build_model
from langchain.agents.middleware import AgentMiddleware
from llm_agents_lab.update_activity import UpdateActivityTransformer
from langchain.agents import create_agent


class UpdateActivityMiddleware(AgentMiddleware):
    """Middleware без единого хука: он нужен только ради проекции потока."""
    transformers = (UpdateActivityTransformer,)


def get_rub_per_vnd() -> float:
    """Возвращает стоимость 1 VND в RUB по курсу ЦБ РФ."""
    return 0.0033


def convert_vnd_to_rub(dongs: int) -> int:
    """Возвращает сконвертированную сумму вьетнамских донгов в российские рубли по актуальному курсу"""
    return round(dongs * get_rub_per_vnd())


def main():
    agent = create_agent(
        model=build_model(temperature=0, max_tokens=1024),
        tools=[get_rub_per_vnd, convert_vnd_to_rub],
        system_prompt="Отвечайте по-русски и коротко.",
        middleware=[UpdateActivityMiddleware()],
    )

    try:
        stream = agent.stream_events(
            {"messages": [{"role": "user", "content": "Сколько в рублях будет 150_000 донгов и с каким курсом ты это посчитал"}]},
            version="v3",
        )

        print("КЛЮЧИ EXTENSIONS:", sorted(stream.extensions))
        totals = iter(stream.extensions["update_totals"])
        print()

        order = []
        for name, item in stream.interleave("messages", "tool_calls", "values"):
            if name == "messages":
                order.append(f"messages(узел {item.node})")
            elif name == "tool_calls":
                order.append(f"tool_calls({item.tool_name}, args={item.input})")
            else:
                order.append(f"values(сообщений {len(item['messages'])})")
        for step, row in enumerate(order, start=1):
            print(f"  {step}. {row}")
        print()
        print("ОТВЕТ:", stream.output["messages"][-1].text)

        print("СЧЁТ ИЗ finalize():", next(totals, None))
    except Exception as error:
        print(f"\n[сбой] {type(error).__name__}: {error}")

if __name__ == "__main__":
    main()
