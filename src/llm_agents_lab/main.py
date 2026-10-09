from langchain.chat_models import BaseChatModel
from langchain.agents import create_agent
from llm_agents_lab.course_model import build_model
from langchain.messages import AIMessage, AIMessageChunk
import urllib.request
import xml.etree.ElementTree as ET
import time


def get_rub_per_vnd() -> float:
    """Возвращает стоимость 1 VND в RUB по курсу ЦБ РФ."""
    url = "https://www.cbr.ru/scripts/XML_daily.asp"

    with urllib.request.urlopen(url, timeout=10) as response:
        root = ET.parse(response).getroot()

    for item in root.findall("Valute"):
        valute: ET.Element = item
        if valute.findtext("CharCode") == "VND":
            nominal = valute.find("Nominal")
            value = valute.find("Value")

            if nominal is None or nominal.text is None:
                raise ValueError("Отсутствует Nominal для VND")
            if value is None or value.text is None:
                raise ValueError("Отсутствует Value для VND")

            return float(value.text.replace(",", ".")) / int(nominal.text)
    raise ValueError("Валюта VND не найдена в ответе ЦБ РФ")


def convert_vnd_to_rub(dongs: int) -> int:
    """Возвращает сконвертированную сумму вьетнамских донгов в российские рубли по актуальному курсу"""
    return round(dongs * get_rub_per_vnd())



def get_order_price_vnd(
        black_coffee_count: int | None=None,
        milk_coffee_count: int | None=None,
        egg_coffee_count: int | None=None,
        salt_coffee_count: int | None=None,
) -> int:
    """Возвращает сумму заказа в кофейне по переданному количеству возможных типов кофе в донгах"""
    total_price: int = 0
    if black_coffee_count: total_price += black_coffee_count * 20_000
    if milk_coffee_count: total_price += milk_coffee_count * 27_000
    if egg_coffee_count: total_price += egg_coffee_count * 30_000
    if salt_coffee_count: total_price += salt_coffee_count * 25_000
    return total_price


def main():
    agent = create_agent(
        model=build_model(temperature=0, max_tokens=1024),
        tools=[get_order_price_vnd, convert_vnd_to_rub],
        system_prompt="You are a multilingual online waiter of a Vietnamese coffee shop.",
    )
    started = time.perf_counter()
    first_text_at = None

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "Здравствуйте, один яичный и один соленый кофе пожалуйста, сколько с меня в рублях?"}]},
            stream_mode=["updates", "messages", "custom"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                # В поток режима messages попадают и готовые ToolMessage, и обрывки
                # вызовов. Пользователю показывается только текст ответа модели.
                if not isinstance(token, AIMessageChunk) or not token.text:
                    continue
                if metadata.get("langgraph_node") != "model":
                    continue
                if first_text_at is None:
                    first_text_at = time.perf_counter() - started
                print(token.text, end="", flush=True)

            elif chunk["type"] == "custom":
                event = chunk["data"]
                print(f"\n[загрузка] страница {event['page']} из {event['of']}, строк {event['rows']}")

            elif chunk["type"] == "updates":
                for _node, update in chunk["data"].items():
                    messages = update.get("messages", []) if isinstance(update, dict) else []
                    for message in messages:
                        if isinstance(message, AIMessage) and message.tool_calls:
                            for call in message.tool_calls:
                                print(f"\n[инструмент] {call['name']}({call['args']})")

    except Exception as error:
        # Отказ провайдера посреди потока: часть ответа уже показана пользователю.
        print(f"\n[сбой] {type(error).__name__}: {error}")

    print()
    if first_text_at is None:
        # Защитная ветка: текста в потоке не было вовсе, показывать было нечего.
        print("первый знак ответа: текста в потоке не было")
    else:
        print(f"первый знак ответа через: {first_text_at:.2f} с")
    print(f"весь ответ через: {time.perf_counter() - started:.2f} с")


if __name__ == "__main__":
    main()
