from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain.messages import HumanMessage, AIMessage, UsageMetadata
from llm_agents_lab.course_model import build_model
from llm_agents_lab.token_budget import make_checker, Checker, show_total_cost


user_input: str
history: list[BaseMessage] = []
model: BaseChatModel


def main():
    model: BaseChatModel = build_model()
    dialog_loop(model)


def dialog_loop(model: BaseChatModel):
    check: Checker = make_checker()
    ai_message: AIMessage | None = None

    print("Dialog session started! Text your message to agent...")
    while True:
        user_input = input()
        if "QUIT" in user_input: break
        if user_input == "": continue

        human_message: HumanMessage = HumanMessage(user_input)
        history.append(human_message)

        ai_message: AIMessage = model.invoke(history)

        print(ai_message.text)

        check(model, ai_message, history)

        history.append(ai_message)

    print("Dialog session successfully executed.")
    show_total_cost(model, ai_message)


if __name__ == "__main__":
    main()
