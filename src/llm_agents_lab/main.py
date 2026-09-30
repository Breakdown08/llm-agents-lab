import os
from importlib.metadata import version

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelAuthenticationError, ModelNotFoundError, ModelConnectionError


def main():
    load_dotenv()
    for package in ("langchain", "langchain-core", "langgraph"):
        print(f"{package}: {version(package)}")

    model_name = os.environ["MODEL_NAME"]
    base_url = os.getenv("MODEL_BASE_URL")

    if base_url:
        model = init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    else:
        model = init_chat_model(model_name, temperature=0)

    try:
        response = model.invoke("Answer, using a single word: abracadabra")
    except ModelAuthenticationError:
        print("[Ошибка]: аутентификация не пройдена, проверьте ваш API ключ")
        raise SystemExit(1)
    except ModelNotFoundError:
        print("[Ошибка]: модель не найдена, проверьте имя модели")
        raise SystemExit(1)
    except ModelConnectionError:
        print("[Ошибка]: проблемы с подключением, проверьте доступность сети")
        raise SystemExit(1)
    else:
        print(response.text)
        print(response.usage_metadata)
        raise SystemExit(0)


if __name__ == "__main__":
    main()
