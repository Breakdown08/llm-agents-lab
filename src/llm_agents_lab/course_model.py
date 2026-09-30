import os

from dotenv import load_dotenv
from typing import Any
from langchain.chat_models import init_chat_model, BaseChatModel


load_dotenv()


def build_model(**kwargs: Any) -> BaseChatModel:
    model_name: str = os.environ["MODEL_NAME"]
    base_url: str | None = os.getenv("MODEL_BASE_URL")

    if base_url:
        return init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            **kwargs,
        )

    return init_chat_model(model_name, **kwargs)
