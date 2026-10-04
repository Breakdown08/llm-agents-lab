import os

from dotenv import load_dotenv
from typing import Any
from langchain.chat_models import init_chat_model, BaseChatModel


load_dotenv()


def gateway_kwargs() -> dict:
    base_url = os.getenv("MODEL_BASE_URL")

    if base_url:
        return {
            "model_provider": "anthropic",
            "base_url": base_url,
            "api_key": os.environ["ANTHROPIC_API_KEY"],
        }

    return {}


def build_model(model_name:str | None = None, **kwargs: Any) -> BaseChatModel:
    model_name = model_name or os.environ["MODEL_NAME"]
    access = gateway_kwargs()

    if access:
        return init_chat_model(model=model_name, **access, **kwargs)

    return init_chat_model(model_name, **kwargs)
