from llm_agents_lab.course_model import build_model
from langchain.messages import AIMessage, UsageMetadata


def build_prompt_translation(language: str, text: str) -> str:
    return f"Переведи на {language}: {text}"


def main():
    prompt: str = build_prompt_translation("французский", "я люблю программировать")

    response: AIMessage = build_model().invoke(prompt)
    usage_metadata: UsageMetadata | None = response.usage_metadata

    print(response.text)
    print(usage_metadata)


if __name__ == "__main__":
    main()
