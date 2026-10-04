from langchain_core.language_models import ModelProfile

FIELDS: tuple = (
    "max_input_tokens",
    "max_output_tokens",
    "tool_calling",
    "structured_output",
    "reasoning_output",
    "reasoning_effort_levels",
    "temperature",
    "image_inputs",
    "pdf_inputs",
)



def show_profile(title: str, profile: ModelProfile):
    print(title)

    if not profile:
        print("  профиля нет: данных по этому имени модели в пакете нет")
        print()
        return

    for field in FIELDS:
        if field in profile:
            print(f"  {field:<24} {profile[field]}")
        else:
            print(f"  {field:<24} поля нет")

    print()
