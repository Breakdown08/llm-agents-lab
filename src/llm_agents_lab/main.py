from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


def main():
    prompt = PromptTemplate.from_template(
        "Переведи на {language}: {text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    print(chain.run(language="французский", text="я люблю программировать"))


if __name__ == "__main__":
    main()
