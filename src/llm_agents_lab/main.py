import anthropic
from dotenv import load_dotenv


def main():
    load_dotenv()
    client = anthropic.Anthropic()
    models = client.models.list()
    for model in models.data:
        print(model.id)


if __name__ == "__main__":
    main()
