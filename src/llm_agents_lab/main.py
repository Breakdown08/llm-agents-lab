from importlib.metadata import version


def main():
    print(version("langchain"))
    print(version("langchain-core"))
    print(version("langgraph"))


if __name__ == "__main__":
    main()
