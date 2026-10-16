"""Командная строка агента-археолога."""

import argparse
from pathlib import Path

from llm_agents_lab.archaeologist import create_archaeologist, print_result
from llm_agents_lab.course_model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="каталог с кодом")
    parser.add_argument(
        "question", nargs="?",
        default="Какие модули здесь самые большие и что в них происходит?",
    )
    parser.add_argument(
        "--reverse-middleware", action="store_true",
        help="поменять местами обработку ошибок и retry для сравнения",
    )
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"каталог не существует: {root}")
    agent, reports = create_archaeologist(
        root, build_model(temperature=0, max_tokens=4096),
        reverse_middleware=args.reverse_middleware,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.question}]},
        config={"recursion_limit": 200},
    )
    print_result(result, reports)


if __name__ == "__main__":
    main()
