"""Поиск по каталогу и устойчивый к ошибкам разбор модулей."""

from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    TodoListMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models import BaseChatModel


SYSTEM_PROMPT = """Вы агент-археолог, исследующий незнакомый каталог с кодом.
Отвечайте по-русски. Начинайте работу с плана; для многошаговой задачи ведите
план через write_todos и обновляйте статусы по мере выполнения.
Сначала найдите файлы через glob_search. Все пути виртуальные: / означает
корень исследуемого каталога. Не исследуйте .git, .venv, node_modules,
кеши, сборки и бинарные файлы; рассматривайте исходный код рекурсивно.
Чтобы сравнить размеры, вызовите module_report для каждого найденного файла
исходного кода. Он возвращает число строк и содержимое небольших файлов.
Обработайте все крупные модули, не останавливайтесь на первой ошибке.
Если module_report сообщает, что файл слишком велик, используйте grep_search
с output_mode='content': path должен быть каталогом файла, include — его
именем (например path='/src', include='main.py'). Найдите определения, импорты,
комментарии и другие характерные строки. Не повторяйте module_report после
такого сообщения. Следуйте подсказкам инструментов; если хранилище недоступно
и повторять вызов не нужно, не повторяйте его самостоятельно.
Сравните модули по числу строк и объясните назначение каждого крупного модуля
на основе кода. Явно отмечайте, какие файлы не удалось исследовать.
Если поиск не нашёл файлов, скажите, что искать нечего, и завершите работу.
Перед ответом обновите итоговые статусы задач, если вы заводили список.
"""


class ModuleReports:
    """Состояние ненадёжного хранилища на один запуск агента."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.attempts: Counter[str] = Counter()
        self._lock = Lock()

    def module_report(self, path: str) -> str:
        """Число строк и код файла до 1000 строк. Путь относительно корня, /foo.py допустим."""
        file = (self.root / path.lstrip("/")).resolve()
        if not file.is_relative_to(self.root):
            raise ValueError("путь должен находиться внутри исследуемого каталога")
        virtual_path = "/" + file.relative_to(self.root).as_posix()
        with self._lock:
            self.attempts[virtual_path] += 1
            attempt = self.attempts[virtual_path]
        if attempt == 1:
            raise ConnectionError("хранилище временно недоступно")
        # Считаем потоково, не загружая крупный файл целиком в память.
        preview = []
        line_count = 0
        with file.open(encoding="utf-8") as source:
            for line_count, line in enumerate(source, start=1):
                if line_count <= 1000:
                    preview.append(line)
        if line_count > 1000:
            raise ValueError(
                f"{virtual_path}: {line_count} строк; файл слишком велик для разбора целиком"
            )
        return f"{virtual_path}: {line_count} строк.\n{''.join(preview)}"


def on_tool_error(exc: Exception, request: ToolCallRequest) -> str | None:
    """Преобразует ожидаемые сбои в сообщения со статусом error."""
    name = request.tool_call["name"]
    if isinstance(exc, ConnectionError):
        return f"{name}: хранилище недоступно. Повторять вызов не нужно."
    if isinstance(exc, ValueError):
        return (
            f"{name}: {exc}. Используйте grep_search с output_mode='content' "
            "для выборочного разбора: укажите в path родительский каталог файла, "
            "а в include — имя файла. Не передавайте путь файла в path."
        )
    if isinstance(exc, (OSError, UnicodeError)):
        return f"{name}: файл недоступен или не является текстом ({type(exc).__name__})."
    return None


def create_archaeologist(
    root: Path, model: BaseChatModel, *, reverse_middleware: bool = False,
) -> tuple[Any, ModuleReports]:
    """Создаёт агента и счётчик реальных запусков инструмента."""
    reports = ModuleReports(root)
    error = ToolErrorMiddleware(on_tool_error)
    retry = ToolRetryMiddleware(
        max_retries=2,
        retry_on=(ConnectionError,),
        on_failure="error",
        initial_delay=0.2,
        jitter=False,
    )
    # Первый middleware — внешняя обёртка. Retry должен увидеть исключение
    # раньше, чем обработчик превратит его в обычный ToolMessage.
    layers = [retry, error] if reverse_middleware else [error, retry]
    agent = create_agent(
        model=model,
        tools=[reports.module_report],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            FilesystemFileSearchMiddleware(root_path=str(root), use_ripgrep=False),
            TodoListMiddleware(),
            *layers,
        ],
    )
    return agent, reports


def print_result(result: dict[str, Any], reports: ModuleReports) -> None:
    """Печатает три блока, включая ошибки и счётчики для проверки retry."""
    print("Список задач с итоговыми статусами:")
    for item in result.get("todos") or []:
        print(f"  [{item['status']}] {item['content']}")
    if not result.get("todos"):
        print("  Список задач пуст.")
    print("\nВызовы инструментов по порядку:")
    calls = [
        call["name"]
        for message in result["messages"] if message.type == "ai"
        for call in message.tool_calls
    ]
    print("  " + (" → ".join(calls) or "Вызовов не было."))
    for message in result["messages"]:
        if message.type == "tool" and message.status == "error":
            print(f"  {message.name}: status=error: {message.text}")
    for path, count in sorted(reports.attempts.items()):
        print(f"  Запусков module_report({path}): {count}")
    print("\nОтвет пользователю:")
    print(result["messages"][-1].text)
