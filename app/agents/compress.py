import asyncio
import logging
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents._context import today_iso
from app.agents._state import ResearcherState
from app.agents._text import content_to_text

LOG = logging.getLogger(__name__)

COMPRESS_SYSTEM_PROMPT = """Сожми ход поиска в заметки.
Факты, цифры, имена и URL — дословно, без перефразирования.
Формат ответа — только маркированный список тезисов вида `- утверждение [N]`
и в конце раздел `## Sources` со списком строк `[N] url — название`.
Не добавляй других разделов и заголовков (в том числе «Результаты поиска», итоговые отчёты,
преамбулы вроде «я проанализировал»).
Не продолжай и не воспроизводи шаблон ленты: запрещены заголовки ролей чата,
блоки `[search:...]`, `[results]`, `[thinking]` и любая имитация новых вызовов инструментов
или дорисовка хода поиска.
Не выдумывай факты, которых нет в ленте. Если по задаче ничего не нашлось — так и напиши.
Фиксируй только то, что относится к поставленной задаче; всё выходящее за её рамки — отбрасывай.
Если последний ответ ассистента содержит невыполненные вызовы tools — сожми только то, что уже есть в ленте поиска.

Сегодня: {today}.
"""

COMPRESS_TRAIL_HUMAN_INTRO = """Ниже — сырая лента наблюдений (внутренний формат).
Не копируй её заголовки и разметку в ответ; извлеки только факты для тезисов и Sources."""

COMPRESS_FAILED_NOTES = (
    "Исследователь не смог сжать результаты поиска (таймаут или ошибка провайдера LLM). "
    "Повторите dispatch_researcher с той же или уточнённой задачей."
)

COMPRESS_LLM_MAX_ATTEMPTS = 3
COMPRESS_LLM_RETRY_DELAY_SEC = 1.5

HTTP_STATUS_GATEWAY_TIMEOUT = 504


def _is_compress_retryable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)

    if code == HTTP_STATUS_GATEWAY_TIMEOUT:
        return True

    return type(exc).__name__ == "InternalServerError"


def _serialize_block(block: Any) -> str:
    match block:
        case str() as text:
            return text

        case {"type": "text", "text": str() as text}:
            return text

        case {"type": "thinking", "thinking": str() as text}:
            return f"[thinking] {text}"

        case {"type": "server_tool_use", "name": str() as name, "input": dict() as args}:
            query = args.get("query", args)

            return f"[search:{name}] {query}"

        case {"type": "web_search_tool_result", "content": list() as results}:
            lines: list[str] = []

            for r in results:
                match r:
                    case {"url": str() as url, "title": str() as title}:
                        snippet = r.get("content") if isinstance(r.get("content"), str) else ""
                        lines.append(f"- {url} — {title} :: {snippet}")

                    case _:
                        pass

            return "[results]\n" + "\n".join(lines)

        case _:
            return ""


def _serialize_message(message: BaseMessage) -> str:
    match message:
        case HumanMessage():
            return f"### Запрос\n{content_to_text(message.content)}"

        case AIMessage() if isinstance(message.content, list):
            parts = [_serialize_block(b) for b in message.content]

            return "### Вывод модели\n" + "\n".join(p for p in parts if p)

        case AIMessage():
            return f"### Вывод модели\n{content_to_text(message.content)}"

        case ToolMessage():
            tool_label = message.name or "tool"

            return f"### Заметка инструмента ({tool_label})\n{content_to_text(message.content)}"

        case _:
            return ""


def serialize_trail(messages: Sequence[BaseMessage]) -> str:
    return "\n\n".join(_serialize_message(m) for m in messages if m)


def build_compress_node(llm: ChatAnthropic) -> Callable[[ResearcherState], Coroutine[Any, Any, dict[str, str]]]:
    async def compress_node(state: ResearcherState) -> dict[str, str]:
        task = state["task"]
        trail = serialize_trail(state["messages"])
        messages = [
            SystemMessage(COMPRESS_SYSTEM_PROMPT.format(today=today_iso())),
            HumanMessage(
                f"# Задача\n\n{task}\n\n{COMPRESS_TRAIL_HUMAN_INTRO}\n\n# Лента для сжатия\n\n{trail}",
            ),
        ]
        last_error: Exception | None = None

        for attempt in range(COMPRESS_LLM_MAX_ATTEMPTS):
            try:
                response = await llm.ainvoke(messages)

                return {"notes": content_to_text(response.content)}

            except Exception as exc:
                last_error = exc

                if not _is_compress_retryable(exc):
                    raise

                if attempt == COMPRESS_LLM_MAX_ATTEMPTS - 1:
                    break

                await asyncio.sleep(COMPRESS_LLM_RETRY_DELAY_SEC)

        LOG.warning(
            "Сжатие ленты исследователя не удалось после %s попыток (задача: %s): %s",
            COMPRESS_LLM_MAX_ATTEMPTS,
            task[:200],
            last_error,
        )

        return {"notes": COMPRESS_FAILED_NOTES}

    return compress_node
