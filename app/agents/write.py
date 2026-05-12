import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._context import today_iso
from app.agents._state import AgentState
from app.agents._text import content_to_text

LOG = logging.getLogger(__name__)

WRITE_LLM_MAX_ATTEMPTS = 3
WRITE_LLM_RETRY_DELAY_SEC = 1.5

HTTP_STATUS_GATEWAY_TIMEOUT = 504

WRITE_FALLBACK_PREAMBLE = """## Ошибка генерации финального отчёта

Не удалось собрать отчёт через модель после нескольких попыток (таймаут или ошибка провайдера).
Ниже сохранены запрос, brief, draft и заметки researchers без финальной сборки LLM.

---

"""

WRITE_SYSTEM_PROMPT = """Собери финальный отчёт по brief, draft и заметкам researchers.

Целься в 4 оси качества:
- Comprehensiveness: покрой все ключевые вопросы из brief, не пропускай аспекты.
- Insight: давай сравнительные выводы, явные trade-offs и рекомендацию, а не только перечисление фактов.
- Instruction Following: отвечай на исходный запрос пользователя в его постановке.
- Readability: заголовки, короткие абзацы, таблицы для сравнений, ключевые числа — выделять.

Жёсткие правила:
- структура — по draft (без маркеров [RESEARCH_NEEDED]);
- все конкретные факты, цифры и URL — только из заметок, дословно;
- ничего не добавляй из своих знаний;
- ссылайся [N], в конце раздел `## Sources` с дедуплицированным списком URL;
- без воды и без повторов между разделами.

Держись рамок запроса: отвечай только на то, о чём спросили; не вводи разделы и темы, которых пользователь не просил;
общий контекст и смежные темы — только если без них ответ нельзя понять.
Comprehensiveness — это полное покрытие вопросов из запроса, а не темы в целом.

Markdown.

Сегодня: {today}.
"""


def _is_write_retryable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)

    if code == HTTP_STATUS_GATEWAY_TIMEOUT:
        return True

    return type(exc).__name__ == "InternalServerError"


def _format_notes(notes: list[str]) -> str:
    if not notes:
        return "_(заметок нет — researchers ничего не нашли)_"

    blocks = [f"## Заметка {i}\n\n{note}" for i, note in enumerate(notes, start=1)]

    return "\n\n".join(blocks)


def _write_human_context(state: AgentState) -> str:
    return (
        f"# Запрос\n\n{state['query']}\n\n"
        f"# Brief\n\n{state['brief']}\n\n"
        f"# Draft\n\n{state['draft']}\n\n"
        f"# Заметки researchers\n\n{_format_notes(state.get('notes') or [])}"
    )


def build_write_node(llm: ChatAnthropic) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, str]]]:
    async def write_node(state: AgentState) -> dict[str, str]:
        context = _write_human_context(state)
        messages = [
            SystemMessage(WRITE_SYSTEM_PROMPT.format(today=today_iso())),
            HumanMessage(context),
        ]
        last_error: Exception | None = None

        for attempt in range(WRITE_LLM_MAX_ATTEMPTS):
            try:
                response = await llm.ainvoke(messages)

                return {"final_report": content_to_text(response.content)}

            except Exception as exc:
                last_error = exc

                if not _is_write_retryable(exc):
                    raise

                if attempt == WRITE_LLM_MAX_ATTEMPTS - 1:
                    break

                await asyncio.sleep(WRITE_LLM_RETRY_DELAY_SEC)

        LOG.warning(
            "Финальный отчёт не сгенерирован после %s попыток (запрос: %s): %s",
            WRITE_LLM_MAX_ATTEMPTS,
            state["query"][:200],
            last_error,
        )

        return {"final_report": WRITE_FALLBACK_PREAMBLE + context}

    return write_node
