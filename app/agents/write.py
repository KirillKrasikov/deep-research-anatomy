from collections.abc import Callable, Coroutine
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._context import today_iso
from app.agents._state import AgentState
from app.agents._text import content_to_text

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

Markdown.

Сегодня: {today}.
"""


def _format_notes(notes: list[str]) -> str:
    if not notes:
        return "_(заметок нет — researchers ничего не нашли)_"

    blocks = [f"## Заметка {i}\n\n{note}" for i, note in enumerate(notes, start=1)]

    return "\n\n".join(blocks)


def build_write_node(llm: ChatAnthropic) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, str]]]:
    async def write_node(state: AgentState) -> dict[str, str]:
        context = (
            f"# Запрос\n\n{state['query']}\n\n"
            f"# Brief\n\n{state['brief']}\n\n"
            f"# Draft\n\n{state['draft']}\n\n"
            f"# Заметки researchers\n\n{_format_notes(state.get('notes') or [])}"
        )

        response = await llm.ainvoke(
            [
                SystemMessage(WRITE_SYSTEM_PROMPT.format(today=today_iso())),
                HumanMessage(context),
            ],
        )

        return {"final_report": content_to_text(response.content)}

    return write_node
