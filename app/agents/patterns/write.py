from collections.abc import Callable, Coroutine
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.patterns.helpers import content_to_text, today_iso
from app.agents.patterns.state import AgentState

WRITE_SYSTEM_PROMPT = """Собери финальный отчёт по brief и заметкам researchers.

Целься в 4 оси качества:
- Comprehensiveness: покрой все ключевые вопросы из brief, не пропускай аспекты.
- Insight: давай сравнительные выводы, явные trade-offs и рекомендацию, а не только перечисление фактов.
- Instruction Following: отвечай на цель и ключевые вопросы brief в их постановке, не подменяй их.
- Readability: заголовки, короткие абзацы, таблицы для сравнений, ключевые числа — выделять.

Жёсткие правила:
- структура — по ключевым вопросам brief;
- все конкретные факты, цифры и URL — только из заметок, дословно;
- ничего не добавляй из своих знаний;
- ссылки в тексте — [N], где N = номер URL в итоговом `## Sources`;
- в конце раздел `## Sources` — дедуплицированный нумерованный список URL;
- без воды и без повторов между разделами.

Держись рамок brief: отвечай только на его цель и ключевые вопросы; не вводи разделы и темы, которых в brief нет;
общий контекст и смежные темы — только если без них ответ нельзя понять.
Comprehensiveness — это полное покрытие вопросов из brief, а не темы в целом.

Markdown.

Сегодня: {today}.
"""


def _format_notes(notes: list[str]) -> str:
    if not notes:
        return "_(заметок нет — researchers ничего не нашли)_"

    blocks = [f"## Заметка {i}\n\n{note}" for i, note in enumerate(notes, start=1)]

    return "\n\n".join(blocks)


def _report_context(brief: str, notes: list[str]) -> str:
    return f"# Brief\n\n{brief}\n\n# Заметки researchers\n\n{_format_notes(notes)}"


async def write_report(llm: ChatAnthropic, brief: str, notes: list[str]) -> str:
    """Паттерн #5 (Write): собирает финальный отчёт из brief и заметок researchers.

    Узкий контракт: brief и список заметок; на выходе — текст отчёта (Markdown).
    Ретраи — на уровне клиента ChatAnthropic; при сбое ошибка пробрасывается наверх.
    Не зависит от AgentState — переносится в свой пайплайн как есть.
    """
    context = _report_context(brief, notes)
    messages = [
        SystemMessage(WRITE_SYSTEM_PROMPT.format(today=today_iso())),
        HumanMessage(context),
    ]
    response = await llm.ainvoke(messages)

    return content_to_text(response.content)


def build_write_node(llm: ChatAnthropic) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, str]]]:
    """Адаптер write_report под ноду графа: читает brief/notes, пишет final_report."""

    async def write_node(state: AgentState) -> dict[str, str]:
        report = await write_report(llm, state["brief"], state.get("notes") or [])

        return {"final_report": report}

    return write_node
