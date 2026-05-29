from collections.abc import Callable, Coroutine
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._context import today_iso
from app.agents._state import AgentState
from app.agents._text import content_to_text

BRIEF_SYSTEM_PROMPT = """Преобразуй запрос пользователя в research brief.
Сформулируй: уточнённую цель, ключевые вопросы (столько, сколько нужно для полного ответа),
параллельные подзадачи, известный из запроса контекст.
Учитывай текущую дату при оценке актуальности вопросов.
Ключевые вопросы — только то, что нужно для ответа на исходный запрос; не расширяй scope в сторону темы в целом.

Сегодня: {today}.
"""


async def generate_brief(llm: ChatAnthropic, query: str) -> str:
    """Паттерн #3 (Brief): превращает запрос пользователя в research brief.

    Узкий контракт: на входе только текст запроса, на выходе — текст брифа.
    Не зависит от AgentState — переносится в свой пайплайн как есть.
    """
    response = await llm.ainvoke(
        [
            SystemMessage(BRIEF_SYSTEM_PROMPT.format(today=today_iso())),
            HumanMessage(query),
        ],
    )

    return content_to_text(response.content)


def build_brief_node(llm: ChatAnthropic) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, str]]]:
    """Адаптер generate_brief под ноду compound-графа: читает query, пишет brief."""

    async def brief_node(state: AgentState) -> dict[str, str]:
        return {"brief": await generate_brief(llm, state["query"])}

    return brief_node
