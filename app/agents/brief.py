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

Сегодня: {today}.
"""


def build_brief_node(llm: ChatAnthropic) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, str]]]:
    async def brief_node(state: AgentState) -> dict[str, str]:
        response = await llm.ainvoke(
            [
                SystemMessage(BRIEF_SYSTEM_PROMPT.format(today=today_iso())),
                HumanMessage(state["query"]),
            ],
        )

        return {"brief": content_to_text(response.content)}

    return brief_node
