from collections.abc import Callable, Coroutine
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._context import today_iso
from app.agents._state import AgentState
from app.agents._text import content_to_text

DIFFUSION_SYSTEM_PROMPT = """Составь draft ответа на запрос пользователя из своих знаний.
Жёсткое правило:
- структуру разделов и общеизвестные принципы пиши прямо;
- любые конкретные значения (даты, цены, адреса, расписания, имена нишевых событий,
  актуальные ссылки, фамилии участников) — НИКОГДА не выдумывай;
- вместо такого факта ставь маркер `[RESEARCH_NEEDED: точный вопрос для поиска]`.
Учитывай текущую дату: всё, что зависит от неё (расписания, цены, новые версии,
события «сейчас»), — почти всегда в маркеры.
Если задача про планирование или актуальные данные — почти весь draft будет в маркерах,
и это правильно: draft превращается в скелет с research-вопросами.
Разделы draft — только те, что прямо вытекают из запроса; не добавляй сопутствующие темы
и общеобразовательный контекст.
Группируй маркеры по теме: один маркер = один крупный смысловой блок (целый подраздел,
связанная группа фактов или таблица по теме), а не отдельная цифра, поле таблицы или микровопрос.
Не плоди десятки узких маркеров: при нехватке слотов сливай близкие темы в один маркер
усилением формулировки, а не количеством.
Итого маркеров — не более 8; если вопросов больше, объединяй близкие в один.
Markdown.

Сегодня: {today}.
"""


def build_diffusion_node(llm: ChatAnthropic) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, str]]]:
    async def diffusion_node(state: AgentState) -> dict[str, str]:
        context = f"# Запрос\n\n{state['query']}\n\n# Brief\n\n{state['brief']}"
        response = await llm.ainvoke(
            [
                SystemMessage(DIFFUSION_SYSTEM_PROMPT.format(today=today_iso())),
                HumanMessage(context),
            ],
        )

        return {"draft": content_to_text(response.content)}

    return diffusion_node
