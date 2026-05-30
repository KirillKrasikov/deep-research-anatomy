from collections.abc import Callable, Coroutine, Sequence
from typing import Any, Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents._context import today_iso
from app.agents._state import AgentState

BRIEF_SYSTEM_PROMPT = """Преобразуй запрос пользователя в research brief.
Сформулируй: уточнённую цель, ключевые вопросы (столько, сколько нужно для полного ответа),
параллельные подзадачи, известный из запроса контекст.
Учитывай текущую дату при оценке актуальности вопросов.
Ключевые вопросы — только то, что нужно для ответа на исходный запрос; не расширяй scope в сторону темы в целом.
Сохраняй формулировку, ключевые термины и явные ограничения пользователя дословно.

Если для ответа критически не хватает данных (неоднозначный scope, отсутствует ключевой параметр) —
поставь needs_clarification=true и задай в questions короткие уточняющие вопросы вместо брифа.
Учитывай всю историю диалога: ответы пользователя на прежние уточнения снимают неоднозначность.

Сегодня: {today}.
"""


class BriefDecision(BaseModel):
    """Решение стадии brief (human-in-the-loop): хватает ли данных или нужно уточнение у пользователя."""

    needs_clarification: bool = Field(description="True, если для ответа критически не хватает данных.")
    questions: str = Field(default="", description="Короткие уточняющие вопросы пользователю при needs_clarification.")
    brief: str = Field(default="", description="Research brief, если данных достаточно.")


async def generate_brief(llm: ChatAnthropic, messages: Sequence[BaseMessage]) -> BriefDecision:
    """Паттерн #3 (Brief): превращает диалог с пользователем в research brief либо в уточняющий вопрос.

    На входе — история сообщений (для замыкания loop'а уточнений), на выходе — типизированное решение.
    Не зависит от AgentState — переносится в свой пайплайн как есть.
    """
    structured = llm.with_structured_output(BriefDecision)
    response = await structured.ainvoke(
        [
            SystemMessage(BRIEF_SYSTEM_PROMPT.format(today=today_iso())),
            *messages,
        ],
    )

    return cast(BriefDecision, response)


def build_brief_node(
    llm: ChatAnthropic,
) -> Callable[[AgentState], Coroutine[Any, Any, Command[Literal["supervisor_llm", "__end__"]]]]:
    """Адаптер generate_brief под ноду compound-графа: сама решает ветку через Command (handoff)."""

    async def brief_node(state: AgentState) -> Command[Literal["supervisor_llm", "__end__"]]:
        decision = await generate_brief(llm, state["input_messages"])

        if decision.needs_clarification:
            return Command(goto=cast(Literal["__end__"], END), update={"final_report": decision.questions})

        return Command(goto="supervisor_llm", update={"brief": decision.brief})

    return brief_node
