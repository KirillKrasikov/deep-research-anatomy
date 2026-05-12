import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.agents._context import today_iso
from app.agents._state import AgentState, ResearcherState
from app.agents.think import think_tool

SUPERVISOR_SYSTEM_PROMPT = """Ты supervisor исследования.
Тебе даны brief и draft с маркерами [RESEARCH_NEEDED].
Каждый маркер — отдельный вызов dispatch_researcher с точной формулировкой задачи.
Запускай независимые dispatch_researcher параллельно одним сообщением, но не более 8 за раз.
Если маркеров больше 8 — выбери самые приоритетные сейчас, остальные закроешь в следующем раунде.
Максимум 3 раунда исследования; после третьего — заверши без вызовов tools, даже если маркеры остались.
Используй think_tool только если результаты неполные или нужно принять нетривиальное решение о следующем шаге.
Когда все маркеры закрыты — заверши без вызовов tools.
Не запускай research-задачи за пределами маркеров из brief и draft.

# Brief

{brief}

# Draft

{draft}

Сегодня: {today}.
"""


def build_dispatch_tool(researcher_graph: CompiledStateGraph[ResearcherState]) -> BaseTool:
    @tool
    async def dispatch_researcher(task: str) -> str:
        """Запускает исследователя по одной узкой задаче (например, по маркеру [RESEARCH_NEEDED]).

        Возвращает сжатые заметки в формате `тезисы [N] + Sources`.
        """
        result = await researcher_graph.ainvoke(
            {
                "task": task,
                "messages": [HumanMessage(f"Задача: {task}")],
            },
        )

        return cast(str, result["notes"])

    return dispatch_researcher


def build_supervisor_llm_node(
    llm: ChatAnthropic,
    dispatch_tool: BaseTool,
) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, list[AIMessage]]]]:
    supervisor_llm = llm.bind_tools([dispatch_tool, think_tool])

    async def supervisor_llm_node(state: AgentState) -> dict[str, list[AIMessage]]:
        sys_text = SUPERVISOR_SYSTEM_PROMPT.format(
            today=today_iso(),
            brief=state["brief"],
            draft=state["draft"],
        )
        history = state.get("messages") or [HumanMessage("Распредели исследование по brief и draft.")]

        response = await supervisor_llm.ainvoke([SystemMessage(sys_text), *history])

        return {"messages": [response]}

    return supervisor_llm_node


def build_supervisor_tools_node(
    dispatch_tool: BaseTool,
) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]:
    handlers: dict[str, BaseTool] = {
        dispatch_tool.name: dispatch_tool,
        think_tool.name: think_tool,
    }

    async def _exec(tc: ToolCall) -> tuple[ToolMessage, str | None]:
        name = tc["name"]
        handler = handlers.get(name)
        call_id = tc["id"] or ""

        if handler is None:
            return ToolMessage(content=f"Неизвестный tool: {name}", name=name, tool_call_id=call_id), None

        result = await handler.ainvoke(tc["args"])
        note = result if name == dispatch_tool.name else None

        return ToolMessage(content=result, name=name, tool_call_id=call_id), note

    async def supervisor_tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]

        match last:
            case AIMessage() if last.tool_calls:
                pairs = await asyncio.gather(*(_exec(tc) for tc in last.tool_calls))

            case _:
                return {"messages": []}

        tool_messages = [tm for tm, _ in pairs]
        new_notes = [note for _, note in pairs if note is not None]

        return {"messages": tool_messages, "notes": new_notes}

    return supervisor_tools_node


def route_after_supervisor(state: AgentState) -> Literal["tools", "write"]:
    last = state["messages"][-1]

    match last:
        case AIMessage() if last.tool_calls:
            return "tools"

        case _:
            return "write"
