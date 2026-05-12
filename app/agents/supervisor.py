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

MAX_PARALLEL_SUPERVISOR_DISPATCH = 4
MAX_SUPERVISOR_TOOL_ROUNDS = 2

_DISPATCH_DEFERRED_MESSAGE = (
    "Лимит параллельных исследователей (4) на этот тур исчерпан. "
    "Объедините близкие маркеры в одну задачу или повторите в следующем раунде."
)


def _plan_supervisor_tool_calls(
    tool_calls: list[ToolCall],
    dispatch_name: str,
) -> tuple[list[tuple[int, ToolCall]], list[ToolMessage | None]]:
    n = len(tool_calls)
    dispatch_quota = 0
    runnable: list[tuple[int, ToolCall]] = []
    slots: list[ToolMessage | None] = [None] * n

    for i, tc in enumerate(tool_calls):
        name = tc["name"]

        if name == dispatch_name:
            if dispatch_quota < MAX_PARALLEL_SUPERVISOR_DISPATCH:
                dispatch_quota += 1
                runnable.append((i, tc))
            else:
                slots[i] = ToolMessage(
                    content=_DISPATCH_DEFERRED_MESSAGE,
                    name=dispatch_name,
                    tool_call_id=tc["id"] or "",
                )

        else:
            runnable.append((i, tc))

    return runnable, slots


SUPERVISOR_SYSTEM_PROMPT = """Ты supervisor исследования.
Тебе даны brief и draft с маркерами [RESEARCH_NEEDED].
Каждый маркер обычно — отдельный вызов dispatch_researcher с точной формулировкой задачи;
если несколько маркеров близки по теме, объединяй их в один dispatch_researcher.
Запускай независимые dispatch_researcher параллельно одним сообщением,
но не более 4 за раз (оркестратор тоже ограничивает).
Если открытых маркеров больше, чем помещается в один тур,
— выбери самые приоритетные сейчас, остальные в следующем раунде.
Максимум 2 раунда исследования; после второго раунда оркестратор принудительно переходит
к финальному отчёту без новых вызовов tools.
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
        """Запускает исследователя по узкой или объединённой задаче (маркер(ы) [RESEARCH_NEEDED]).

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
    dispatch_name = dispatch_tool.name

    async def _exec(tc: ToolCall) -> tuple[ToolMessage, str | None]:
        name = tc["name"]
        handler = handlers.get(name)
        call_id = tc["id"] or ""

        if handler is None:
            return ToolMessage(content=f"Неизвестный tool: {name}", name=name, tool_call_id=call_id), None

        result = await handler.ainvoke(tc["args"])
        note = result if name == dispatch_name else None

        return ToolMessage(content=result, name=name, tool_call_id=call_id), note

    async def supervisor_tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]

        match last:
            case AIMessage() if last.tool_calls:
                pass

            case _:
                return {"messages": []}

        tool_calls = last.tool_calls
        n = len(tool_calls)
        runnable, slots = _plan_supervisor_tool_calls(tool_calls, dispatch_name)

        async def _run_at(index: int, tc: ToolCall) -> tuple[int, ToolMessage, str | None]:
            tm, note = await _exec(tc)

            return index, tm, note

        gathered = await asyncio.gather(*(_run_at(i, tc) for i, tc in runnable))
        new_notes: list[str] = []

        for idx, tm, note in gathered:
            slots[idx] = tm

            if note is not None:
                new_notes.append(note)

        tool_messages = cast(list[ToolMessage], [slots[i] for i in range(n)])
        prev_rounds = state.get("completed_supervisor_tool_rounds", 0)

        return {
            "messages": tool_messages,
            "notes": new_notes,
            "completed_supervisor_tool_rounds": prev_rounds + 1,
        }

    return supervisor_tools_node


def route_after_supervisor(state: AgentState) -> Literal["tools", "write"]:
    last = state["messages"][-1]

    match last:
        case AIMessage() if last.tool_calls:
            return "tools"

        case _:
            return "write"


def route_after_tools(state: AgentState) -> Literal["supervisor_llm", "write"]:
    if state.get("completed_supervisor_tool_rounds", 0) >= MAX_SUPERVISOR_TOOL_ROUNDS:
        return "write"

    return "supervisor_llm"
