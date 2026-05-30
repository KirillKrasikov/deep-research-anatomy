import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from typing import Any, Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents._context import today_iso
from app.agents._state import SupervisorResearcherState
from app.agents._text import content_to_text
from app.agents.base import BaseResearchAgent
from app.agents.react_researcher import ReactResearchAgent

MAX_PARALLEL_DISPATCH = 4

_DISPATCH_DEFERRED_MESSAGE = (
    f"Лимит параллельных researchers ({MAX_PARALLEL_DISPATCH}) на этот раунд исчерпан. "
    "Объедините близкие вопросы или повторите задачу в следующем раунде."
)

SUPERVISOR_SYSTEM_PROMPT = """Ты supervisor исследования. Отдельных стадий brief и write нет —
их роли на тебе: ты сам декомпозируешь запрос и сам пишешь финальный отчёт.

Сначала преврати запрос пользователя в план: ключевые вопросы и узкие подзадачи (учитывай текущую дату,
не расширяй scope за пределы запроса). Раздавай подзадачи через dispatch_researcher: каждая подзадача —
отдельный вызов с точной формулировкой; независимые запускай параллельно одним сообщением,
но не более {max_parallel} за раз.
Оценивай ответы researchers и дозапускай недостающее, но не больше 3 итераций поиска;
останавливайся раньше, как только собранного достаточно.

Когда собранного достаточно — заверши без вызовов tools и напиши финальный отчёт:
- структура по ключевым вопросам запроса; целься в полноту покрытия, сравнительные выводы с рекомендацией,
  точное следование исходному запросу, читаемость (заголовки, короткие абзацы, таблицы для сравнений);
- факты, цифры и URL — только из ответов researchers, дословно; ничего из собственных знаний;
  не выдумывай ссылки: если URL для утверждения нет в ответах researchers — оставь его без ссылки;
- ссылайся [N], в конце раздел `## Sources` с дедуплицированным списком URL;
- держись рамок запроса, без лишних тем.

Сегодня: {today}.
"""

_RESEARCHER_TASK_TEMPLATE = """Задача: {task}

Заверши ответ разделом «## Sources» со списком реальных URL из результатов поиска (по строке на источник).
Если по задаче ничего не нашлось — прямо напиши об этом и не выдумывай источники."""


def build_dispatch_tool(researcher: ReactResearchAgent) -> BaseTool:
    @tool
    async def dispatch_researcher(task: str) -> str:
        """Запускает researcher (ReAct + web_search) по одной узкой задаче.

        Возвращает сырой ответ researcher с найденными фактами и источниками (URL).
        """
        chunk = await researcher.complete([HumanMessage(_RESEARCHER_TASK_TEMPLATE.format(task=task))])

        return content_to_text(chunk.content)

    return dispatch_researcher


def build_supervisor_llm_node(
    llm: ChatAnthropic,
    dispatch_tool: BaseTool,
) -> Callable[[SupervisorResearcherState], Coroutine[Any, Any, dict[str, list[AIMessage]]]]:
    """Шаг планирования: декомпозирует запрос, раздаёт задачи через dispatch_researcher либо сам пишет финал.

    Сколько раундов исследования делать и когда остановиться — решает сам супервайзер
    (финал = ответ без tool_calls).
    """
    supervisor_llm = llm.bind_tools([dispatch_tool])

    async def supervisor_llm_node(state: SupervisorResearcherState) -> dict[str, list[AIMessage]]:
        sys_text = SUPERVISOR_SYSTEM_PROMPT.format(today=today_iso(), max_parallel=MAX_PARALLEL_DISPATCH)
        response = await supervisor_llm.ainvoke([SystemMessage(sys_text), *state["messages"]])

        return {"messages": [response]}

    return supervisor_llm_node


def build_supervisor_tools_node(
    dispatch_tool: BaseTool,
) -> Callable[[SupervisorResearcherState], Coroutine[Any, Any, dict[str, Any]]]:
    dispatch_name = dispatch_tool.name

    async def _exec(tc: ToolCall) -> ToolMessage:
        result = await dispatch_tool.ainvoke(tc["args"])

        return ToolMessage(content=result, name=dispatch_name, tool_call_id=tc["id"] or "")

    async def supervisor_tools_node(state: SupervisorResearcherState) -> dict[str, Any]:
        last = state["messages"][-1]

        match last:
            case AIMessage() if last.tool_calls:
                tool_calls = last.tool_calls

            case _:
                return {"messages": []}

        runnable = tool_calls[:MAX_PARALLEL_DISPATCH]
        deferred = tool_calls[MAX_PARALLEL_DISPATCH:]

        executed = await asyncio.gather(*(_exec(tc) for tc in runnable))
        tool_messages: list[ToolMessage] = list(executed)
        tool_messages.extend(
            ToolMessage(content=_DISPATCH_DEFERRED_MESSAGE, name=tc["name"], tool_call_id=tc["id"] or "")
            for tc in deferred
        )

        return {"messages": tool_messages}

    return supervisor_tools_node


def route_after_supervisor(state: SupervisorResearcherState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]

    match last:
        case AIMessage() if last.tool_calls:
            return "tools"

        case _:
            return "__end__"


def _build_graph(llm: ChatAnthropic, researcher: ReactResearchAgent) -> CompiledStateGraph[SupervisorResearcherState]:
    dispatch_tool = build_dispatch_tool(researcher)

    graph: StateGraph[SupervisorResearcherState] = StateGraph(SupervisorResearcherState)
    # add_node не выводит NodeInputT для наших нод — гасим через Any, как в compound.
    graph.add_node("supervisor_llm", cast(Any, build_supervisor_llm_node(llm, dispatch_tool)))
    graph.add_node("tools", cast(Any, build_supervisor_tools_node(dispatch_tool)))

    graph.add_edge(START, "supervisor_llm")
    graph.add_conditional_edges(
        "supervisor_llm",
        route_after_supervisor,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "supervisor_llm")

    return graph.compile()


class SupervisorResearchAgent(BaseResearchAgent):
    def __init__(
        self,
        llm: ChatAnthropic,
        researcher: ReactResearchAgent,
        langfuse_callback_handler: CallbackHandler | None,
    ) -> None:
        super().__init__(llm)
        self._langfuse_callback_handler = langfuse_callback_handler
        self._graph = _build_graph(llm, researcher)

    def _invoke_config(self) -> RunnableConfig | None:
        if self._langfuse_callback_handler is None:
            return None

        return RunnableConfig(callbacks=[self._langfuse_callback_handler])

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        initial: SupervisorResearcherState = {"messages": list(messages)}
        state = cast(SupervisorResearcherState, await self._graph.ainvoke(initial, config=self._invoke_config()))
        final = state["messages"][-1]

        yield AIMessageChunk(content=content_to_text(final.content), response_metadata={"stop_reason": "end_turn"})
