from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.agents._context import today_iso
from app.agents._state import ResearcherState
from app.agents.compress import build_compress_node
from app.agents.think import think_tool

WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
WEB_SEARCH_MAX_USES = 4
MAX_RESEARCHER_TOOL_ROUNDS = 2
RESEARCHER_SYSTEM_PROMPT = """Ты исследователь по одной узкой задаче.
Используй web_search для поиска и think_tool для рефлексии.
Используй think_tool, если находки неоднозначны или нужно скорректировать направление поиска.
Когда ответ полный и опирается на найденные источники — заверши без вызова tools.
Не более двух раундов с вызовами tools на задачу; затем заверши ответ без tool_calls.
Не выдумывай факты, которых нет в результатах поиска.
Ищи строго в рамках поставленной задачи; не уходи в смежные темы и сопутствующий контекст.

Сегодня: {today}.
"""


def _route_after_llm(state: ResearcherState) -> str:
    last = state["messages"][-1]
    rounds_done = state.get("completed_tool_rounds", 0)
    match last:
        case AIMessage() if last.tool_calls:
            if rounds_done >= MAX_RESEARCHER_TOOL_ROUNDS:
                return "compress"

            return "tools"

        case _:
            return "compress"


def build_researcher_graph(llm: ChatAnthropic) -> CompiledStateGraph[ResearcherState]:
    researcher_llm = llm.bind_tools(
        [
            {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": WEB_SEARCH_MAX_USES},
            think_tool,
        ],
    )

    async def researcher_llm_node(state: ResearcherState) -> dict[str, list[AIMessage]]:
        response = await researcher_llm.ainvoke(
            [SystemMessage(RESEARCHER_SYSTEM_PROMPT.format(today=today_iso())), *state["messages"]],
        )

        return {"messages": [response]}

    compress_node = build_compress_node(llm)
    think_tools = ToolNode([think_tool])

    async def tools_round_node(state: ResearcherState) -> dict[str, Any]:
        update = await think_tools.ainvoke(state)
        prev = state.get("completed_tool_rounds", 0)

        return {**cast(dict[str, Any], update), "completed_tool_rounds": prev + 1}

    graph: StateGraph[ResearcherState] = StateGraph(ResearcherState)
    graph.add_node("researcher_llm", researcher_llm_node)
    graph.add_node("tools", tools_round_node)
    graph.add_node("compress", compress_node)  # type: ignore[arg-type]

    graph.add_edge(START, "researcher_llm")
    graph.add_conditional_edges("researcher_llm", _route_after_llm, {"tools": "tools", "compress": "compress"})
    graph.add_edge("tools", "researcher_llm")
    graph.add_edge("compress", END)

    return graph.compile()
