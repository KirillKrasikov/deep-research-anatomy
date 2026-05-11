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
RESEARCHER_SYSTEM_PROMPT = """Ты исследователь по одной узкой задаче.
Используй web_search для поиска и think_tool для рефлексии.
После каждого web_search обязательно вызови think_tool: что нашёл, что осталось, следующий шаг.
Когда ответ полный и опирается на найденные источники — заверши без вызова tools.
Не выдумывай факты, которых нет в результатах поиска.

Сегодня: {today}.
"""


def _route_after_llm(state: ResearcherState) -> str:
    last = state["messages"][-1]
    match last:
        case AIMessage() if last.tool_calls:
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
    tools_node = ToolNode([think_tool])

    graph: StateGraph[ResearcherState] = StateGraph(ResearcherState)
    graph.add_node("researcher_llm", researcher_llm_node)
    graph.add_node("tools", tools_node)
    graph.add_node("compress", compress_node)  # type: ignore[arg-type]

    graph.add_edge(START, "researcher_llm")
    graph.add_conditional_edges("researcher_llm", _route_after_llm, {"tools": "tools", "compress": "compress"})
    graph.add_edge("tools", "researcher_llm")
    graph.add_edge("compress", END)

    return graph.compile()
