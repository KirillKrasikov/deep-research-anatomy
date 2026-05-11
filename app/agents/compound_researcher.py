from collections.abc import AsyncIterator, Sequence
from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents._state import AgentState
from app.agents._text import content_to_text
from app.agents.base import BaseResearchAgent
from app.agents.brief import build_brief_node
from app.agents.diffusion import build_diffusion_node
from app.agents.researcher import build_researcher_graph
from app.agents.supervisor import (
    build_dispatch_tool,
    build_supervisor_llm_node,
    build_supervisor_tools_node,
    route_after_supervisor,
)
from app.agents.write import build_write_node

WRITE_NODE_NAME = "write"


def _extract_query(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        match message:
            case HumanMessage():
                return content_to_text(message.content)

            case _:
                continue

    return ""


def _build_compound_graph(llm: ChatAnthropic) -> CompiledStateGraph[AgentState]:
    researcher_graph = build_researcher_graph(llm)
    dispatch_tool = build_dispatch_tool(researcher_graph)

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("brief", build_brief_node(llm))  # type: ignore[arg-type]
    graph.add_node("diffusion", build_diffusion_node(llm))  # type: ignore[arg-type]
    graph.add_node("supervisor_llm", build_supervisor_llm_node(llm, dispatch_tool))  # type: ignore[arg-type]
    graph.add_node("tools", build_supervisor_tools_node(dispatch_tool))  # type: ignore[arg-type]
    graph.add_node(WRITE_NODE_NAME, build_write_node(llm))  # type: ignore[arg-type]

    graph.add_edge(START, "brief")
    graph.add_edge("brief", "diffusion")
    graph.add_edge("diffusion", "supervisor_llm")
    graph.add_conditional_edges(
        "supervisor_llm",
        route_after_supervisor,
        {"tools": "tools", "write": WRITE_NODE_NAME},
    )
    graph.add_edge("tools", "supervisor_llm")
    graph.add_edge(WRITE_NODE_NAME, END)

    return graph.compile()


class CompoundResearchAgent(BaseResearchAgent):
    def __init__(
        self,
        llm: ChatAnthropic,
        langfuse_callback_handler: CallbackHandler | None,
    ) -> None:
        super().__init__(llm)
        self._langfuse_callback_handler = langfuse_callback_handler
        self._graph = _build_compound_graph(llm)

    def _invoke_config(self) -> RunnableConfig | None:
        if self._langfuse_callback_handler is None:
            return None

        return RunnableConfig(callbacks=[self._langfuse_callback_handler])

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        query = _extract_query(messages)
        config = self._invoke_config()
        initial: AgentState = {"query": query, "messages": [], "notes": []}

        async for event in self._graph.astream_events(initial, version="v2", config=config):
            if event["event"] != "on_chat_model_stream":
                continue

            metadata = event.get("metadata") or {}
            if metadata.get("langgraph_node") != WRITE_NODE_NAME:
                continue

            chunk = (event.get("data") or {}).get("chunk")
            if chunk is None:
                continue

            yield cast(AIMessageChunk, chunk)
