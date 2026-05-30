import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.base import BaseResearchAgent
from app.agents.patterns.brief import build_brief_node
from app.agents.patterns.researcher import build_researcher_graph
from app.agents.patterns.state import AgentState
from app.agents.patterns.supervisor import (
    build_dispatch_tool,
    build_supervisor_llm_node,
    build_supervisor_tools_node,
    route_after_supervisor,
    route_after_tools,
)
from app.agents.patterns.write import build_write_node
from app.enums import AssistantType
from app.services.research_run import (
    build_compound_run_label,
    compound_state_to_chunk,
)

WRITE_NODE_NAME = "write"

log = logging.getLogger(__name__)


def _build_compound_graph(llm: ChatAnthropic, compress_llm: ChatAnthropic) -> CompiledStateGraph[AgentState]:
    researcher_graph = build_researcher_graph(llm, compress_llm)
    dispatch_tool = build_dispatch_tool(researcher_graph)

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    # add_node не выводит NodeInputT для наших нод (Command-нода brief к тому же сбивает overload у соседей) —
    # гасим единообразно через Any, иначе mypy флапает коды ignore между arg-type и call-overload.
    graph.add_node("brief", cast(Any, build_brief_node(llm)))
    graph.add_node("supervisor_llm", cast(Any, build_supervisor_llm_node(llm, dispatch_tool)))
    graph.add_node("tools", cast(Any, build_supervisor_tools_node(dispatch_tool)))
    graph.add_node(WRITE_NODE_NAME, cast(Any, build_write_node(llm)))

    graph.add_edge(START, "brief")
    graph.add_conditional_edges(
        "supervisor_llm",
        route_after_supervisor,
        {"tools": "tools", "write": WRITE_NODE_NAME},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"supervisor_llm": "supervisor_llm", "write": WRITE_NODE_NAME},
    )
    graph.add_edge(WRITE_NODE_NAME, END)

    return graph.compile()


class CompoundResearchAgent(BaseResearchAgent):
    def __init__(
        self,
        llm: ChatAnthropic,
        compress_llm: ChatAnthropic,
        langfuse_callback_handler: CallbackHandler | None,
    ) -> None:
        super().__init__(llm)
        self._langfuse_callback_handler = langfuse_callback_handler
        self._graph = _build_compound_graph(llm, compress_llm)

    def _runnable_config(self) -> RunnableConfig | None:
        if self._langfuse_callback_handler is None:
            return None

        return RunnableConfig(callbacks=[self._langfuse_callback_handler])

    async def ainvoke_compound(
        self,
        messages: Sequence[BaseMessage],
        *,
        run_label: str | None = None,
    ) -> AgentState:
        config = self._runnable_config()
        initial: AgentState = {
            "input_messages": list(messages),
            "messages": [],
            "notes": [],
            "completed_supervisor_tool_rounds": 0,
        }

        if run_label:
            log.info("compound [%s]: запуск графа", run_label)
        else:
            log.info("compound: запуск графа")

        return cast(AgentState, await self._graph.ainvoke(initial, config=config))

    async def complete(self, messages: Sequence[BaseMessage]) -> AIMessageChunk:
        label = build_compound_run_label(AssistantType.COMPOUND, messages)
        state = await self.ainvoke_compound(messages, run_label=label)

        return compound_state_to_chunk(state)

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        config = self._runnable_config()
        initial: AgentState = {
            "input_messages": list(messages),
            "messages": [],
            "notes": [],
            "completed_supervisor_tool_rounds": 0,
        }

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
