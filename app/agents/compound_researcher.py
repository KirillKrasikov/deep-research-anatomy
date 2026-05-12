import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler
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
    route_after_tools,
)
from app.agents.write import build_write_node
from app.enums import AssistantType
from app.services.research_run import (
    CompoundStageHandler,
    build_compound_run_label,
    compound_state_to_chunk,
)

WRITE_NODE_NAME = "write"

LOG = logging.getLogger(__name__)


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
    graph.add_node("brief", build_brief_node(llm))  # type: ignore[call-overload]
    graph.add_node("diffusion", build_diffusion_node(llm))  # type: ignore[call-overload]
    graph.add_node("supervisor_llm", build_supervisor_llm_node(llm, dispatch_tool))  # type: ignore[call-overload]
    graph.add_node("tools", build_supervisor_tools_node(dispatch_tool))  # type: ignore[call-overload]
    graph.add_node(WRITE_NODE_NAME, build_write_node(llm))  # type: ignore[call-overload]

    graph.add_edge(START, "brief")
    graph.add_edge("brief", "diffusion")
    graph.add_edge("diffusion", "supervisor_llm")
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
        langfuse_callback_handler: CallbackHandler | None,
    ) -> None:
        super().__init__(llm)
        self._langfuse_callback_handler = langfuse_callback_handler
        self._graph = _build_compound_graph(llm)

    def _runnable_config(self, *extra_handlers: AsyncCallbackHandler) -> RunnableConfig | None:
        callbacks: list[Any] = []

        if self._langfuse_callback_handler is not None:
            callbacks.append(self._langfuse_callback_handler)

        callbacks.extend(extra_handlers)

        if not callbacks:
            return None

        return RunnableConfig(callbacks=callbacks)

    async def ainvoke_compound(
        self,
        messages: Sequence[BaseMessage],
        *,
        progress_queue: asyncio.Queue[dict[str, Any]] | None = None,
        run_label: str | None = None,
    ) -> AgentState:
        stage = CompoundStageHandler(progress_queue=progress_queue, run_label=run_label)
        config = self._runnable_config(stage)
        query = _extract_query(messages)
        initial: AgentState = {
            "query": query,
            "messages": [],
            "notes": [],
            "completed_supervisor_tool_rounds": 0,
        }

        if run_label:
            LOG.info("compound [%s]: запуск графа", run_label)
        else:
            LOG.info("compound: запуск графа")

        return cast(AgentState, await self._graph.ainvoke(initial, config=config))

    async def complete(self, messages: Sequence[BaseMessage]) -> AIMessageChunk:
        label = build_compound_run_label(AssistantType.COMPOUND, messages)
        state = await self.ainvoke_compound(messages, progress_queue=None, run_label=label)

        return compound_state_to_chunk(state)

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        query = _extract_query(messages)
        label = build_compound_run_label(AssistantType.COMPOUND, messages)
        stage = CompoundStageHandler(run_label=label)
        config = self._runnable_config(stage)
        initial: AgentState = {
            "query": query,
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
