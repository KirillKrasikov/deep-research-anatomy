import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessageChunk

from app.agents._state import AgentState

LOG = logging.getLogger(__name__)

STAGE_NODES = frozenset({"brief", "diffusion", "supervisor_llm", "tools", "write"})


class CompoundStageHandler(AsyncCallbackHandler):
    def __init__(self, progress_queue: asyncio.Queue[dict[str, Any]] | None = None) -> None:
        self._progress_queue = progress_queue

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (outputs, run_id, parent_run_id, tags)
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")

        if node not in STAGE_NODES:
            return

        LOG.info("Стадия compound: узел «%s» завершён", node)

        if self._progress_queue is None:
            return

        payload = {
            "object": "deep_research.stage",
            "node": node,
            "phase": "finished",
        }
        await self._progress_queue.put(payload)


def compound_state_to_chunk(state: AgentState) -> AIMessageChunk:
    report = state.get("final_report") or ""

    return AIMessageChunk(
        content=report,
        response_metadata={"stop_reason": "end_turn"},
    )


def compound_artifacts_from_state(state: AgentState) -> dict[str, Any]:
    return {
        "brief": state.get("brief") or "",
        "draft": state.get("draft") or "",
        "notes": state.get("notes") or [],
        "final_report": state.get("final_report") or "",
    }


async def iter_compound_progress_queue(
    queue: asyncio.Queue[dict[str, Any]],
    task: asyncio.Task[AgentState],
) -> AsyncIterator[dict[str, Any]]:
    while True:
        if task.done() and queue.empty():
            break

        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.05)

            yield item

        except TimeoutError:
            if task.done() and queue.empty():
                break

            continue

    while not queue.empty():
        yield await queue.get()
