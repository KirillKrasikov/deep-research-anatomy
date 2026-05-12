import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage

from app.agents._state import AgentState
from app.agents._text import content_to_text

LOG = logging.getLogger(__name__)

STAGE_NODES = frozenset({"brief", "diffusion", "supervisor_llm", "tools", "write"})


def build_compound_run_label(model: str, messages: Sequence[BaseMessage], preview_max: int = 120) -> str:
    text = ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = content_to_text(message.content)

            break

    one_line = " ".join(text.split())

    if len(one_line) > preview_max:
        one_line = one_line[: preview_max - 1] + "…"

    return f"{model} | {one_line}" if one_line else model


def _compound_stage_from_metadata(kwargs: dict[str, Any]) -> str | None:
    metadata = kwargs.get("metadata") or {}
    node = metadata.get("langgraph_node")

    if node not in STAGE_NODES:
        return None

    return str(node)


class CompoundStageHandler(AsyncCallbackHandler):
    def __init__(
        self,
        progress_queue: asyncio.Queue[dict[str, Any]] | None = None,
        *,
        run_label: str | None = None,
    ) -> None:
        self._progress_queue = progress_queue
        self._run_label = run_label

    def _log_ctx(self) -> str:
        if self._run_label:
            return f"compound [{self._run_label}]"

        return "compound"

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (serialized, inputs, run_id, parent_run_id, tags)
        node = _compound_stage_from_metadata(kwargs)

        if node is None:
            return

        LOG.info("%s: стадия «%s» начата", self._log_ctx(), node)

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
        node = _compound_stage_from_metadata(kwargs)

        if node is None:
            return

        LOG.info("%s: стадия «%s» завершена", self._log_ctx(), node)

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
