from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage

from app.agents.patterns.helpers import content_to_text
from app.agents.patterns.state import AgentState


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


def compound_state_to_chunk(state: AgentState) -> AIMessageChunk:
    report = state.get("final_report") or ""

    return AIMessageChunk(
        content=report,
        response_metadata={"stop_reason": "end_turn"},
    )


def compound_artifacts_from_state(state: AgentState) -> dict[str, Any]:
    return {
        "brief": state.get("brief") or "",
        "notes": state.get("notes") or [],
        "final_report": state.get("final_report") or "",
    }
