import time
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.schemas import ChatMessageInput


def chat_messages_to_langchain(messages: Sequence[ChatMessageInput]) -> list[BaseMessage]:
    out: list[BaseMessage] = []

    for m in messages:
        match m.role:
            case "system":
                out.append(SystemMessage(content=m.content))

            case "user":
                if m.name:
                    out.append(HumanMessage(content=m.content, name=m.name))

                else:
                    out.append(HumanMessage(content=m.content))

            case "assistant":
                out.append(AIMessage(content=m.content))

            case "tool":
                tcid = m.tool_call_id or ""
                out.append(ToolMessage(content=m.content, tool_call_id=tcid))

    return out


def ai_chunk_content_to_text(chunk: AIMessageChunk) -> str:
    return _content_to_plain_text(chunk.content)


def _content_to_plain_text(content: Any) -> str:
    match content:
        case str() as text:
            return text

        case list() as blocks:
            parts: list[str] = []

            for block in blocks:
                match block:
                    case str() as fragment:
                        parts.append(fragment)

                    case dict() as mapping:
                        text_val = mapping.get("text")
                        if isinstance(text_val, str):
                            parts.append(text_val)

                    case _:
                        pass

            return "".join(parts)

        case _:
            return str(content) if content is not None else ""


def build_chat_completion_payload(
    *,
    model: str,
    chunk: AIMessageChunk,
    research_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = ai_chunk_content_to_text(chunk)

    meta = chunk.response_metadata or {}
    finish = meta.get("stop_reason")

    payload: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish if isinstance(finish, str) else "stop",
            },
        ],
        "usage": {},
    }

    if research_artifacts is not None:
        payload["brief"] = research_artifacts["brief"]
        payload["draft"] = research_artifacts["draft"]
        payload["notes"] = research_artifacts["notes"]
        payload["final_report"] = research_artifacts["final_report"]

    return payload
