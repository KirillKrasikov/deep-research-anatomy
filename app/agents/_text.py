from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage


def content_to_text(content: Any) -> str:
    match content:
        case str() as text:
            return text

        case list() as blocks:
            parts: list[str] = []

            for block in blocks:
                match block:
                    case str() as text:
                        parts.append(text)

                    case {"text": str() as text}:
                        parts.append(text)

                    case _:
                        pass

            return "".join(parts)

        case _:
            return ""


def last_user_text(messages: Sequence[BaseMessage]) -> str:
    """Текст последнего сообщения пользователя — узкая research-цель для write."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return content_to_text(message.content)

    return ""
