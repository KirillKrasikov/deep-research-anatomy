from datetime import UTC, datetime
from typing import Any


def today_iso() -> str:
    """Текущая дата в UTC (ISO), для подстановки в конец системных промптов (prefix caching)."""
    return datetime.now(UTC).date().isoformat()


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
