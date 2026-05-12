from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents._context import today_iso
from app.agents._state import ResearcherState
from app.agents._text import content_to_text

COMPRESS_SYSTEM_PROMPT = """Сожми ход поиска в заметки.
Факты, цифры, имена и URL — дословно, без перефразирования.
Формат:
- тезисы вида `- утверждение [N]`;
- в конце раздел `## Sources` со списком `[N] url — название`.
Не выдумывай факты, которых нет в ленте. Если по задаче ничего не нашлось — так и напиши.
Фиксируй только то, что относится к поставленной задаче; всё выходящее за её рамки — отбрасывай.
Если последний ответ ассистента содержит невыполненные вызовы tools — сожми только то, что уже есть в ленте поиска.

Сегодня: {today}.
"""


def _serialize_block(block: Any) -> str:
    match block:
        case str() as text:
            return text

        case {"type": "text", "text": str() as text}:
            return text

        case {"type": "thinking", "thinking": str() as text}:
            return f"[thinking] {text}"

        case {"type": "server_tool_use", "name": str() as name, "input": dict() as args}:
            query = args.get("query", args)

            return f"[search:{name}] {query}"

        case {"type": "web_search_tool_result", "content": list() as results}:
            lines: list[str] = []

            for r in results:
                match r:
                    case {"url": str() as url, "title": str() as title}:
                        snippet = r.get("content") if isinstance(r.get("content"), str) else ""
                        lines.append(f"- {url} — {title} :: {snippet}")

                    case _:
                        pass

            return "[results]\n" + "\n".join(lines)

        case _:
            return ""


def _serialize_message(message: BaseMessage) -> str:
    match message:
        case HumanMessage():
            return f"### User\n{content_to_text(message.content)}"

        case AIMessage() if isinstance(message.content, list):
            parts = [_serialize_block(b) for b in message.content]

            return "### Assistant\n" + "\n".join(p for p in parts if p)

        case AIMessage():
            return f"### Assistant\n{content_to_text(message.content)}"

        case ToolMessage():
            return f"### Tool ({message.name or 'tool'})\n{content_to_text(message.content)}"

        case _:
            return ""


def serialize_trail(messages: Sequence[BaseMessage]) -> str:
    return "\n\n".join(_serialize_message(m) for m in messages if m)


def build_compress_node(llm: ChatAnthropic) -> Callable[[ResearcherState], Coroutine[Any, Any, dict[str, str]]]:
    async def compress_node(state: ResearcherState) -> dict[str, str]:
        task = state["task"]
        trail = serialize_trail(state["messages"])

        response = await llm.ainvoke(
            [
                SystemMessage(COMPRESS_SYSTEM_PROMPT.format(today=today_iso())),
                HumanMessage(f"# Задача\n\n{task}\n\n# Лента поиска\n\n{trail}"),
            ],
        )

        return {"notes": content_to_text(response.content)}

    return compress_node
