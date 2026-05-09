from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, BaseMessage


class BaseResearchAgent(ABC):
    def __init__(self, llm: ChatAnthropic) -> None:
        self._llm = llm

    @abstractmethod
    def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]: ...

    async def complete(self, messages: Sequence[BaseMessage]) -> AIMessageChunk:
        last: AIMessageChunk | None = None

        async for chunk in self.astream(messages):
            last = chunk

        if last is None:
            return AIMessageChunk(content="")

        return last
