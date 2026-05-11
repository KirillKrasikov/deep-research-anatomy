from collections.abc import AsyncIterator, Sequence
from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

from app.agents._context import today_iso
from app.agents.base import BaseResearchAgent

WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
WEB_SEARCH_MAX_USES = 4
REACT_SYSTEM_PROMPT = """Ты исследователь. Не используй собственные знания —
любое утверждение должно быть подкреплено результатами поиска.
Ищи информацию до тех пор, пока не будешь готов ответить на вопрос полно и с источниками.

Сегодня: {today}.
"""


class ReactResearchAgent(BaseResearchAgent):
    def __init__(
        self,
        llm: ChatAnthropic,
        langfuse_callback_handler: CallbackHandler | None,
    ) -> None:
        super().__init__(llm)
        self._langfuse_callback_handler = langfuse_callback_handler
        self._llm_with_tools = llm.bind_tools(
            [
                {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": WEB_SEARCH_MAX_USES},
            ],
        )

    def _invoke_config(self) -> RunnableConfig | None:
        if self._langfuse_callback_handler is None:
            return None

        return RunnableConfig(callbacks=[self._langfuse_callback_handler])

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        full_turn: list[BaseMessage] = [SystemMessage(REACT_SYSTEM_PROMPT.format(today=today_iso())), *messages]
        invoke_config = self._invoke_config()
        async for chunk in self._llm_with_tools.astream(full_turn, config=invoke_config):
            # После bind_tools stubs дают AIMessage; фактически это AIMessageChunk
            yield cast(AIMessageChunk, chunk)
