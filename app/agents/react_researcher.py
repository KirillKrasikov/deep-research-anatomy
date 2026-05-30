from collections.abc import AsyncIterator, Sequence
from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

from app.agents._context import today_iso
from app.agents.base import BaseResearchAgent

WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
# Бюджет поисков на ход. По умолчанию — узкий (для роли sub-researcher в supervisor_researcher).
WEB_SEARCH_MAX_USES = 4
# Standalone-режим (model: react) отвечает на весь запрос сам — даём больше свободы по поиску.
WEB_SEARCH_MAX_USES_STANDALONE = 16
REACT_SYSTEM_PROMPT = """Ты исследователь. Не используй собственные знания —
любое утверждение должно быть подкреплено результатами поиска.
Ищи информацию до тех пор, пока не будешь готов ответить на вопрос полно и с источниками.
Отвечай на исходный запрос в его рамках, без расширения темы.

Сегодня: {today}.
"""


class ReactResearchAgent(BaseResearchAgent):
    def __init__(
        self,
        llm: ChatAnthropic,
        langfuse_callback_handler: CallbackHandler | None,
        *,
        web_search_max_uses: int = WEB_SEARCH_MAX_USES,
    ) -> None:
        super().__init__(llm)
        self._langfuse_callback_handler = langfuse_callback_handler
        self._llm_with_tools = llm.bind_tools(
            [
                {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": web_search_max_uses},
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
