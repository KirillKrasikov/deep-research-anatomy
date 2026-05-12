from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessageInput(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_tool(self) -> Self:
        if self.role == "tool" and not self.tool_call_id:
            msg = "Для role=tool нужен непустой tool_call_id"
            raise ValueError(msg)

        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessageInput] = Field(min_length=1)
    stream: bool = False
    stream_progress: bool = False
    include_research_artifacts: bool = False
    user: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] = Field(default_factory=dict)
