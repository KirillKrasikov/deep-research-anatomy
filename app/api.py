from typing import Any

import fastapi
from fastapi.responses import StreamingResponse

from app.schemas import ChatCompletionResponse

router = fastapi.APIRouter()


@router.get("/health")
async def health() -> None: ...


@router.get("/v1/models")
async def get_models() -> dict[str, Any]:
    raise NotImplementedError


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion() -> StreamingResponse | ChatCompletionResponse:
    raise NotImplementedError
