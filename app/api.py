import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

import fastapi
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST

from app.agents.react_researcher import ReactResearchAgent
from app.container import Container
from app.schemas import ChatCompletionRequest
from app.services.chat_completion import (
    build_chat_completion_payload,
    chat_messages_to_langchain,
)

router = fastapi.APIRouter()


@router.get("/health")
async def health() -> None: ...


@router.get("/v1/models")
async def get_models() -> dict[str, Any]:
    raise NotImplementedError


@router.post("/v1/chat/completions", response_model=None)
@inject
async def create_chat_completion(
    request: ChatCompletionRequest,
    agent: Annotated[
        ReactResearchAgent,
        Depends(Provide[Container.react_researcher]),
    ],
) -> StreamingResponse | JSONResponse:
    if not any(m.role == "user" for m in request.messages):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Нужно хотя бы одно сообщение с role=user",
        )

    lc_messages = chat_messages_to_langchain(request.messages)
    final = await agent.complete(lc_messages)
    payload = build_chat_completion_payload(model=request.model, chunk=final)

    if request.stream:
        return StreamingResponse(_sse_final_only(payload), media_type="text/event-stream")

    return JSONResponse(content=payload)


async def _sse_final_only(payload: dict[str, Any]) -> AsyncIterator[str]:
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
