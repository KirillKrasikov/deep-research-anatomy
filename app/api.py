import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any

import fastapi
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST

from app.agents.base import BaseResearchAgent
from app.agents.compound_researcher import CompoundResearchAgent
from app.container import Container
from app.enums import AssistantType
from app.schemas import ChatCompletionRequest
from app.services.chat_completion import (
    build_chat_completion_payload,
    chat_messages_to_langchain,
)
from app.services.research_run import (
    build_compound_run_label,
    compound_artifacts_from_state,
    compound_state_to_chunk,
    iter_compound_progress_queue,
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
    react_factory: Annotated[
        Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
        Depends(Provide[Container.react_researcher.provider]),
    ],
    compound_factory: Annotated[
        Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
        Depends(Provide[Container.compound_researcher.provider]),
    ],
) -> StreamingResponse | JSONResponse:
    if not any(m.role == "user" for m in request.messages):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Нужно хотя бы одно сообщение с role=user",
        )

    lc_messages = chat_messages_to_langchain(request.messages)
    agent = await _instantiate_agent(request.model, react_factory, compound_factory)

    if isinstance(agent, CompoundResearchAgent):
        compound_label = build_compound_run_label(request.model, lc_messages)

        if request.stream and request.stream_progress:

            async def sse_with_progress() -> AsyncIterator[str]:
                queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                task = asyncio.create_task(
                    agent.ainvoke_compound(lc_messages, progress_queue=queue, run_label=compound_label),
                )

                async for item in iter_compound_progress_queue(queue, task):
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

                state = await task
                chunk = compound_state_to_chunk(state)
                artifacts = compound_artifacts_from_state(state) if request.include_research_artifacts else None
                payload = build_chat_completion_payload(
                    model=request.model,
                    chunk=chunk,
                    research_artifacts=artifacts,
                )
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_with_progress(), media_type="text/event-stream")

        state = await agent.ainvoke_compound(
            lc_messages,
            progress_queue=None,
            run_label=compound_label,
        )
        chunk = compound_state_to_chunk(state)
        artifacts = compound_artifacts_from_state(state) if request.include_research_artifacts else None

    else:
        chunk = await agent.complete(lc_messages)
        artifacts = None

    payload = build_chat_completion_payload(
        model=request.model,
        chunk=chunk,
        research_artifacts=artifacts,
    )

    if request.stream:
        return StreamingResponse(_sse_final_only(payload), media_type="text/event-stream")

    return JSONResponse(content=payload)


async def _instantiate_agent(
    model: str,
    react_factory: Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
    compound_factory: Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
) -> BaseResearchAgent:
    # Зависимости Agent-фабрик тянут async Resource (langfuse) — sync-вызов фабрики отдаёт Future.
    match model:
        case AssistantType.COMPOUND:
            raw = compound_factory()

        case _:
            raw = react_factory()

    if inspect.isawaitable(raw):
        return await raw

    return raw


async def _sse_final_only(payload: dict[str, Any]) -> AsyncIterator[str]:
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"
