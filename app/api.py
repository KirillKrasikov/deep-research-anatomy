import inspect
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import fastapi
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
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
    supervisor_factory: Annotated[
        Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
        Depends(Provide[Container.supervisor_researcher.provider]),
    ],
) -> JSONResponse:
    if not any(m.role == "user" for m in request.messages):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Нужно хотя бы одно сообщение с role=user",
        )

    lc_messages = chat_messages_to_langchain(request.messages)
    agent = await _instantiate_agent(request.model, react_factory, compound_factory, supervisor_factory)

    if isinstance(agent, CompoundResearchAgent):
        compound_label = build_compound_run_label(request.model, lc_messages)
        state = await agent.ainvoke_compound(lc_messages, run_label=compound_label)
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

    return JSONResponse(content=payload)


async def _instantiate_agent(
    model: str,
    react_factory: Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
    compound_factory: Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
    supervisor_factory: Callable[[], BaseResearchAgent | Awaitable[BaseResearchAgent]],
) -> BaseResearchAgent:
    # Зависимости Agent-фабрик тянут async Resource (langfuse) — sync-вызов фабрики отдаёт Future.
    match model:
        case AssistantType.COMPOUND:
            raw = compound_factory()

        case AssistantType.SUPERVISOR_RESEARCHER:
            raw = supervisor_factory()

        case _:
            raw = react_factory()

    if inspect.isawaitable(raw):
        return await raw

    return raw
