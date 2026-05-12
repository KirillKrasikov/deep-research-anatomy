from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import anthropic._base_client as anthropic_base_client
from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.ext.starlette import Lifespan
from dependency_injector.providers import Factory, Resource, Self, Singleton
from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.agents.compound_researcher import CompoundResearchAgent
from app.agents.react_researcher import ReactResearchAgent
from app.settings import Settings, get_settings


def _apply_anthropic_retry_backoff(settings: Settings) -> None:
    mod = cast(Any, anthropic_base_client)
    mod.INITIAL_RETRY_DELAY = settings.anthropic_retry_initial_delay_seconds
    mod.MAX_RETRY_DELAY = settings.anthropic_retry_max_delay_seconds


@asynccontextmanager
async def langfuse_client_manager(settings: Settings) -> AsyncGenerator[Langfuse, None]:
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url.encoded_string(),
        sample_rate=settings.langfuse_sample_rate,
        timeout=settings.langfuse_timeout_seconds,
    )
    try:
        yield client
    finally:
        client.shutdown()


def langfuse_callback_handler_manager(langfuse: Langfuse, settings: Settings) -> CallbackHandler:  # noqa: ARG001
    return CallbackHandler(public_key=settings.langfuse_public_key)


def _get_chat_anthropic(settings: Settings, model: str) -> ChatAnthropic:
    _apply_anthropic_retry_backoff(settings)

    return ChatAnthropic(
        model=model,
        temperature=0.0,
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url.encoded_string(),
        max_tokens=settings.anthropic_max_tokens,
        effort=settings.anthropic_effort,
        max_retries=settings.anthropic_max_retries,
    )


def llm_fast_manager(settings: Settings) -> ChatAnthropic:
    return _get_chat_anthropic(settings, settings.anthropic_model_fast)


def llm_balanced_manager(settings: Settings) -> ChatAnthropic:
    return _get_chat_anthropic(settings, settings.anthropic_model_balanced)


def llm_sota_manager(settings: Settings) -> ChatAnthropic:
    return _get_chat_anthropic(settings, settings.anthropic_model_sota)


class Container(DeclarativeContainer):
    wiring_config = WiringConfiguration(packages=["app"], auto_wire=True)

    __self__ = Self()
    settings = Singleton(provides=get_settings)

    langfuse_client = Resource(provides=langfuse_client_manager, settings=settings)
    langfuse_callback_handler = Resource(
        provides=langfuse_callback_handler_manager,
        langfuse=langfuse_client,
        settings=settings,
    )
    llm_fast = Factory(provides=llm_fast_manager, settings=settings)
    llm_balanced = Factory(provides=llm_balanced_manager, settings=settings)
    llm_sota = Factory(provides=llm_sota_manager, settings=settings)

    react_researcher = Factory(
        provides=ReactResearchAgent,
        llm=llm_balanced,
        langfuse_callback_handler=langfuse_callback_handler,
    )
    compound_researcher = Factory(
        provides=CompoundResearchAgent,
        llm=llm_balanced,
        langfuse_callback_handler=langfuse_callback_handler,
    )

    lifespan = Singleton(provides=Lifespan, container=__self__)
    app = Singleton(provides=FastAPI, lifespan=lifespan)
