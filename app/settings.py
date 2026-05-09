from functools import cache

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "deep-research-anatomy"

    langfuse_base_url: HttpUrl = HttpUrl("http://localhost:3000")
    langfuse_public_key: str
    langfuse_secret_key: SecretStr
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_label: str = "latest"

    anthropic_base_url: HttpUrl = HttpUrl("https://api.anthropic.com")
    anthropic_api_key: SecretStr
    anthropic_model_fast: str = "claude-haiku-4-5-20251001"
    anthropic_model_balanced: str = "claude-sonnet-4-6"
    anthropic_model_sota: str = "claude-opus-4-7"
    anthropic_max_tokens: int = 32_000
    anthropic_effort: str = "high"


@cache
def get_settings() -> Settings:
    return Settings()
