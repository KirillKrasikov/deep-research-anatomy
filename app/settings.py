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
    langfuse_timeout_seconds: int = Field(default=30, ge=1, le=300)

    anthropic_base_url: HttpUrl = HttpUrl("https://api.anthropic.com")
    anthropic_api_key: SecretStr
    anthropic_model_fast: str = "claude-haiku-4-5-20251001"
    anthropic_model_balanced: str = "claude-sonnet-4-6"
    anthropic_model_sota: str = "claude-opus-4-7"
    anthropic_max_tokens: int = 32_000
    anthropic_effort: str = "high"
    anthropic_max_retries: int = Field(default=3, ge=0, le=15)
    anthropic_retry_initial_delay_seconds: float = Field(default=1.0, ge=0.1)
    anthropic_retry_max_delay_seconds: float = Field(default=16.0, ge=0.5)


@cache
def get_settings() -> Settings:
    return Settings()
