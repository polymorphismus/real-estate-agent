from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")

    OPENAI_MODEL: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI model for routing/planning/response tasks",
    )
    OPENAI_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    OPENAI_MAX_OUTPUT_TOKENS: int = Field(default=800, gt=0)
    OPENAI_MAX_OUTPUT_TOKENS_EXTRACTOR: int = Field(default=320, gt=0)
    OPENAI_MAX_OUTPUT_TOKENS_CODEGEN: int = Field(default=700, gt=0)
    OPENAI_MAX_OUTPUT_TOKENS_ANSWER: int = Field(default=220, gt=0)

    # Request settings
    OPENAI_TIMEOUT_SEC: int = Field(default=60, gt=0)


settings = Settings()
