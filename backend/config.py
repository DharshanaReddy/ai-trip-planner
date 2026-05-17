from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # LangSmith observability
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "ai-trip-planner"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Redis cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
