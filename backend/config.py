from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq (free — get key at console.groq.com)
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"

    # LangSmith observability
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "ai-trip-planner"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # OpenAI (optional — only needed for embeddings if not using local)
    openai_api_key: str = ""

    # Redis cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
