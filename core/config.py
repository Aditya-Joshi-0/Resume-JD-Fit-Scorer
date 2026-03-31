import os
from dotenv import load_dotenv
from functools import lru_cache
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    llm_provider: str
    groq_api_key: str
    nvidia_api_key: str
    llm_model: str
    embedding_model: str
    debug_mode: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", ""),
            debug_mode=os.getenv("DEBUG_MODE", False)
        )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
