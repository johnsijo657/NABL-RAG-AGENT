from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Database
    APP_DATABASE_URL: str
    
    # JWT Auth
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    
    # LLM Config
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:7b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    
    CLOUD_API_KEY: Optional[str] = None
    CLOUD_CHAT_MODEL: Optional[str] = None
    
    # RAG Config
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    RERANK_TOP_N: int = 3
    MAX_CHAT_MESSAGES: int = 10
    
    CHAINLIT_AUTH_SECRET: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
