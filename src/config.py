"""
Configuration settings for SupplyChainGPT
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Application Settings
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        alias="CHROMA_PERSIST_DIR"
    )
    docs_dir: Path = Field(
        default=Path("./data/documents"),
        alias="DOCS_DIR"
    )

    # Model Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        alias="LLM_MODEL"
    )
    llm_temperature: float = Field(
        default=0.1,
        alias="LLM_TEMPERATURE"
    )

    # Retrieval Settings
    top_k_retrieval: int = Field(default=8, alias="TOP_K_RETRIEVAL")
    chunk_size: int = Field(default=750, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    # Document Types
    supported_extensions: set = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md"}

    # Document type enum values
    doc_types: list = [
        "policy", "sop", "contract", "report",
        "meeting_notes", "email", "manual", "export"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "data" / "forecasts").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "data" / "sample_data").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
