"""Configuration management for Contract Analyzer."""

import os
from pathlib import Path
from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Azure OpenAI Configuration
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2025-01-01-preview", env="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_name: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Available Models Configuration
    azure_openai_models: str = Field(default="gpt-4.1,gpt-4.1-mini", env="AZURE_OPENAI_MODELS")
    default_graph_model: str = Field(default="gpt-4.1", env="DEFAULT_GRAPH_MODEL")
    default_chat_model: str = Field(default="gpt-4.1", env="DEFAULT_CHAT_MODEL")
    
    # Azure OpenAI Embeddings Configuration
    azure_openai_embedding_api_key: str = Field(..., env="AZURE_OPENAI_EMBEDDING_API_KEY")
    azure_openai_embedding_endpoint: str = Field(..., env="AZURE_OPENAI_EMBEDDING_ENDPOINT")
    azure_openai_embedding_deployment_name: str = Field(..., env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_openai_embedding_api_version: str = Field(default="2023-05-15", env="AZURE_OPENAI_EMBEDDING_API_VERSION")
    
    # Application Configuration
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    gradio_port: int = Field(default=7860, env="GRADIO_PORT")
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    knowledge_stores_dir: str = Field(default="./data/knowledge_stores", env="KNOWLEDGE_STORES_DIR")
    
    # Model Configuration
    embedding_model: str = Field(default="azure-openai")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_tokens: int = Field(default=32000)
    temperature: float = Field(default=0.1)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_available_models(self) -> List[str]:
        """Get list of available Azure OpenAI models."""
        return [model.strip() for model in self.azure_openai_models.split(",")]


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()

# Ensure data directories exist
Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
Path(settings.knowledge_stores_dir).mkdir(parents=True, exist_ok=True)
Path("./data/uploads").mkdir(parents=True, exist_ok=True)
