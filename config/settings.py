"""
Configuration settings for SONA AI Assistant.
Handles environment variables and application configuration.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings configuration."""

    # Application Configuration
    app_name: str = Field(default="SONA-AI-Assistant", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Server Configuration
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")

    # AI Service API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    serp_api_key: Optional[str] = Field(default=None, env="SERP_API_KEY")

    # AI Model Configuration
    speech_to_text_model: str = Field(default="whisper", env="SPEECH_TO_TEXT_MODEL")
    intent_detection_model: str = Field(default="openai", env="INTENT_DETECTION_MODEL")
    image_generation_model: str = Field(default="gemini", env="IMAGE_GENERATION_MODEL")
    web_search_model: str = Field(default="serp", env="WEB_SEARCH_MODEL")

    # Whisper Configuration
    whisper_model_size: str = Field(default="base", env="WHISPER_MODEL_SIZE")

    # OpenAI Configuration
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=150, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")

    # Gemini Configuration
    gemini_model: str = Field(default="gemini-pro-vision", env="GEMINI_MODEL")

    # Audio Configuration
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    audio_channels: int = Field(default=1, env="AUDIO_CHANNELS")
    audio_format: str = Field(default="wav", env="AUDIO_FORMAT")

    # File Upload Configuration
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_audio_formats: str = Field(default="wav,mp3,m4a,flac", env="ALLOWED_AUDIO_FORMATS")

    # Cache Configuration
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour

    # Fix for pydantic protected namespace warning
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ()  # This fixes the model_ warning
    }

    def get_allowed_audio_formats(self) -> list[str]:
        """Get list of allowed audio formats."""
        return [fmt.strip() for fmt in self.allowed_audio_formats.split(",")]

    def validate_api_keys(self) -> dict[str, bool]:
        """Validate required API keys are present."""
        validation_result = {
            "openai": bool(self.openai_api_key),
            "gemini": bool(self.gemini_api_key),
            "serp": bool(self.serp_api_key),
        }
        return validation_result

    def get_backend_url(self) -> str:
        """Get backend URL."""
        return f"http://{self.backend_host}:{self.backend_port}"

    def get_streamlit_url(self) -> str:
        """Get Streamlit URL."""
        return f"http://{self.backend_host}:{self.streamlit_port}"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
