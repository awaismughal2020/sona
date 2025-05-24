"""
Base class for speech-to-text services.
Provides interface for all speech-to-text implementations.
"""
from abc import ABC, abstractmethod
from typing import Union, Optional
import asyncio
from loguru import logger

class SpeechToTextBase(ABC):
    """Abstract base class for speech-to-text services."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize speech-to-text service.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the speech-to-text model."""
        pass

    @abstractmethod
    async def transcribe_audio(self, audio_data: Union[bytes, str]) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data as bytes or file path

        Returns:
            Transcribed text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available."""
        pass

    async def health_check(self) -> dict:
        """Perform health check on the service."""
        try:
            is_available = self.is_available()
            return {
                "service": f"speech_to_text_{self.model_name}",
                "status": "healthy" if is_available else "unhealthy",
                "initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return {
                "service": f"speech_to_text_{self.model_name}",
                "status": "unhealthy",
                "error": str(e)
            }

