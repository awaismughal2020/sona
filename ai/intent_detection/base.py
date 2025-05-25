"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Base class for intent detection services.
Provides interface for all intent detection implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from loguru import logger
from utils.constants import IntentType

class IntentDetectionBase(ABC):
    """Abstract base class for intent detection services."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize intent detection service.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the intent detection model."""
        pass

    @abstractmethod
    async def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        Detect intent from text input.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing intent information:
            {
                "intent": IntentType,
                "confidence": float,
                "entities": dict,
                "response": str
            }
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
                "service": f"intent_detection_{self.model_name}",
                "status": "healthy" if is_available else "unhealthy",
                "initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return {
                "service": f"intent_detection_{self.model_name}",
                "status": "unhealthy",
                "error": str(e)
            }
