"""
Base class for image generation services.
Provides interface for all image generation implementations.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
from loguru import logger


class ImageGenerationBase(ABC):
    """Abstract base class for image generation services."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize image generation service.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the image generation model."""
        pass

    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image from text prompt.

        Args:
            prompt: Text description for image generation
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated image information:
            {
                "image_url": str,
                "image_data": bytes,
                "metadata": dict
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
                "service": f"image_generation_{self.model_name}",
                "status": "healthy" if is_available else "unhealthy",
                "initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return {
                "service": f"image_generation_{self.model_name}",
                "status": "unhealthy",
                "error": str(e)
            }
