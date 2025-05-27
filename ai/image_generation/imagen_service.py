"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Google Imagen 3 image generation service implementation.
Generates images using Google's dedicated Imagen 3 API.
"""
import google.generativeai as genai
from typing import Dict, Any, Optional
import asyncio
import base64
from loguru import logger

from .base import ImageGenerationBase
from config.settings import get_settings


class ImagenService(ImageGenerationBase):
    """Google Imagen 3 image generation service."""

    def __init__(self, **kwargs):
        """Initialize Imagen service."""
        settings = get_settings()
        super().__init__(
            model_name="imagen",
            api_key=settings.gemini_api_key,  # Uses same API key as Gemini
            model="imagen-3.0-generate-001",
            **kwargs
        )
        self.model_instance = None

    async def initialize(self) -> None:
        """Initialize Imagen client."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("Google API key is required for Imagen")

            logger.info("Initializing Google Imagen 3 service")

            # Configure API
            genai.configure(api_key=self.config['api_key'])

            # Initialize model - Imagen uses a different endpoint
            self.model_instance = genai.ImageGenerationModel("imagen-3.0-generate-001")

            self.is_initialized = True
            logger.info("Imagen 3 service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Imagen service: {e}")
            raise RuntimeError(f"Imagen initialization failed: {e}")

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using Imagen 3.

        Args:
            prompt: Text description for image generation
            **kwargs: Additional parameters

        Returns:
            Dictionary containing generated image information
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Generating image with Imagen 3: {prompt[:100]}...")

            # Imagen 3 parameters
            aspect_ratio = kwargs.get('aspect_ratio', '1:1')  # 1:1, 3:4, 4:3, 9:16, 16:9
            number_of_images = kwargs.get('number_of_images', 1)

            # Generate image using Imagen 3
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model_instance.generate_images(
                    prompt=prompt,
                    number_of_images=number_of_images,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_only_high",
                    include_rai_reason=True
                )
            )

            if not response or not response.images:
                raise Exception("No image data returned from Imagen 3")

            # Get the first image
            image = response.images[0]

            # Convert PIL Image to base64
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "success": True,
                "image_url": None,
                "image_data": image_base64,
                "mime_type": "image/png",
                "metadata": {
                    "original_prompt": prompt,
                    "model": "imagen-3.0-generate-001",
                    "aspect_ratio": aspect_ratio,
                    "generated_at": asyncio.get_event_loop().time(),
                    "type": "imagen3_image_generation"
                }
            }

        except Exception as e:
            logger.error(f"Imagen 3 image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None,
                "image_data": None,
                "metadata": {
                    "prompt": prompt,
                    "error": str(e),
                    "model": "imagen-3.0-generate-001"
                }
            }

    def is_available(self) -> bool:
        """Check if Imagen service is available."""
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"Imagen availability check failed: {e}")
            return False

    async def get_model_info(self) -> dict:
        """Get information about the Imagen model."""
        return {
            "service_name": self.model_name,
            "model": "imagen-3.0-generate-001",
            "capabilities": ["high_quality_image_generation", "aspect_ratio_control", "safety_filtering"],
            "supported_formats": ["PNG"],
            "supported_aspect_ratios": ["1:1", "3:4", "4:3", "9:16", "16:9"],
            "initialized": self.is_initialized,
            "available": self.is_available(),
            "description": "Google Imagen 3 dedicated image generation service"
        }


