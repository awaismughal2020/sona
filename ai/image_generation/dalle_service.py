"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

OpenAI DALL-E 3 image generation service implementation.
Generates high-quality images using OpenAI's DALL-E 3 API.
"""
import openai
from typing import Dict, Any, Optional
import asyncio
import base64
import requests
from loguru import logger

from .base import ImageGenerationBase
from config.settings import get_settings


class DalleImageService(ImageGenerationBase):
    """OpenAI DALL-E 3 image generation service."""

    def __init__(self, **kwargs):
        """Initialize DALL-E service."""
        settings = get_settings()
        super().__init__(
            model_name="dalle",
            api_key=settings.openai_api_key,
            model="dall-e-3",
            **kwargs
        )
        self.client = None

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("OpenAI API key is required for DALL-E")

            logger.info("Initializing OpenAI DALL-E 3 service")

            # Initialize OpenAI client
            self.client = openai.AsyncOpenAI(api_key=self.config['api_key'])

            # Test connection
            await self._test_connection()

            self.is_initialized = True
            logger.info("DALL-E 3 service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DALL-E service: {e}")
            raise RuntimeError(f"DALL-E initialization failed: {e}")

    async def _test_connection(self) -> None:
        """Test OpenAI API connection."""
        try:
            # Test with a simple request
            models = await self.client.models.list()
            logger.info("OpenAI API connection test successful")
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using DALL-E 3.

        Args:
            prompt: Text description for image generation
            **kwargs: Additional parameters (size, quality, style)

        Returns:
            Dictionary containing generated image information
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Generating image with DALL-E 3: {prompt[:100]}...")

            # DALL-E 3 parameters
            size = kwargs.get('size', '1024x1024')  # 1024x1024, 1792x1024, or 1024x1792
            quality = kwargs.get('quality', 'standard')  # standard or hd
            style = kwargs.get('style', 'vivid')  # vivid or natural

            # Generate image
            response = await self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1,  # DALL-E 3 only supports n=1
                response_format="url"  # Get URL first, then download
            )

            if not response.data:
                raise Exception("No image data returned from DALL-E 3")

            image_data = response.data[0]
            image_url = image_data.url

            # Download the image and convert to base64
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()

            image_base64 = base64.b64encode(image_response.content).decode()

            return {
                "success": True,
                "image_url": image_url,
                "image_data": image_base64,
                "mime_type": "image/png",
                "metadata": {
                    "original_prompt": prompt,
                    "revised_prompt": image_data.revised_prompt if hasattr(image_data, 'revised_prompt') else prompt,
                    "model": "dall-e-3",
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "generated_at": asyncio.get_event_loop().time(),
                    "type": "dalle3_image_generation"
                }
            }

        except Exception as e:
            logger.error(f"DALL-E 3 image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None,
                "image_data": None,
                "metadata": {
                    "prompt": prompt,
                    "error": str(e),
                    "model": "dall-e-3"
                }
            }

    def is_available(self) -> bool:
        """Check if DALL-E service is available."""
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"DALL-E availability check failed: {e}")
            return False

    async def get_model_info(self) -> dict:
        """Get information about the DALL-E model."""
        return {
            "service_name": self.model_name,
            "model": "dall-e-3",
            "capabilities": ["high_quality_image_generation", "prompt_revision", "style_control"],
            "supported_formats": ["PNG"],
            "supported_sizes": ["1024x1024", "1792x1024", "1024x1792"],
            "quality_options": ["standard", "hd"],
            "style_options": ["vivid", "natural"],
            "initialized": self.is_initialized,
            "available": self.is_available(),
            "description": "OpenAI DALL-E 3 high-quality image generation"
        }
