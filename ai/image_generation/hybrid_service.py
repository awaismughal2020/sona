"""
Hybrid image generation service that uses Gemini for prompt enhancement
and OpenAI DALL-E for actual image generation.
"""
import google.generativeai as genai
import openai
from typing import Dict, Any, Optional
import asyncio
import base64
import io
import requests
from loguru import logger

from .base import ImageGenerationBase
from config.settings import get_settings
from utils.constants import ERROR_MESSAGES, SUCCESS_MESSAGES


class HybridImageService(ImageGenerationBase):
    """Hybrid image service using Gemini + DALL-E."""

    def __init__(self, **kwargs):
        """Initialize hybrid image service."""
        settings = get_settings()
        super().__init__(
            model_name="hybrid_gemini_dalle",
            gemini_api_key=settings.gemini_api_key,
            openai_api_key=settings.openai_api_key,
            **kwargs
        )
        self.gemini_model = None
        self.openai_client = None

    async def initialize(self) -> None:
        """Initialize both Gemini and OpenAI clients."""
        try:
            # Initialize Gemini for prompt enhancement
            if self.config.get('gemini_api_key'):
                genai.configure(api_key=self.config['gemini_api_key'])
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                logger.info("Gemini client initialized for prompt enhancement")

            # Initialize OpenAI for image generation
            if self.config.get('openai_api_key'):
                self.openai_client = openai.AsyncOpenAI(api_key=self.config['openai_api_key'])
                logger.info("OpenAI client initialized for image generation")

            if not self.gemini_model and not self.openai_client:
                raise ValueError("At least one API key (Gemini or OpenAI) is required")

            self.is_initialized = True
            logger.info("Hybrid image service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hybrid service: {e}")
            raise RuntimeError(f"Hybrid service initialization failed: {e}")

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using Gemini enhancement + DALL-E generation."""
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Generating image with hybrid approach: {prompt[:100]}...")

            # Step 1: Enhance prompt with Gemini (if available)
            enhanced_prompt = prompt
            if self.gemini_model:
                enhanced_prompt = await self._enhance_prompt_with_gemini(prompt)

            # Step 2: Generate image with DALL-E (if available)
            if self.openai_client:
                return await self._generate_with_dalle(enhanced_prompt, prompt, **kwargs)
            else:
                # Fallback to descriptive response
                return await self._create_descriptive_response(enhanced_prompt, prompt)

        except Exception as e:
            logger.error(f"Hybrid image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None,
                "image_data": None,
                "metadata": {
                    "prompt": prompt,
                    "error": str(e)
                }
            }

    async def _enhance_prompt_with_gemini(self, prompt: str) -> str:
        """Enhance prompt using Gemini 2.0."""
        try:
            enhancement_request = f"""
            Enhance this image generation prompt to be more detailed and specific for DALL-E:

            Original prompt: "{prompt}"

            Please provide a single, detailed prompt (max 400 characters) that includes:
            - Visual details and composition
            - Artistic style if not specified
            - Quality indicators

            Return only the enhanced prompt, nothing else.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(enhancement_request)
            )

            enhanced = response.text.strip()
            logger.info(f"Prompt enhanced by Gemini: {enhanced[:100]}...")
            return enhanced

        except Exception as e:
            logger.error(f"Failed to enhance prompt: {e}")
            return prompt  # Return original on failure

    async def _generate_with_dalle(self, enhanced_prompt: str, original_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image using DALL-E."""
        try:
            # Use DALL-E 3 for best quality
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size=kwargs.get("size", "1024x1024"),
                quality=kwargs.get("quality", "standard"),
                n=1
            )

            image_url = response.data[0].url

            # Download the image
            image_response = requests.get(image_url)
            image_data = base64.b64encode(image_response.content).decode()

            return {
                "success": True,
                "image_url": image_url,
                "image_data": image_data,
                "mime_type": "image/png",
                "metadata": {
                    "original_prompt": original_prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "model": "dall-e-3",
                    "enhanced_by": "gemini-2.0-flash-exp",
                    "size": kwargs.get("size", "1024x1024"),
                    "quality": kwargs.get("quality", "standard"),
                    "generated_at": asyncio.get_event_loop().time()
                }
            }

        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            raise

    async def _create_descriptive_response(self, enhanced_description: str, original_prompt: str) -> Dict[str, Any]:
        """Fallback descriptive response."""
        return {
            "success": False,
            "error": "Image generation requires OpenAI API key for DALL-E",
            "image_url": None,
            "image_data": None,
            "enhanced_description": enhanced_description,
            "metadata": {
                "original_prompt": original_prompt,
                "enhanced_prompt": enhanced_description,
                "note": "Enhanced by Gemini but image generation requires DALL-E API access"
            }
        }

    def is_available(self) -> bool:
        """Check if hybrid service is available."""
        return self.is_initialized and (bool(self.gemini_model) or bool(self.openai_client))
