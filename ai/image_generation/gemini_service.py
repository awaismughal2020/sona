"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Google Gemini image generation service implementation with smaller 400x400px images.
Handles image generation using Gemini AI models with proper image descriptions.
"""
import google.generativeai as genai
from typing import Dict, Any, Optional
import asyncio
import base64
import io
from loguru import logger

from .base import ImageGenerationBase
from config.settings import get_settings
from utils.constants import ERROR_MESSAGES, SUCCESS_MESSAGES


class GeminiImageService(ImageGenerationBase):
    """Real Gemini 2.0 Flash image generation service."""

    def __init__(self, **kwargs):
        """Initialize Gemini image generation service."""
        settings = get_settings()
        super().__init__(
            model_name="gemini",
            api_key=settings.gemini_api_key,
            model="gemini-2.0-flash-exp",  # Model with image generation
            **kwargs
        )
        self.client = None
        self.model_instance = None

    async def initialize(self) -> None:
        """Initialize Gemini client and model."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("Gemini API key is required")

            logger.info("Initializing Gemini 2.0 Flash image generation service")

            # Configure Gemini API
            genai.configure(api_key=self.config['api_key'])

            # Initialize the model for image generation
            # Use experimental model that supports image generation
            self.model_instance = genai.GenerativeModel("gemini-2.0-flash-exp")

            # Test the connection
            await self._test_connection()

            self.is_initialized = True
            logger.info("Gemini 2.0 Flash image generation service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            raise RuntimeError(f"Gemini initialization failed: {e}")

    async def _test_connection(self) -> None:
        """Test Gemini API connection."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model_instance.generate_content("Hello, test connection")
            )
            logger.info("Gemini API connection test successful")
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            raise

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate actual image using Gemini 2.0 Flash.

        Args:
            prompt: Text description for image generation
            **kwargs: Additional parameters (style, size, etc.)

        Returns:
            Dictionary containing generated image information
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Generating real image with Gemini 2.0 Flash: {prompt[:100]}...")

            # Create image generation prompt
            # Gemini 2.0 Flash supports multimodal output including images
            generation_prompt = f"Generate an image: {prompt}"

            # Optional: Add style or quality instructions
            style = kwargs.get('style', 'realistic')
            if style and style != 'realistic':
                generation_prompt = f"Generate an image in {style} style: {prompt}"

            # Generate content with image output
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model_instance.generate_content(
                    generation_prompt,
                    generation_config=genai.GenerationConfig(
                        # Configure for image generation
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
            )

            # Process response - Gemini 2.0 Flash returns images in the response
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    # Check if this part contains image data
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Extract image data
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type

                        return {
                            "success": True,
                            "image_url": None,
                            "image_data": image_data,  # Already base64 encoded
                            "mime_type": mime_type,
                            "metadata": {
                                "original_prompt": prompt,
                                "model": "gemini-2.0-flash-exp",
                                "generated_at": asyncio.get_event_loop().time(),
                                "style": style,
                                "type": "real_image_generation"
                            }
                        }

            # If no image found in response, fallback to descriptive response
            logger.warning("No image generated by Gemini 2.0 Flash, creating descriptive response")
            return await self._create_descriptive_fallback(prompt, response.text if hasattr(response, 'text') else "")

        except Exception as e:
            logger.error(f"Gemini image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None,
                "image_data": None,
                "metadata": {
                    "prompt": prompt,
                    "error": str(e),
                    "model": "gemini-2.0-flash-exp"
                }
            }

    async def _create_descriptive_fallback(self, prompt: str, description: str) -> Dict[str, Any]:
        """Create a descriptive card when actual image generation fails."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import textwrap

            # Create a larger canvas for better quality
            width, height = 800, 600
            image = Image.new('RGB', (width, height), color='#f0f8ff')
            draw = ImageDraw.Draw(image)

            # Try to load fonts
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                desc_font = ImageFont.truetype("arial.ttf", 16)
                footer_font = ImageFont.truetype("arial.ttf", 12)
            except:
                title_font = ImageFont.load_default()
                desc_font = ImageFont.load_default()
                footer_font = ImageFont.load_default()

            # Add gradient background
            for i in range(height):
                alpha = int(255 * (1 - i / height) * 0.1)
                color = (240 - alpha//4, 248 - alpha//4, 255 - alpha//4)
                draw.line([(0, i), (width, i)], fill=color)

            # Title
            title = "🎨 AI Image Description"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, 30), title, fill='#1e3a8a', font=title_font)

            # Prompt
            prompt_title = "Your Request:"
            draw.text((40, 80), prompt_title, fill='#374151', font=desc_font)

            wrapped_prompt = textwrap.fill(f'"{prompt}"', width=80)
            prompt_y = 110
            for line in wrapped_prompt.split('\n')[:3]:  # Max 3 lines
                draw.text((50, prompt_y), line, fill='#1f2937', font=desc_font)
                prompt_y += 25

            # AI Description
            if description:
                desc_title = "AI Generated Description:"
                desc_y = prompt_y + 30
                draw.text((40, desc_y), desc_title, fill='#374151', font=desc_font)

                wrapped_desc = textwrap.fill(description, width=80)
                desc_y += 30
                for line in wrapped_desc.split('\n')[:15]:  # Max 15 lines
                    if desc_y > height - 100:
                        draw.text((50, desc_y), "...", fill='#6b7280', font=desc_font)
                        break
                    draw.text((50, desc_y), line, fill='#374151', font=desc_font)
                    desc_y += 20

            # Footer
            footer_text = "✨ Powered by Gemini 2.0 Flash • SONA AI"
            footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
            footer_width = footer_bbox[2] - footer_bbox[0]
            footer_x = (width - footer_width) // 2
            draw.text((footer_x, height - 40), footer_text, fill='#9ca3af', font=footer_font)

            # Border
            draw.rectangle([5, 5, width-6, height-6], outline='#3b82f6', width=2)

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', quality=95, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "success": True,
                "image_url": None,
                "image_data": image_base64,
                "mime_type": "image/png",
                "metadata": {
                    "original_prompt": prompt,
                    "model": "gemini-2.0-flash-exp",
                    "generated_at": asyncio.get_event_loop().time(),
                    "type": "descriptive_fallback",
                    "description": description
                }
            }

        except Exception as e:
            logger.error(f"Failed to create descriptive fallback: {e}")
            return {
                "success": False,
                "error": f"Image generation and fallback both failed: {str(e)}",
                "image_url": None,
                "image_data": None,
                "metadata": {"prompt": prompt, "error": str(e)}
            }

    def is_available(self) -> bool:
        """Check if Gemini service is available."""
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"Gemini availability check failed: {e}")
            return False

    async def get_model_info(self) -> dict:
        """Get information about the Gemini model."""
        return {
            "service_name": self.model_name,
            "model": "gemini-2.0-flash-exp",
            "capabilities": ["real_image_generation", "descriptive_fallback", "style_variations"],
            "supported_formats": ["PNG"],
            "model_type": "multimodal_with_image_output",
            "initialized": self.is_initialized,
            "available": self.is_available(),
            "description": "Gemini 2.0 Flash with native image generation capabilities"
        }

