"""
Google Gemini image generation service implementation.
Handles image generation using Gemini AI models with proper image descriptions.
"""
import google.generativeai as genai
from typing import Dict, Any, Optional
import asyncio
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import textwrap
from loguru import logger

from .base import ImageGenerationBase
from config.settings import get_settings
from utils.constants import ERROR_MESSAGES, SUCCESS_MESSAGES


class GeminiImageService(ImageGenerationBase):
    """Google Gemini image generation service."""

    def __init__(self, **kwargs):
        """Initialize Gemini image generation service."""
        settings = get_settings()
        super().__init__(
            model_name="gemini",
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            **kwargs
        )
        self.client = None
        self.model_instance = None

    async def initialize(self) -> None:
        """Initialize Gemini client and model."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("Gemini API key is required")

            logger.info("Initializing Gemini image generation service")

            # Configure Gemini API
            genai.configure(api_key=self.config['api_key'])

            # Initialize the model for image generation
            model_name = "gemini-2.0-flash-exp"  # Gemini 2.0 model
            self.model_instance = genai.GenerativeModel(model_name)

            # Test the connection
            await self._test_connection()

            self.is_initialized = True
            logger.info("Gemini image generation service initialized successfully")

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
        Generate image using Gemini AI with enhanced descriptions and visual representation.

        Args:
            prompt: Text description for image generation
            **kwargs: Additional parameters (size, style, etc.)

        Returns:
            Dictionary containing generated image information
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Generating image representation for: {prompt[:100]}...")

            # Step 1: Use Gemini to create detailed image description
            detailed_description = await self._create_detailed_description(prompt)

            # Step 2: Create visual representation with the description
            image_data = await self._create_visual_representation(prompt, detailed_description)

            # Step 3: Return successful result
            result = {
                "success": True,
                "image_url": None,  # We're creating data, not URL
                "image_data": image_data,
                "mime_type": "image/png",
                "metadata": {
                    "original_prompt": prompt,
                    "detailed_description": detailed_description,
                    "model": "gemini-2.0-flash-exp",
                    "generated_at": asyncio.get_event_loop().time(),
                    "type": "gemini_enhanced_description_with_visual"
                }
            }

            logger.info("Image representation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
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

    async def _create_detailed_description(self, prompt: str) -> str:
        """
        Use Gemini 2.0 to create a detailed description of the requested image.

        Args:
            prompt: Original user prompt

        Returns:
            Detailed image description
        """
        try:
            enhancement_prompt = f"""
            You are an expert image description generator. The user has requested an image with this description: "{prompt}"

            Please provide a vivid, detailed description of what this image would look like, including:

            1. **Visual Elements**: What objects, people, or scenes are present
            2. **Setting & Environment**: Where does this take place, what's the background
            3. **Colors & Lighting**: What colors dominate, what's the lighting like
            4. **Composition & Style**: How are elements arranged, what's the artistic style
            5. **Mood & Atmosphere**: What feeling or emotion does the image convey
            6. **Specific Details**: Any unique or interesting details that make it special

            Write this as a single, flowing description that paints a clear picture in the reader's mind. Make it engaging and vivid, as if you're describing a real photograph or artwork.

            Keep the description focused and around 200-300 words.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model_instance.generate_content(enhancement_prompt)
            )

            detailed_description = response.text.strip()
            logger.info("Detailed description created by Gemini")
            return detailed_description

        except Exception as e:
            logger.error(f"Failed to create detailed description: {e}")
            return f"A visual representation of: {prompt}"

    async def _create_visual_representation(self, original_prompt: str, detailed_description: str) -> str:
        """
        Create a visual representation with the prompt and description.

        Args:
            original_prompt: Original user prompt
            detailed_description: Gemini's detailed description

        Returns:
            Base64 encoded image data
        """
        try:
            # Create a larger, more attractive image
            width, height = 800, 600

            # Create gradient background
            image = Image.new('RGB', (width, height), color='#f0f4f8')
            draw = ImageDraw.Draw(image)

            # Create a subtle gradient effect
            for i in range(height):
                alpha = int(255 * (1 - i / height) * 0.1)
                color = (240 - alpha // 4, 244 - alpha // 4, 248 - alpha // 4)
                draw.line([(0, i), (width, i)], fill=color)

            # Try to load a better font
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                desc_font = ImageFont.truetype("arial.ttf", 16)
                footer_font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    title_font = ImageFont.load_default()
                    desc_font = ImageFont.load_default()
                    footer_font = ImageFont.load_default()
                except:
                    # Create a more basic version if fonts fail
                    return await self._create_simple_placeholder(original_prompt)

            # Title
            title = "🎨 Image Description by SONA"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, 30), title, fill='#2c3e50', font=title_font)

            # Original prompt box
            prompt_title = "Your Request:"
            draw.text((40, 80), prompt_title, fill='#34495e', font=desc_font)

            # Wrap the original prompt
            wrapped_prompt = textwrap.fill(f'"{original_prompt}"', width=80)
            prompt_y = 110
            for line in wrapped_prompt.split('\n'):
                draw.text((60, prompt_y), line, fill='#2980b9', font=desc_font)
                prompt_y += 25

            # Detailed description
            desc_title = "Detailed Visual Description:"
            desc_y = prompt_y + 20
            draw.text((40, desc_y), desc_title, fill='#34495e', font=desc_font)

            # Wrap and display the detailed description
            wrapped_desc = textwrap.fill(detailed_description, width=85)
            desc_y += 35

            for line in wrapped_desc.split('\n'):
                if desc_y > height - 100:  # Leave space for footer
                    draw.text((60, desc_y), "...", fill='#7f8c8d', font=desc_font)
                    break
                draw.text((60, desc_y), line, fill='#2c3e50', font=desc_font)
                desc_y += 22

            # Footer
            footer_text = "✨ Enhanced by Gemini 2.0 Flash • SONA AI Assistant"
            footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
            footer_width = footer_bbox[2] - footer_bbox[0]
            footer_x = (width - footer_width) // 2
            draw.text((footer_x, height - 40), footer_text, fill='#95a5a6', font=footer_font)

            # Add a decorative border
            border_color = '#bdc3c7'
            draw.rectangle([10, 10, width - 10, height - 10], outline=border_color, width=2)

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return image_base64

        except Exception as e:
            logger.error(f"Failed to create visual representation: {e}")
            return await self._create_simple_placeholder(original_prompt)

    async def _create_simple_placeholder(self, prompt: str) -> str:
        """Create a simple placeholder if the main visual creation fails."""
        try:
            width, height = 600, 400
            image = Image.new('RGB', (width, height), color='lightblue')
            draw = ImageDraw.Draw(image)

            # Simple text
            text = f"Image Request:\n{prompt}\n\nProcessed by SONA + Gemini 2.0"
            wrapped_text = textwrap.fill(text, width=40)

            # Center the text
            bbox = draw.textbbox((0, 0), wrapped_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (width - text_width) // 2
            y = (height - text_height) // 2

            draw.text((x, y), wrapped_text, fill='darkblue')

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return image_base64

        except Exception as e:
            logger.error(f"Failed to create simple placeholder: {e}")
            return ""

    def is_available(self) -> bool:
        """
        Check if Gemini service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"Gemini availability check failed: {e}")
            return False

    async def get_supported_styles(self) -> list[str]:
        """
        Get list of supported image styles.

        Returns:
            List of supported styles
        """
        return [
            "realistic",
            "artistic",
            "cartoon",
            "photographic",
            "digital_art",
            "oil_painting",
            "watercolor",
            "sketch",
            "fantasy",
            "sci-fi",
            "abstract",
            "minimalist"
        ]

    async def get_model_info(self) -> dict:
        """
        Get information about the Gemini model.

        Returns:
            Dictionary with model information
        """
        return {
            "service_name": self.model_name,
            "model": "gemini-2.0-flash-exp",
            "capabilities": ["detailed_image_descriptions", "visual_representations", "enhanced_prompts"],
            "supported_formats": ["PNG"],
            "initialized": self.is_initialized,
            "available": self.is_available(),
            "description": "Uses Gemini 2.0 Flash to create detailed image descriptions and visual representations"
        }
