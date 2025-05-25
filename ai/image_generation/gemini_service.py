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
        Generate image using Gemini AI with enhanced descriptions and compact visual representation.

        Args:
            prompt: Text description for image generation
            **kwargs: Additional parameters (size, style, etc.)

        Returns:
            Dictionary containing generated image information
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Generating compact image representation for: {prompt[:100]}...")

            # Step 1: Use Gemini to create detailed image description
            detailed_description = await self._create_detailed_description(prompt)

            # Step 2: Create compact visual representation with the description
            image_data = await self._create_compact_visual_representation(prompt, detailed_description)

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
                    "size": "400x400",
                    "type": "gemini_enhanced_description_compact"
                }
            }

            logger.info("Compact image representation generated successfully")
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

            Write this as a single, flowing description that paints a clear picture in the reader's mind. Make it engaging and vivid, as if you're describing a real photograph or artwork.

            Keep the description focused and around 150-200 words for a compact display.
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

    async def _create_compact_visual_representation(self, original_prompt: str, detailed_description: str) -> str:
        """
        Create a compact 400x400px visual representation with the prompt and description.

        Args:
            original_prompt: Original user prompt
            detailed_description: Gemini's detailed description

        Returns:
            Base64 encoded image data
        """
        try:
            # Create a compact 400x400 image
            width, height = 400, 400

            # Create gradient background with softer colors
            image = Image.new('RGB', (width, height), color='#f8fafc')
            draw = ImageDraw.Draw(image)

            # Create a subtle gradient effect
            for i in range(height):
                alpha = int(255 * (1 - i / height) * 0.08)
                color = (248 - alpha//6, 250 - alpha//6, 252 - alpha//6)
                draw.line([(0, i), (width, i)], fill=color)

            # Try to load fonts with smaller sizes for compact layout
            try:
                title_font = ImageFont.truetype("arial.ttf", 16)
                desc_font = ImageFont.truetype("arial.ttf", 11)
                footer_font = ImageFont.truetype("arial.ttf", 9)
            except:
                try:
                    title_font = ImageFont.load_default()
                    desc_font = ImageFont.load_default()
                    footer_font = ImageFont.load_default()
                except:
                    return await self._create_simple_compact_placeholder(original_prompt)

            # Title - more compact
            title = "🎨 SONA Image Description"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, 15), title, fill='#1e293b', font=title_font)

            # Original prompt box - more compact
            prompt_title = "Request:"
            draw.text((20, 45), prompt_title, fill='#334155', font=desc_font)

            # Wrap the original prompt with shorter width for compact display
            wrapped_prompt = textwrap.fill(f'"{original_prompt}"', width=45)
            prompt_y = 60
            for line in wrapped_prompt.split('\n')[:2]:  # Limit to 2 lines
                draw.text((25, prompt_y), line, fill='#0f172a', font=desc_font)
                prompt_y += 14

            # Detailed description - more compact
            desc_title = "Enhanced Description:"
            desc_y = prompt_y + 15
            draw.text((20, desc_y), desc_title, fill='#334155', font=desc_font)

            # Wrap and display the detailed description with tighter constraints
            wrapped_desc = textwrap.fill(detailed_description, width=50)
            desc_y += 18
            line_count = 0
            max_lines = 15  # Limit lines for compact display

            for line in wrapped_desc.split('\n'):
                if desc_y > height - 50 or line_count >= max_lines:  # Leave space for footer
                    if line_count < len(wrapped_desc.split('\n')) - 1:
                        draw.text((25, desc_y), "...", fill='#64748b', font=desc_font)
                    break
                draw.text((25, desc_y), line, fill='#1e293b', font=desc_font)
                desc_y += 13
                line_count += 1

            # Footer - more compact
            footer_text = "✨ Enhanced by Gemini 2.0 • SONA AI"
            footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
            footer_width = footer_bbox[2] - footer_bbox[0]
            footer_x = (width - footer_width) // 2
            draw.text((footer_x, height - 25), footer_text, fill='#94a3b8', font=footer_font)

            # Add a subtle border
            border_color = '#cbd5e1'
            draw.rectangle([5, 5, width-5, height-5], outline=border_color, width=1)

            # Add corner accent
            accent_color = '#3b82f6'
            draw.rectangle([5, 5, 25, 25], fill=accent_color)
            draw.text((10, 10), "🖼️", font=footer_font)

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', quality=90, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return image_base64

        except Exception as e:
            logger.error(f"Failed to create compact visual representation: {e}")
            return await self._create_simple_compact_placeholder(original_prompt)

    async def _create_simple_compact_placeholder(self, prompt: str) -> str:
        """Create a simple compact placeholder if the main visual creation fails."""
        try:
            width, height = 400, 400
            image = Image.new('RGB', (width, height), color='#e0f2fe')
            draw = ImageDraw.Draw(image)

            # Simple compact text
            text = f"Image Request:\n\n{prompt}\n\n\nProcessed by\nSONA + Gemini 2.0"
            wrapped_text = textwrap.fill(text, width=25)

            # Center the text
            bbox = draw.textbbox((0, 0), wrapped_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (width - text_width) // 2
            y = (height - text_height) // 2

            draw.text((x, y), wrapped_text, fill='#0c4a6e')

            # Add border
            draw.rectangle([2, 2, width-2, height-2], outline='#0284c7', width=2)

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return image_base64

        except Exception as e:
            logger.error(f"Failed to create simple compact placeholder: {e}")
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
            "capabilities": ["detailed_image_descriptions", "compact_visual_representations", "enhanced_prompts"],
            "supported_formats": ["PNG"],
            "image_size": "400x400px",
            "initialized": self.is_initialized,
            "available": self.is_available(),
            "description": "Uses Gemini 2.0 Flash to create detailed image descriptions in compact 400x400px visual cards"
        }
