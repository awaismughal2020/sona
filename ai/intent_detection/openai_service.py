"""
OpenAI-powered intent detection service implementation.
Handles intent classification and response generation using OpenAI GPT models.
"""
import openai
from typing import Dict, Any
import json
import asyncio
from loguru import logger

from .base import IntentDetectionBase
from config.settings import get_settings
from utils.constants import IntentType, ERROR_MESSAGES, SUCCESS_MESSAGES, SONA_PERSONA


class OpenAIIntentService(IntentDetectionBase):
    """OpenAI-powered intent detection service."""

    def __init__(self, **kwargs):
        """Initialize OpenAI intent detection service."""
        settings = get_settings()
        super().__init__(
            model_name="openai",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature,
            **kwargs
        )
        self.client = None

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("OpenAI API key is required")

            logger.info("Initializing OpenAI client for intent detection")

            # Initialize OpenAI client
            openai.api_key = self.config['api_key']
            self.client = openai.AsyncOpenAI(api_key=self.config['api_key'])

            # Test the connection
            await self._test_connection()

            self.is_initialized = True
            logger.info("OpenAI intent detection service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {e}")
            raise RuntimeError(f"OpenAI initialization failed: {e}")

    async def _test_connection(self) -> None:
        """Test OpenAI API connection."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config['model'],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            logger.info("OpenAI API connection test successful")
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise

    async def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        Detect intent from text using OpenAI.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing intent information
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Detecting intent for text: {text[:50]}...")

            # Create intent detection prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(text)

            # Get response from OpenAI
            response = await self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )

            # Parse the response
            response_text = response.choices[0].message.content.strip()
            result = self._parse_intent_response(response_text, text)

            logger.info(f"Intent detected: {result['intent']} (confidence: {result['confidence']})")
            return result

        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return {
                "intent": IntentType.UNKNOWN,
                "confidence": 0.0,
                "entities": {},
                "response": f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                "error": str(e)
            }

    def _create_system_prompt(self) -> str:
        """Create system prompt for intent detection."""
        return f"""You are {SONA_PERSONA['name']}, {SONA_PERSONA['personality']}.

Your task is to analyze user input and determine the intent, then provide an appropriate response.

Available intents:
1. WEB_SEARCH - User wants to search for information online (e.g., "what's the price of Bitcoin?", "best crypto exchanges in Pakistan")
2. IMAGE_GENERATION - User wants to generate an image (e.g., "make an image of children playing", "create a picture of...")
3. GENERAL_CHAT - General conversation, greetings, or questions that don't require web search or image generation

Instructions:
- Analyze the user's input carefully
- Determine the most appropriate intent
- If the user asks about current information, prices, news, or facts that require up-to-date data, classify as WEB_SEARCH
- If the user asks to generate, create, make, or draw an image, classify as IMAGE_GENERATION
- For greetings, general questions, or casual conversation, classify as GENERAL_CHAT
- Extract any relevant entities (search terms, image descriptions, etc.)
- Provide a helpful response based on the intent

Respond in this JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "entities": {{"key": "value"}},
    "response": "your response text"
}}"""

    def _create_user_prompt(self, text: str) -> str:
        """Create user prompt for intent detection."""
        return f"User input: \"{text}\"\n\nAnalyze this input and respond with the JSON format specified."

    def _parse_intent_response(self, response_text: str, original_text: str) -> Dict[str, Any]:
        """
        Parse OpenAI response into structured intent result.

        Args:
            response_text: Raw response from OpenAI
            original_text: Original user input

        Returns:
            Structured intent result
        """
        try:
            # Try to parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            result = json.loads(response_text)

            # Map intent string to IntentType enum
            intent_mapping = {
                "web_search": IntentType.WEB_SEARCH,
                "image_generation": IntentType.IMAGE_GENERATION,
                "general_chat": IntentType.GENERAL_CHAT
            }

            intent_str = result.get("intent", "unknown").lower()
            intent = intent_mapping.get(intent_str, IntentType.UNKNOWN)

            return {
                "intent": intent,
                "confidence": float(result.get("confidence", 0.0)),
                "entities": result.get("entities", {}),
                "response": result.get("response", "I'm not sure how to help with that."),
                "original_text": original_text
            }

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response_text}")
            # Fallback: simple keyword-based detection
            return self._fallback_intent_detection(original_text, response_text)
        except Exception as e:
            logger.error(f"Error parsing intent response: {e}")
            return self._fallback_intent_detection(original_text, response_text)

    def _fallback_intent_detection(self, text: str, ai_response: str = None) -> Dict[str, Any]:
        """
        Fallback intent detection using simple keyword matching.

        Args:
            text: Original user input
            ai_response: AI response if available

        Returns:
            Basic intent result
        """
        text_lower = text.lower()

        # Keywords for different intents
        search_keywords = ['price', 'cost', 'what is', 'who is', 'when', 'where', 'how much',
                           'latest', 'news', 'current', 'best', 'compare', 'find', 'search']
        image_keywords = ['image', 'picture', 'draw', 'create', 'make', 'generate', 'photo']

        # Determine intent based on keywords
        if any(keyword in text_lower for keyword in image_keywords):
            intent = IntentType.IMAGE_GENERATION
            confidence = 0.7
        elif any(keyword in text_lower for keyword in search_keywords):
            intent = IntentType.WEB_SEARCH
            confidence = 0.7
        else:
            intent = IntentType.GENERAL_CHAT
            confidence = 0.6

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": {"query": text} if intent == IntentType.WEB_SEARCH else
            {"prompt": text} if intent == IntentType.IMAGE_GENERATION else {},
            "response": ai_response or f"I understand you want to {intent.value.replace('_', ' ')}. Let me help you with that.",
            "original_text": text,
            "fallback": True
        }

    def is_available(self) -> bool:
        """
        Check if OpenAI service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"OpenAI availability check failed: {e}")
            return False

    async def get_model_info(self) -> dict:
        """
        Get information about the OpenAI model.

        Returns:
            Dictionary with model information
        """
        return {
            "service_name": self.model_name,
            "model": self.config.get('model', 'unknown'),
            "max_tokens": self.config.get('max_tokens', 150),
            "temperature": self.config.get('temperature', 0.7),
            "initialized": self.is_initialized,
            "available": self.is_available()
        }
