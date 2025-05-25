"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

AI Orchestrator - Enhanced version with better response synthesis.
Handles routing between different AI components and manages service abstraction.
"""
from typing import Dict, Any, Optional, Union, List
import asyncio
from loguru import logger

from config.settings import get_settings
from utils.constants import IntentType, ModelType, ERROR_MESSAGES, SONA_PERSONA

# Import AI service classes
from .speech_to_text.whisper_service import WhisperService
from .intent_detection.openai_service import OpenAIIntentService
from .image_generation.gemini_service import GeminiImageService
from .web_search.serp_service import SerpSearchService


class AIOrchestrator:
    """
    Enhanced AI orchestrator with better response synthesis.
    Provides abstraction layer and handles service coordination.
    """

    def __init__(self):
        """Initialize AI orchestrator with all services."""
        self.settings = get_settings()

        # Initialize service instances
        self.services = {
            # Speech-to-Text Services
            "speech_to_text": {
                ModelType.WHISPER: WhisperService()
            },

            # Intent Detection Services
            "intent_detection": {
                ModelType.OPENAI: OpenAIIntentService()
            },

            # Image Generation Services
            "image_generation": {
                ModelType.GEMINI: GeminiImageService()
            },

            # Web Search Services
            "web_search": {
                ModelType.SERP: SerpSearchService()
            }
        }

        self.is_initialized = False
        self.active_models = {}

    async def initialize(self) -> None:
        """Initialize all configured AI services."""
        try:
            logger.info("Initializing AI Orchestrator...")

            # Set active models based on configuration
            self.active_models = {
                "speech_to_text": ModelType(self.settings.speech_to_text_model),
                "intent_detection": ModelType(self.settings.intent_detection_model),
                "image_generation": ModelType(self.settings.image_generation_model),
                "web_search": ModelType(self.settings.web_search_model)
            }

            # Initialize services concurrently
            initialization_tasks = []

            for service_type, model_type in self.active_models.items():
                if model_type in self.services[service_type]:
                    service = self.services[service_type][model_type]
                    initialization_tasks.append(
                        self._initialize_service(service, f"{service_type}_{model_type.value}")
                    )

            # Wait for all services to initialize
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

            # Check initialization results
            failed_services = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    service_name = list(self.active_models.keys())[i]
                    failed_services.append(f"{service_name}: {str(result)}")
                    logger.error(f"Failed to initialize {service_name}: {result}")

            if failed_services:
                logger.warning(f"Some services failed to initialize: {failed_services}")

            self.is_initialized = True
            logger.info("AI Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Orchestrator: {e}")
            raise RuntimeError(f"AI Orchestrator initialization failed: {e}")

    async def _initialize_service(self, service, service_name: str) -> None:
        """Initialize a single AI service."""
        try:
            await service.initialize()
            logger.info(f"Service {service_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {service_name}: {e}")
            raise

    async def process_audio(self, audio_data: Union[bytes, str]) -> str:
        """
        Process audio input using speech-to-text service.

        Args:
            audio_data: Audio data as bytes or file path

        Returns:
            Transcribed text
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get active speech-to-text service
            stt_model = self.active_models["speech_to_text"]
            stt_service = self.services["speech_to_text"][stt_model]

            if not stt_service.is_available():
                raise RuntimeError(f"Speech-to-text service {stt_model.value} is not available")

            # Transcribe audio
            text = await stt_service.transcribe_audio(audio_data)

            logger.info(f"Audio transcribed successfully: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise RuntimeError(f"Audio processing failed: {e}")

    async def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        Detect intent from text input.

        Args:
            text: Input text to analyze

        Returns:
            Intent detection result
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get active intent detection service
            intent_model = self.active_models["intent_detection"]
            intent_service = self.services["intent_detection"][intent_model]

            if not intent_service.is_available():
                raise RuntimeError(f"Intent detection service {intent_model.value} is not available")

            # Detect intent
            result = await intent_service.detect_intent(text)

            logger.info(f"Intent detected: {result['intent'].value} (confidence: {result['confidence']})")
            return result

        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return {
                "intent": IntentType.UNKNOWN,
                "confidence": 0.0,
                "entities": {},
                "response": f"I'm sorry, I encountered an error: {str(e)}",
                "error": str(e)
            }

    async def perform_web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using configured search service.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get active web search service
            search_model = self.active_models["web_search"]
            search_service = self.services["web_search"][search_model]

            if not search_service.is_available():
                raise RuntimeError(f"Web search service {search_model.value} is not available")

            # Perform search
            results = await search_service.search(query, num_results)

            logger.info(f"Web search completed: {len(results)} results found")
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise RuntimeError(f"Web search failed: {e}")

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using configured image generation service.

        Args:
            prompt: Image generation prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated image result
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get active image generation service
            image_model = self.active_models["image_generation"]
            image_service = self.services["image_generation"][image_model]

            if not image_service.is_available():
                raise RuntimeError(f"Image generation service {image_model.value} is not available")

            # Generate image
            result = await image_service.generate_image(prompt, **kwargs)

            logger.info(f"Image generation completed: {result.get('success', False)}")
            return result

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None,
                "image_data": None,
                "metadata": {"prompt": prompt, "error": str(e)}
            }

    async def _synthesize_search_results_with_ai(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Use AI to synthesize search results into a coherent answer.

        Args:
            query: Original search query
            search_results: List of search results

        Returns:
            Synthesized answer
        """
        try:
            if not search_results:
                return f"I couldn't find any information about '{query}'. Could you try rephrasing your question?"

            # Get the intent detection service (which uses OpenAI) to synthesize results
            intent_model = self.active_models["intent_detection"]
            intent_service = self.services["intent_detection"][intent_model]

            if not intent_service.is_available():
                return self._format_search_results_fallback(query, search_results)

            # Create context from search results
            context = f"User question: {query}\n\nSearch results:\n"
            for i, result in enumerate(search_results[:3], 1):  # Use top 3 results
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description available")
                url = result.get("url", "")
                context += f"{i}. {title}\n{snippet}\n{url}\n\n"

            # Create synthesis prompt
            synthesis_prompt = f"""Based on the search results provided, answer the user's question directly and concisely.

{context}

Instructions:
1. Provide a direct, factual answer to the question
2. Use information from the search results
3. Keep the answer concise but complete
4. If there are specific numbers or facts, include them
5. End with 1-2 reference links in this format: "Sources: [Title](URL)"

Answer the question: {query}"""

            # Use OpenAI to synthesize the answer
            client = intent_service.client
            response = await client.chat.completions.create(
                model=intent_service.config['model'],
                messages=[
                    {"role": "system",
                     "content": f"You are {SONA_PERSONA['name']}, {SONA_PERSONA['personality']}. Provide accurate, helpful answers based on search results."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=300,
                temperature=0.3  # Lower temperature for factual responses
            )

            synthesized_answer = response.choices[0].message.content.strip()

            # Ensure we have a good answer
            if synthesized_answer and len(synthesized_answer) > 20:
                logger.info("AI synthesis completed successfully")
                return synthesized_answer
            else:
                return self._format_search_results_fallback(query, search_results)

        except Exception as e:
            logger.error(f"AI synthesis failed: {e}")
            return self._format_search_results_fallback(query, search_results)

    def _format_search_results_fallback(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Fallback method to format search results when AI synthesis fails.

        Args:
            query: Original search query
            search_results: List of search results

        Returns:
            Formatted response string
        """
        if not search_results:
            return f"I couldn't find any results for '{query}'. Try rephrasing your search."

        # Try to extract a direct answer from the first result
        first_result = search_results[0]
        title = first_result.get("title", "")
        snippet = first_result.get("snippet", "")
        url = first_result.get("url", "")

        # Create a basic answer
        response = f"Based on my search about '{query}':\n\n"

        if snippet:
            response += f"{snippet}\n\n"

        # Add top 2 reference links
        response += "**Sources:**\n"
        for i, result in enumerate(search_results[:2], 1):
            result_title = result.get("title", "Search Result")
            result_url = result.get("url", "")
            if result_url:
                response += f"{i}. [{result_title}]({result_url})\n"

        return response

    async def process_user_input(self, input_text: str, input_type: str = "text") -> Dict[str, Any]:
        """
        Process user input end-to-end using SONA's capabilities with enhanced synthesis.

        Args:
            input_text: User input (text or transcribed from audio)
            input_type: Type of input ("text" or "audio")

        Returns:
            Complete response with appropriate action taken
        """
        try:
            logger.info(f"Processing user input ({input_type}): {input_text[:100]}...")

            # Step 1: Detect intent
            intent_result = await self.detect_intent(input_text)
            intent = intent_result["intent"]

            # Step 2: Process based on intent
            if intent == IntentType.WEB_SEARCH:
                # Extract search query from entities or use original text
                search_query = intent_result["entities"].get("query", input_text)
                search_results = await self.perform_web_search(search_query)

                # Use AI to synthesize the search results into a proper answer
                synthesized_response = await self._synthesize_search_results_with_ai(search_query, search_results)

                return {
                    "intent": intent.value,
                    "response_type": "text",
                    "response": synthesized_response,
                    "data": search_results,
                    "confidence": intent_result["confidence"]
                }

            elif intent == IntentType.IMAGE_GENERATION:
                # Extract image prompt from entities or use original text
                image_prompt = intent_result["entities"].get("prompt", input_text)
                image_result = await self.generate_image(image_prompt)

                if image_result["success"]:
                    response = f"I've created a detailed visual description for: '{image_prompt}'"

                    # Add enhanced description if available
                    if "enhanced_description" in image_result.get("metadata", {}):
                        enhanced_desc = image_result["metadata"]["enhanced_description"]
                        if enhanced_desc:
                            response += f"\n\n**Enhanced Description:** {enhanced_desc[:200]}..."
                else:
                    response = f"I'm sorry, I couldn't generate the image description. {image_result.get('error', '')}"

                return {
                    "intent": intent.value,
                    "response_type": "image",
                    "response": response,
                    "data": image_result,
                    "confidence": intent_result["confidence"]
                }

            elif intent == IntentType.GENERAL_CHAT:
                # Use the AI-generated response for general chat
                return {
                    "intent": intent.value,
                    "response_type": "text",
                    "response": intent_result["response"],
                    "data": None,
                    "confidence": intent_result["confidence"]
                }

            else:  # IntentType.UNKNOWN
                return {
                    "intent": intent.value,
                    "response_type": "text",
                    "response": "I'm not sure how to help with that. Could you please rephrase your request?",
                    "data": None,
                    "confidence": intent_result["confidence"]
                }

        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return {
                "intent": "error",
                "response_type": "text",
                "response": f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                "data": None,
                "confidence": 0.0,
                "error": str(e)
            }

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all AI services.

        Returns:
            Health status information
        """
        try:
            health_checks = []

            # Check each active service
            for service_type, model_type in self.active_models.items():
                if model_type in self.services[service_type]:
                    service = self.services[service_type][model_type]
                    health_check = await service.health_check()
                    health_checks.append(health_check)

            # Overall health status
            healthy_services = sum(1 for check in health_checks if check["status"] == "healthy")
            total_services = len(health_checks)

            overall_status = "healthy" if healthy_services == total_services else "degraded"
            if healthy_services == 0:
                overall_status = "unhealthy"

            return {
                "overall_status": overall_status,
                "services": health_checks,
                "summary": {
                    "healthy": healthy_services,
                    "total": total_services,
                    "initialized": self.is_initialized
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "unhealthy",
                "services": [],
                "summary": {"healthy": 0, "total": 0, "initialized": False},
                "error": str(e)
            }

    async def switch_model(self, service_type: str, model_type: str) -> bool:
        """
        Switch to a different model for a service type.

        Args:
            service_type: Type of service (e.g., "speech_to_text")
            model_type: New model type to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        try:
            if service_type not in self.services:
                logger.error(f"Unknown service type: {service_type}")
                return False

            new_model = ModelType(model_type)

            if new_model not in self.services[service_type]:
                logger.error(f"Model {model_type} not available for {service_type}")
                return False

            # Initialize new service if not already done
            new_service = self.services[service_type][new_model]
            if not new_service.is_initialized:
                await new_service.initialize()

            # Switch active model
            self.active_models[service_type] = new_model

            logger.info(f"Switched {service_type} to {model_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models for each service type.

        Returns:
            Dictionary mapping service types to available models
        """
        available_models = {}

        for service_type, models in self.services.items():
            available_models[service_type] = [model.value for model in models.keys()]

        return available_models
