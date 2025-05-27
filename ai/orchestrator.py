"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Enhanced AI Orchestrator with conversation context support.
This replaces the original ai/orchestrator.py file.
"""

from typing import Dict, Any, Optional, Union, List
import asyncio
from loguru import logger

from ai.image_generation import DalleImageService
from config.settings import get_settings
from utils.constants import IntentType, ModelType, ERROR_MESSAGES, SONA_PERSONA

# Import AI service classes
from ai.speech_to_text.whisper_service import WhisperService
from ai.intent_detection.openai_service import OpenAIIntentService
from ai.image_generation.gemini_service import GeminiImageService
from ai.web_search.serp_service import SerpSearchService


class EnhancedAIOrchestrator:
    """
    Enhanced AI orchestrator with conversation context management.
    Maintains context across conversation turns for better user experience.
    """

    def __init__(self):
        """Initialize enhanced AI orchestrator with context support."""
        self.settings = get_settings()
        # Import context store here to avoid circular imports
        from utils.context_store import get_context_store
        self.context_store = get_context_store()

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
                ModelType.GEMINI: GeminiImageService(),
                ModelType.DALLE: DalleImageService()
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
            logger.info("Initializing Enhanced AI Orchestrator with context support...")

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
            logger.info("Enhanced AI Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced AI Orchestrator: {e}")
            raise RuntimeError(f"Enhanced AI Orchestrator initialization failed: {e}")

    async def _initialize_service(self, service, service_name: str) -> None:
        """Initialize a single AI service."""
        try:
            await service.initialize()
            logger.info(f"Service {service_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {service_name}: {e}")
            raise

    def get_or_create_context(self, session_id: str) -> Any:
        """Get existing context or create new one for session."""
        try:
            context = self.context_store.get_context(session_id)
            if context is None:
                context = self.context_store.create_context(session_id)
                logger.info(f"Created new conversation context for session: {session_id}")
            else:
                logger.debug(f"Retrieved existing context for session: {session_id}")

            return context

        except Exception as e:
            logger.error(f"Failed to get/create context for session {session_id}: {e}")
            # Return basic context as fallback
            try:
                from utils.conversation_context import ConversationContext
                return ConversationContext(session_id)
            except ImportError as import_err:
                logger.error(f"Cannot import ConversationContext: {import_err}")
                raise RuntimeError(f"Context system unavailable: {e}")

    async def process_audio(self, audio_data: Union[bytes, str], session_id: str) -> str:
        """
        Process audio input using speech-to-text service.

        Args:
            audio_data: Audio data as bytes or file path
            session_id: Session identifier for context

        Returns:
            Transcribed text
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get conversation context
            context = self.get_or_create_context(session_id)

            # Get active speech-to-text service
            stt_model = self.active_models["speech_to_text"]
            stt_service = self.services["speech_to_text"][stt_model]

            if not stt_service.is_available():
                raise RuntimeError(f"Speech-to-text service {stt_model.value} is not available")

            # Transcribe audio
            text = await stt_service.transcribe_audio(audio_data)

            # Store audio processing context
            from utils.constants import ContextType
            context.add_context_item(
                ContextType.CONVERSATION_FLOW,
                "last_input_type",
                "audio",
                confidence=1.0,
                ttl_minutes=60
            )

            # Save context
            self.context_store.save_context(session_id, context)

            logger.info(f"Audio transcribed successfully: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise RuntimeError(f"Audio processing failed: {e}")

    async def detect_intent_with_context(self, text: str, session_id: str) -> Dict[str, Any]:
        """
        Detect intent from text input using conversation context.

        Args:
            text: Input text to analyze
            session_id: Session identifier for context

        Returns:
            Enhanced intent detection result with context
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get conversation context
            context = self.get_or_create_context(session_id)

            # Get active intent detection service
            intent_model = self.active_models["intent_detection"]
            intent_service = self.services["intent_detection"][intent_model]

            if not intent_service.is_available():
                raise RuntimeError(f"Intent detection service {intent_model.value} is not available")

            # Create enhanced prompt with context
            enhanced_text = self._create_contextual_prompt(text, context)

            # Detect intent
            result = await intent_service.detect_intent(enhanced_text)

            # Enhance result with context
            result["context_used"] = context.get_recent_context_for_prompt(max_turns=3)
            result["session_id"] = session_id

            logger.info(f"Intent detected with context: {result['intent'].value} (confidence: {result['confidence']})")
            return result

        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return {
                "intent": IntentType.UNKNOWN,
                "confidence": 0.0,
                "entities": {},
                "response": f"I'm sorry, I encountered an error: {str(e)}",
                "error": str(e),
                "session_id": session_id
            }

    def _create_contextual_prompt(self, text: str, context: Any) -> str:
        """Create enhanced prompt with conversation context."""
        try:
            # Get recent context
            recent_context = context.get_recent_context_for_prompt(max_turns=3)

            if not recent_context:
                return text

            # Create enhanced prompt
            enhanced_prompt = f"""
Context from previous conversation:
{recent_context}

Current user input: {text}

Please analyze this input considering the conversation context above.
"""
            return enhanced_prompt

        except Exception as e:
            logger.error(f"Failed to create contextual prompt: {e}")
            return text

    async def perform_contextual_web_search(self, query: str, session_id: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using conversation context to enhance query.

        Args:
            query: Search query
            session_id: Session identifier for context
            num_results: Number of results to return

        Returns:
            List of search results
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get conversation context
            context = self.get_or_create_context(session_id)

            # Enhance query with context
            enhanced_query = self._enhance_search_query(query, context)

            # Get active web search service
            search_model = self.active_models["web_search"]
            search_service = self.services["web_search"][search_model]

            if not search_service.is_available():
                raise RuntimeError(f"Web search service {search_model.value} is not available")

            # Perform search
            results = await search_service.search(enhanced_query, num_results)

            # Store search in context
            from utils.constants import ContextType
            context.add_context_item(
                ContextType.SEARCH_HISTORY,
                f"search_{int(asyncio.get_event_loop().time())}",
                query,
                confidence=0.9,
                ttl_minutes=120
            )

            # Extract and store entities from results
            self._extract_entities_from_search_results(results, context)

            # Save context
            self.context_store.save_context(session_id, context)

            logger.info(f"Contextual web search completed: {len(results)} results found")
            return results

        except Exception as e:
            logger.error(f"Contextual web search failed: {e}")
            raise RuntimeError(f"Web search failed: {e}")

    def _enhance_search_query(self, query: str, context: Any) -> str:
        """Enhance search query with conversation context."""
        try:
            # Get current topic for context
            if context.current_topic:
                # Check if topic is already in query
                if context.current_topic.lower() not in query.lower():
                    query = f"{query} {context.current_topic}"

            # Add relevant entities
            from utils.constants import ContextType
            entities = context.get_context_by_type(ContextType.ENTITY)
            for entity in entities[:2]:  # Add top 2 relevant entities
                entity_value = str(entity.value)
                if entity_value.lower() not in query.lower():
                    query = f"{query} {entity_value}"

            return query

        except Exception as e:
            logger.error(f"Failed to enhance search query: {e}")
            return query

    def _extract_entities_from_search_results(self, results: List[Dict[str, Any]], context: Any) -> None:
        """Extract entities from search results and add to context."""
        try:
            import re
            from utils.constants import ContextType

            for result in results[:3]:  # Process top 3 results
                title = result.get("title", "")
                snippet = result.get("snippet", "")

                # Extract capitalized terms (potential entities)
                text = f"{title} {snippet}"
                entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

                for entity in entities[:2]:  # Limit entities per result
                    if len(entity.split()) <= 2:  # Only short entities
                        context.add_context_item(
                            ContextType.ENTITY,
                            entity.lower().replace(' ', '_'),
                            entity,
                            confidence=0.5,
                            ttl_minutes=60
                        )

        except Exception as e:
            logger.error(f"Failed to extract entities from search results: {e}")

    async def generate_contextual_image(self, prompt: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using conversation context to enhance prompt.

        Args:
            prompt: Image generation prompt
            session_id: Session identifier for context
            **kwargs: Additional generation parameters

        Returns:
            Generated image result
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get conversation context
            context = self.get_or_create_context(session_id)

            # Enhance prompt with context
            enhanced_prompt = self._enhance_image_prompt(prompt, context)

            # Get active image generation service
            image_model = self.active_models["image_generation"]
            image_service = self.services["image_generation"][image_model]

            if not image_service.is_available():
                raise RuntimeError(f"Image generation service {image_model.value} is not available")

            # Generate image
            result = await image_service.generate_image(enhanced_prompt, **kwargs)

            # Store image generation in context
            from utils.constants import ContextType
            if result.get("success"):
                context.add_context_item(
                    ContextType.IMAGE_HISTORY,
                    f"image_{int(asyncio.get_event_loop().time())}",
                    prompt,
                    confidence=0.9,
                    ttl_minutes=180
                )

            # Save context
            self.context_store.save_context(session_id, context)

            logger.info(f"Contextual image generation completed: {result.get('success', False)}")
            return result

        except Exception as e:
            logger.error(f"Contextual image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None,
                "image_data": None,
                "metadata": {"prompt": prompt, "error": str(e)}
            }

    def _enhance_image_prompt(self, prompt: str, context: Any) -> str:
        """Enhance image prompt with conversation context."""
        try:
            # Get recent image history for style consistency
            from utils.constants import ContextType
            image_history = context.get_context_by_type(ContextType.IMAGE_HISTORY)

            # If user has previous image requests, maintain similar style
            if image_history:
                recent_image = image_history[0]
                # This is a simple enhancement - could be more sophisticated
                if "style" not in prompt.lower() and "art" in str(recent_image.value).lower():
                    prompt = f"{prompt}, artistic style"

            return prompt

        except Exception as e:
            logger.error(f"Failed to enhance image prompt: {e}")
            return prompt

    async def process_user_input_with_context(self, input_text: str, session_id: str, input_type: str = "text") -> Dict[str, Any]:
        """
        Process user input end-to-end with full context awareness.

        Args:
            input_text: User input (text or transcribed from audio)
            session_id: Session identifier for context
            input_type: Type of input ("text" or "audio")

        Returns:
            Complete response with context-aware processing
        """
        try:
            logger.info(f"Processing contextual user input ({input_type}) for session {session_id}: {input_text[:100]}...")

            # Get conversation context
            context = self.get_or_create_context(session_id)

            # Step 1: Detect intent with context
            intent_result = await self.detect_intent_with_context(input_text, session_id)
            intent = intent_result["intent"]

            # Step 2: Process based on intent with context
            if intent == IntentType.WEB_SEARCH:
                # Extract search query from entities or use original text
                search_query = intent_result["entities"].get("query", input_text)
                search_results = await self.perform_contextual_web_search(search_query, session_id)

                # Use AI to synthesize the search results into a proper answer
                synthesized_response = await self._synthesize_search_results_with_context(
                    search_query, search_results, context
                )

                response_data = {
                    "intent": intent.value,
                    "response_type": "text",
                    "response": synthesized_response,
                    "data": search_results,
                    "confidence": intent_result["confidence"],
                    "context_used": intent_result.get("context_used", "")
                }

            elif intent == IntentType.IMAGE_GENERATION:
                # Extract image prompt from entities or use original text
                image_prompt = intent_result["entities"].get("prompt", input_text)
                image_result = await self.generate_contextual_image(image_prompt, session_id)

                if image_result["success"]:
                    response = f"I've created a detailed visual description for: '{image_prompt}'"

                    # Add enhanced description if available
                    if "metadata" in image_result and "detailed_description" in image_result["metadata"]:
                        enhanced_desc = image_result["metadata"]["detailed_description"]
                        if enhanced_desc:
                            response += f"\n\n**Enhanced Description:** {enhanced_desc[:200]}..."
                else:
                    response = f"I'm sorry, I couldn't generate the image description. {image_result.get('error', '')}"

                response_data = {
                    "intent": intent.value,
                    "response_type": "image",
                    "response": response,
                    "data": image_result,
                    "confidence": intent_result["confidence"],
                    "context_used": intent_result.get("context_used", "")
                }

            elif intent == IntentType.GENERAL_CHAT:
                # Use the AI-generated response for general chat with context awareness
                response_data = {
                    "intent": intent.value,
                    "response_type": "text",
                    "response": intent_result["response"],
                    "data": None,
                    "confidence": intent_result["confidence"],
                    "context_used": intent_result.get("context_used", "")
                }

            else:  # IntentType.UNKNOWN
                response_data = {
                    "intent": intent.value,
                    "response_type": "text",
                    "response": "I'm not sure how to help with that. Could you please rephrase your request?",
                    "data": None,
                    "confidence": intent_result["confidence"],
                    "context_used": intent_result.get("context_used", "")
                }

            # Add conversation turn to context
            turn_id = context.add_conversation_turn(
                user_input=input_text,
                assistant_response=response_data["response"],
                intent=intent.value,
                confidence=intent_result["confidence"],
                metadata={
                    "input_type": input_type,
                    "response_type": response_data["response_type"],
                    "context_used": response_data.get("context_used", "")
                }
            )

            # Save updated context
            self.context_store.save_context(session_id, context)

            # Add turn ID to response
            response_data["turn_id"] = turn_id
            response_data["session_id"] = session_id

            return response_data

        except Exception as e:
            logger.error(f"Failed to process contextual user input: {e}")
            return {
                "intent": "error",
                "response_type": "text",
                "response": f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                "data": None,
                "confidence": 0.0,
                "error": str(e),
                "session_id": session_id
            }

    async def _synthesize_search_results_with_context(self,
                                                     query: str,
                                                     search_results: List[Dict[str, Any]],
                                                     context: Any) -> str:
        """
        Use AI to synthesize search results with conversation context.

        Args:
            query: Original search query
            search_results: List of search results
            context: Conversation context

        Returns:
            Synthesized answer with context awareness
        """
        try:
            if not search_results:
                return f"I couldn't find any information about '{query}'. Could you try rephrasing your question?"

            # Get the intent detection service (which uses OpenAI) to synthesize results
            intent_model = self.active_models["intent_detection"]
            intent_service = self.services["intent_detection"][intent_model]

            if not intent_service.is_available():
                return self._format_search_results_fallback(query, search_results)

            # Create context from search results with conversation context
            recent_context = context.get_recent_context_for_prompt(max_turns=2)

            context_prompt = f"User question: {query}\n\n"

            if recent_context:
                context_prompt += f"Conversation context:\n{recent_context}\n\n"

            context_prompt += "Search results:\n"

            for i, result in enumerate(search_results[:3], 1):  # Use top 3 results
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description available")
                url = result.get("url", "")
                context_prompt += f"{i}. {title}\n{snippet}\n{url}\n\n"

            # Create synthesis prompt
            synthesis_prompt = f"""Based on the search results and conversation context provided, answer the user's question directly and naturally.

{context_prompt}

Instructions:
1. Provide a direct, conversational answer that considers the previous context
2. Use information from the search results
3. Keep the answer natural and engaging
4. If there are specific numbers or facts, include them
5. Reference the conversation context when relevant
6. End with 1-2 reference links in this format: "Sources: [Title](URL)"

Answer the question naturally: {query}"""

            # Use OpenAI to synthesize the answer
            client = intent_service.client
            response = await client.chat.completions.create(
                model=intent_service.config['model'],
                messages=[
                    {"role": "system",
                     "content": f"You are {SONA_PERSONA['name']}, {SONA_PERSONA['personality']}. Provide accurate, helpful answers that consider conversation context."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=400,
                temperature=0.4  # Balanced temperature for natural yet factual responses
            )

            synthesized_answer = response.choices[0].message.content.strip()

            # Ensure we have a good answer
            if synthesized_answer and len(synthesized_answer) > 20:
                logger.info("AI synthesis with context completed successfully")
                return synthesized_answer
            else:
                return self._format_search_results_fallback(query, search_results)

        except Exception as e:
            logger.error(f"AI synthesis with context failed: {e}")
            return self._format_search_results_fallback(query, search_results)

    def _format_search_results_fallback(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Fallback method to format search results when AI synthesis fails."""
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

    def get_context_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get context statistics for a session."""
        return self.context_store.get_context_stats(session_id)

    def clear_session_context(self, session_id: str) -> bool:
        """Clear context for a specific session."""
        return self.context_store.delete_context(session_id)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all AI services and context store."""
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

            # Add context store information
            context_info = {
                "context_store": {
                    "status": "healthy",
                    "memory_usage": self.context_store.get_memory_usage(),
                    "active_sessions": len(self.context_store.list_sessions())
                }
            }

            return {
                "overall_status": overall_status,
                "services": health_checks,
                "summary": {
                    "healthy": healthy_services,
                    "total": total_services,
                    "initialized": self.is_initialized
                },
                "context_management": context_info
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


# For backward compatibility, also export the original AIOrchestrator name
AIOrchestrator = EnhancedAIOrchestrator
