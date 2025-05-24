"""
FastAPI backend with real OpenAI integration for SONA AI Assistant.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, Any
import asyncio
from loguru import logger
import sys
import os
import openai
import requests
from serpapi import GoogleSearch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from utils.constants import SONA_PERSONA


class SONABackend:
    """SONA AI Assistant Backend with Real AI Integration."""

    def __init__(self):
        """Initialize SONA backend with AI services."""
        self.settings = get_settings()
        self.app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.app_version,
            description="AI-powered assistant with real OpenAI integration",
            debug=self.settings.debug
        )

        # Initialize AI clients
        self._setup_ai_clients()

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

        logger.info("SONA Real AI Backend initialized")

    def _setup_ai_clients(self):
        """Setup AI service clients."""
        try:
            # OpenAI client
            if self.settings.openai_api_key:
                openai.api_key = self.settings.openai_api_key
                self.openai_client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                self.openai_client = None
                logger.warning("OpenAI API key not provided")

            # SERP API (for web search)
            self.serp_api_key = self.settings.serp_api_key
            if self.serp_api_key:
                logger.info("SERP API key configured")
            else:
                logger.warning("SERP API key not provided")

        except Exception as e:
            logger.error(f"Failed to setup AI clients: {e}")

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with basic info."""
            return {
                "name": SONA_PERSONA["name"],
                "version": self.settings.app_version,
                "status": "running",
                "capabilities": SONA_PERSONA["capabilities"],
                "ai_services": {
                    "openai": bool(self.openai_client),
                    "serp_search": bool(self.serp_api_key)
                }
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": asyncio.get_event_loop().time(),
                "ai_services": {
                    "openai": "connected" if self.openai_client else "not_configured",
                    "serp": "configured" if self.serp_api_key else "not_configured"
                }
            }

        @self.app.post("/api/v1/chat")
        async def chat_endpoint(
                message: str = Form(...),
                session_id: Optional[str] = Form(None)
        ):
            """
            Main chat endpoint with real AI processing.
            """
            try:
                logger.info(f"Processing chat message: {message[:50]}...")

                # Detect intent and get response
                intent, response, data = await self._process_with_real_ai(message)

                return {
                    "success": True,
                    "response": response,
                    "intent": intent,
                    "response_type": "text",
                    "confidence": 0.9,
                    "data": data,
                    "session_id": session_id
                }

            except Exception as e:
                logger.error(f"Chat processing failed: {e}")
                return {
                    "success": False,
                    "response": f"I apologize, but I encountered an error: {str(e)}",
                    "intent": "error",
                    "response_type": "text",
                    "confidence": 0.0,
                    "session_id": session_id
                }

        @self.app.post("/api/v1/upload-audio")
        async def upload_audio_endpoint(
                audio_file: UploadFile = File(...),
                session_id: Optional[str] = Form(None)
        ):
            """
            Audio upload and processing endpoint with real speech-to-text.
            """
            try:
                logger.info(f"Processing audio upload: {audio_file.filename}")

                # Read audio file content
                audio_content = await audio_file.read()

                if not self.openai_client:
                    return {
                        "success": False,
                        "error": "OpenAI API not configured",
                        "transcription": "",
                        "response": "Speech-to-text requires OpenAI API key. Please configure your API key."
                    }

                # Use OpenAI Whisper for transcription
                transcribed_text = await self._transcribe_audio_with_whisper(audio_content, audio_file.filename)

                if not transcribed_text or not transcribed_text.strip():
                    return {
                        "success": False,
                        "error": "No speech detected",
                        "transcription": "",
                        "response": "I couldn't detect any speech in the audio file. Please try again with a clearer recording."
                    }

                # Process the transcribed text with AI
                intent, ai_response, data = await self._process_with_real_ai(transcribed_text)

                return {
                    "success": True,
                    "transcription": transcribed_text,
                    "response": ai_response,
                    "intent": intent,
                    "response_type": "text",
                    "confidence": 0.9,
                    "data": data,
                    "session_id": session_id
                }

            except Exception as e:
                logger.error(f"Audio processing failed: {e}")
                return {
                    "success": False,
                    "error": f"Audio processing failed: {str(e)}",
                    "transcription": "",
                    "response": f"Sorry, I couldn't process your audio file: {str(e)}"
                }

        @self.app.get("/api/v1/models")
        async def get_available_models():
            """Get available AI models for each service."""
            return {
                "success": True,
                "models": {
                    "speech_to_text": ["whisper-1"],
                    "intent_detection": ["gpt-3.5-turbo", "gpt-4"],
                    "image_generation": ["dall-e-2", "dall-e-3"],
                    "web_search": ["serp"]
                }
            }

    async def _process_with_real_ai(self, message: str) -> tuple[str, str, Optional[Dict]]:
        """
        Process message with real AI services.

        Returns:
            Tuple of (intent, response, data)
        """
        try:
            # First, use OpenAI to understand the intent and provide initial response
            if self.openai_client:
                intent_response = await self._get_openai_response(message)

                # Check if this needs web search
                if self._needs_web_search(message):
                    # Perform web search
                    search_results = await self._perform_web_search(message)
                    if search_results:
                        # Use OpenAI to synthesize the search results
                        final_response = await self._synthesize_search_results(message, search_results)
                        return "web_search", final_response, search_results

                # Check if this is an image generation request
                elif self._needs_image_generation(message):
                    return "image_generation", "I understand you want to generate an image. Image generation with DALL-E is not yet implemented, but I can help you with other tasks!", None

                # Regular chat response
                return "general_chat", intent_response, None

            else:
                # Fallback without OpenAI
                return "general_chat", "I'm running without OpenAI integration. Please configure your OpenAI API key to enable full AI capabilities.", None

        except Exception as e:
            logger.error(f"Real AI processing failed: {e}")
            return "error", f"I encountered an error while processing your request: {str(e)}", None

    async def _transcribe_audio_with_whisper(self, audio_content: bytes, filename: str) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            # Create a temporary file for the audio
            import tempfile
            import os

            # Get file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext:
                file_ext = '.mp3'  # Default extension

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name

            try:
                # Use OpenAI Whisper API for transcription
                with open(temp_file_path, "rb") as audio_file:
                    transcript = await self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )

                return transcript.strip() if transcript else ""

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise Exception(f"Speech transcription failed: {str(e)}")

    async def _get_openai_response(self, message: str) -> str:
        """Get response from OpenAI."""
        try:
            system_prompt = f"""You are {SONA_PERSONA['name']}, {SONA_PERSONA['personality']}.

You are an AI assistant that can:
- Answer questions and have conversations
- Help with information and explanations
- Provide helpful responses to user queries

Guidelines:
- Be friendly, helpful, and conversational
- Provide informative and accurate responses
- If you don't know something, say so honestly
- Keep responses concise but helpful
- Use a warm, professional tone"""

            response = await self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=self.settings.openai_max_tokens,
                temperature=self.settings.openai_temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"I'm having trouble connecting to my AI services. Error: {str(e)}"

    def _needs_web_search(self, message: str) -> bool:
        """Determine if message needs web search."""
        search_keywords = [
            'current', 'latest', 'recent', 'today', 'now', 'price', 'cost',
            'weather', 'news', 'what is happening', 'stock price',
            'cryptocurrency', 'bitcoin', 'ethereum', 'market',
            'exchange rate', 'currency', 'usd', 'pkr'
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in search_keywords)

    def _needs_image_generation(self, message: str) -> bool:
        """Determine if message is requesting image generation."""
        image_keywords = [
            'create image', 'generate image', 'make image', 'draw',
            'picture of', 'image of', 'show me image', 'create picture'
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in image_keywords)

    async def _perform_web_search(self, query: str) -> Optional[list]:
        """Perform web search using SERP API."""
        try:
            if not self.serp_api_key:
                return None

            search = GoogleSearch({
                "q": query,
                "api_key": self.serp_api_key,
                "num": 3
            })

            results = search.get_dict()

            if "organic_results" in results:
                formatted_results = []
                for result in results["organic_results"]:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("link", ""),
                        "source": result.get("source", "")
                    })
                return formatted_results

            return None

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return None

    async def _synthesize_search_results(self, query: str, search_results: list) -> str:
        """Use OpenAI to synthesize search results into a coherent response."""
        try:
            if not self.openai_client:
                return self._format_search_results_simple(query, search_results)

            # Create context from search results
            context = "Search results:\n"
            for i, result in enumerate(search_results, 1):
                context += f"{i}. {result['title']}\n{result['snippet']}\n\n"

            system_prompt = f"""You are {SONA_PERSONA['name']}, {SONA_PERSONA['personality']}.

The user asked: "{query}"

I've performed a web search and got the following results. Please synthesize this information into a helpful, informative response. Include key facts and figures, and mention that the information is from recent web search results.

Format your response in a clear, organized way with relevant details."""

            response = await self.openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=300,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Search synthesis failed: {e}")
            return self._format_search_results_simple(query, search_results)

    def _format_search_results_simple(self, query: str, search_results: list) -> str:
        """Format search results without OpenAI synthesis."""
        response = f"Here's what I found about '{query}':\n\n"

        for i, result in enumerate(search_results, 1):
            response += f"**{i}. {result['title']}**\n"
            response += f"{result['snippet']}\n\n"

        response += "Source: Recent web search results"
        return response

    def run(self, host: str = None, port: int = None):
        """Run the FastAPI application."""
        host = host or self.settings.backend_host
        port = port or self.settings.backend_port

        logger.info(f"Starting SONA Real AI backend on {host}:{port}")

        uvicorn.run(
            "backend.app:app",
            host=host,
            port=port,
            log_level=self.settings.log_level.lower(),
            reload=False
        )


# Create global app instance
sona_backend = SONABackend()
app = sona_backend.app

if __name__ == "__main__":
    sona_backend.run()
