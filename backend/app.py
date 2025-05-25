"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

FastAPI backend with real AI integration including proper image generation for SONA AI Assistant.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, Any
import asyncio
from loguru import logger
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from utils.constants import SONA_PERSONA
from ai.orchestrator import AIOrchestrator  # Import the orchestrator


class SONABackend:
    """SONA AI Assistant Backend with Full AI Integration."""

    def __init__(self):
        """Initialize SONA backend with AI services."""
        self.settings = get_settings()
        self.app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.app_version,
            description="AI-powered assistant with full AI service integration",
            debug=self.settings.debug
        )

        # Initialize AI orchestrator instead of separate clients
        self.ai_orchestrator = AIOrchestrator()
        self.is_ai_initialized = False

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

        logger.info("SONA Full AI Backend initialized")

    async def _initialize_ai_services(self):
        """Initialize AI services if not already done."""
        if not self.is_ai_initialized:
            try:
                await self.ai_orchestrator.initialize()
                self.is_ai_initialized = True
                logger.info("AI orchestrator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI orchestrator: {e}")
                # Don't raise - let the app start but with limited functionality

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
            # Get AI service status
            ai_status = {}
            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    ai_status = {
                        "status": health["overall_status"],
                        "services": len(health["services"]),
                        "healthy_services": health["summary"]["healthy"]
                    }
                except:
                    ai_status = {"status": "unknown"}
            else:
                ai_status = {"status": "not_initialized"}

            return {
                "name": SONA_PERSONA["name"],
                "version": self.settings.app_version,
                "status": "running",
                "capabilities": SONA_PERSONA["capabilities"],
                "ai_services": ai_status
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    return {
                        "status": "healthy",
                        "timestamp": asyncio.get_event_loop().time(),
                        "ai_services": health
                    }
                except Exception as e:
                    return {
                        "status": "degraded",
                        "timestamp": asyncio.get_event_loop().time(),
                        "ai_services": {"error": str(e)}
                    }
            else:
                return {
                    "status": "starting",
                    "timestamp": asyncio.get_event_loop().time(),
                    "ai_services": {"status": "initializing"}
                }

        @self.app.post("/api/v1/chat")
        async def chat_endpoint(
                message: str = Form(...),
                session_id: Optional[str] = Form(None)
        ):
            """
            Main chat endpoint with full AI orchestrator integration.
            """
            try:
                logger.info(f"Processing chat message: {message[:50]}...")

                # Initialize AI services if needed
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {
                        "success": False,
                        "response": "AI services are still initializing. Please try again in a moment.",
                        "intent": "error",
                        "response_type": "text",
                        "confidence": 0.0,
                        "session_id": session_id
                    }

                # Use the AI orchestrator to process the complete user input
                result = await self.ai_orchestrator.process_user_input(message, "text")

                # Format the response for the frontend
                response_data = {
                    "success": True,
                    "response": result["response"],
                    "intent": result["intent"],
                    "response_type": result["response_type"],
                    "confidence": result.get("confidence", 0.9),
                    "data": result.get("data"),
                    "session_id": session_id
                }

                return response_data

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
            Audio upload and processing endpoint with full AI integration.
            """
            try:
                logger.info(f"Processing audio upload: {audio_file.filename}")

                # Initialize AI services if needed
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {
                        "success": False,
                        "error": "AI services are still initializing",
                        "transcription": "",
                        "response": "Speech-to-text services are starting up. Please try again in a moment."
                    }

                # Read audio file content
                audio_content = await audio_file.read()

                # Use AI orchestrator for speech-to-text
                try:
                    transcribed_text = await self.ai_orchestrator.process_audio(audio_content)
                except Exception as e:
                    logger.error(f"Audio transcription failed: {e}")
                    return {
                        "success": False,
                        "error": f"Speech transcription failed: {str(e)}",
                        "transcription": "",
                        "response": "I couldn't process your audio file. Please try again with a clear recording."
                    }

                if not transcribed_text or not transcribed_text.strip():
                    return {
                        "success": False,
                        "error": "No speech detected",
                        "transcription": "",
                        "response": "I couldn't detect any speech in the audio file. Please try again with a clearer recording."
                    }

                # Process the transcribed text with full AI pipeline
                result = await self.ai_orchestrator.process_user_input(transcribed_text, "audio")

                return {
                    "success": True,
                    "transcription": transcribed_text,
                    "response": result["response"],
                    "intent": result["intent"],
                    "response_type": result["response_type"],
                    "confidence": result.get("confidence", 0.9),
                    "data": result.get("data"),
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
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                available_models = self.ai_orchestrator.get_available_models()
                return {
                    "success": True,
                    "models": available_models
                }
            else:
                return {
                    "success": False,
                    "models": {},
                    "error": "AI services not initialized"
                }

        @self.app.post("/api/v1/switch-model")
        async def switch_model_endpoint(
                service_type: str = Form(...),
                model_type: str = Form(...)
        ):
            """Switch AI model for a service type."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                success = await self.ai_orchestrator.switch_model(service_type, model_type)

                if success:
                    return {
                        "success": True,
                        "message": f"Switched {service_type} to {model_type}"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to switch {service_type} to {model_type}"
                    }

            except Exception as e:
                logger.error(f"Model switching failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/v1/ai-status")
        async def get_ai_status():
            """Get detailed AI services status."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    return {
                        "success": True,
                        "status": health
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            else:
                return {
                    "success": False,
                    "error": "AI services not initialized"
                }

    def run(self, host: str = None, port: int = None):
        """Run the FastAPI application."""
        host = host or self.settings.backend_host
        port = port or self.settings.backend_port

        logger.info(f"Starting SONA Full AI backend on {host}:{port}")

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
