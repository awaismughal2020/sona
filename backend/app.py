"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Updated FastAPI backend with conversation context support for SONA AI Assistant.
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
# Import with lazy loading to avoid circular imports


class SONABackendWithContext:
    """SONA AI Assistant Backend with Full Context Management."""

    def __init__(self):
        """Initialize SONA backend with AI services and context management."""
        self.settings = get_settings()
        self.app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.app_version,
            description="AI-powered assistant with conversation context management",
            debug=self.settings.debug
        )

        # Initialize context store
        from utils.context_store import initialize_context_store
        self.context_store = initialize_context_store()

        # Initialize enhanced AI orchestrator with context support
        from ai.orchestrator import EnhancedAIOrchestrator
        self.ai_orchestrator = EnhancedAIOrchestrator()
        self.is_ai_initialized = False

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

        logger.info("SONA Backend with Context Management initialized")

    async def _initialize_ai_services(self):
        """Initialize AI services if not already done."""
        if not self.is_ai_initialized:
            try:
                await self.ai_orchestrator.initialize()
                self.is_ai_initialized = True
                logger.info("Enhanced AI orchestrator with context initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced AI orchestrator: {e}")
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
        """Setup API routes with context support."""

        @self.app.get("/")
        async def root():
            """Root endpoint with basic info and context statistics."""
            # Get AI service status
            ai_status = {}
            context_stats = {}

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    ai_status = {
                        "status": health["overall_status"],
                        "services": len(health["services"]),
                        "healthy_services": health["summary"]["healthy"]
                    }

                    # Get context management stats
                    if "context_management" in health:
                        context_stats = health["context_management"]["context_store"]

                except:
                    ai_status = {"status": "unknown"}
            else:
                ai_status = {"status": "not_initialized"}

            return {
                "name": SONA_PERSONA["name"],
                "version": self.settings.app_version,
                "status": "running",
                "capabilities": SONA_PERSONA["capabilities"],
                "ai_services": ai_status,
                "context_management": context_stats,
                "features": [
                    "Multi-turn conversation context",
                    "Context-aware responses",
                    "Persistent conversation memory",
                    "Enhanced search with context",
                    "Contextual image generation"
                ]
            }

        @self.app.get("/health")
        async def health_check():
            """Enhanced health check endpoint with context information."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    return {
                        "status": "healthy",
                        "timestamp": asyncio.get_event_loop().time(),
                        "ai_services": health,
                        "context_features": {
                            "context_persistence": True,
                            "multi_turn_awareness": True,
                            "session_management": True
                        }
                    }
                except Exception as e:
                    return {
                        "status": "degraded",
                        "timestamp": asyncio.get_event_loop().time(),
                        "ai_services": {"error": str(e)},
                        "context_features": {
                            "context_persistence": False,
                            "multi_turn_awareness": False,
                            "session_management": False
                        }
                    }
            else:
                return {
                    "status": "starting",
                    "timestamp": asyncio.get_event_loop().time(),
                    "ai_services": {"status": "initializing"},
                    "context_features": {
                        "context_persistence": False,
                        "multi_turn_awareness": False,
                        "session_management": False
                    }
                }

        @self.app.post("/api/v1/chat")
        async def chat_endpoint_with_context(
                message: str = Form(...),
                session_id: Optional[str] = Form(None)
        ):
            """
            Enhanced chat endpoint with full conversation context support.
            """
            try:
                logger.info(f"Processing contextual chat message for session {session_id}: {message[:50]}...")

                # Initialize AI services if needed
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {
                        "success": False,
                        "response": "AI services are still initializing. Please try again in a moment.",
                        "intent": "error",
                        "response_type": "text",
                        "confidence": 0.0,
                        "session_id": session_id,
                        "context_aware": False
                    }

                # Ensure session ID is provided
                if not session_id:
                    session_id = f"session_{int(asyncio.get_event_loop().time())}"

                # Use the enhanced AI orchestrator with context
                result = await self.ai_orchestrator.process_user_input_with_context(
                    message, session_id, "text"
                )

                # Format the response for the frontend
                response_data = {
                    "success": True,
                    "response": result["response"],
                    "intent": result["intent"],
                    "response_type": result["response_type"],
                    "confidence": result.get("confidence", 0.9),
                    "data": result.get("data"),
                    "session_id": session_id,
                    "turn_id": result.get("turn_id"),
                    "context_aware": True,
                    "context_used": result.get("context_used", "")
                }

                # Add context information if available
                if result.get("context_used"):
                    response_data["context_summary"] = "Used previous conversation context"

                return response_data

            except Exception as e:
                logger.error(f"Contextual chat processing failed: {e}")
                return {
                    "success": False,
                    "response": f"I apologize, but I encountered an error: {str(e)}",
                    "intent": "error",
                    "response_type": "text",
                    "confidence": 0.0,
                    "session_id": session_id,
                    "context_aware": False
                }

        @self.app.post("/api/v1/upload-audio")
        async def upload_audio_endpoint_with_context(
                audio_file: UploadFile = File(...),
                session_id: Optional[str] = Form(None)
        ):
            """
            Enhanced audio upload and processing endpoint with context support.
            """
            try:
                logger.info(f"Processing contextual audio upload for session {session_id}: {audio_file.filename}")

                # Initialize AI services if needed
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {
                        "success": False,
                        "error": "AI services are still initializing",
                        "transcription": "",
                        "response": "Speech-to-text services are starting up. Please try again in a moment.",
                        "context_aware": False
                    }

                # Ensure session ID is provided
                if not session_id:
                    session_id = f"session_{int(asyncio.get_event_loop().time())}"

                # Read audio file content
                audio_content = await audio_file.read()

                # Use enhanced AI orchestrator for speech-to-text with context
                try:
                    transcribed_text = await self.ai_orchestrator.process_audio(audio_content, session_id)
                except Exception as e:
                    logger.error(f"Audio transcription failed: {e}")
                    return {
                        "success": False,
                        "error": f"Speech transcription failed: {str(e)}",
                        "transcription": "",
                        "response": "I couldn't process your audio file. Please try again with a clear recording.",
                        "context_aware": False
                    }

                if not transcribed_text or not transcribed_text.strip():
                    return {
                        "success": False,
                        "error": "No speech detected",
                        "transcription": "",
                        "response": "I couldn't detect any speech in the audio file. Please try again with a clearer recording.",
                        "context_aware": False
                    }

                # Process the transcribed text with full context-aware AI pipeline
                result = await self.ai_orchestrator.process_user_input_with_context(
                    transcribed_text, session_id, "audio"
                )

                return {
                    "success": True,
                    "transcription": transcribed_text,
                    "response": result["response"],
                    "intent": result["intent"],
                    "response_type": result["response_type"],
                    "confidence": result.get("confidence", 0.9),
                    "data": result.get("data"),
                    "session_id": session_id,
                    "turn_id": result.get("turn_id"),
                    "context_aware": True,
                    "context_used": result.get("context_used", "")
                }

            except Exception as e:
                logger.error(f"Contextual audio processing failed: {e}")
                return {
                    "success": False,
                    "error": f"Audio processing failed: {str(e)}",
                    "transcription": "",
                    "response": f"Sorry, I couldn't process your audio file: {str(e)}",
                    "context_aware": False
                }

        @self.app.get("/api/v1/context/{session_id}")
        async def get_session_context(session_id: str):
            """Get context information for a specific session."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                context_stats = self.ai_orchestrator.get_context_stats(session_id)

                if context_stats:
                    return {
                        "success": True,
                        "session_id": session_id,
                        "context": context_stats
                    }
                else:
                    return {
                        "success": False,
                        "error": "Session context not found"
                    }

            except Exception as e:
                logger.error(f"Failed to get session context: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.delete("/api/v1/context/{session_id}")
        async def clear_session_context(session_id: str):
            """Clear context for a specific session."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                success = self.ai_orchestrator.clear_session_context(session_id)

                if success:
                    return {
                        "success": True,
                        "message": f"Context cleared for session {session_id}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Failed to clear session context"
                    }

            except Exception as e:
                logger.error(f"Failed to clear session context: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/v1/context")
        async def list_active_sessions():
            """List all active sessions with context."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                sessions = self.context_store.list_sessions()

                session_info = []
                for session_id in sessions:
                    stats = self.context_store.get_context_stats(session_id)
                    if stats:
                        session_info.append({
                            "session_id": session_id,
                            "turn_count": stats.get("turn_count", 0),
                            "last_activity": stats.get("last_activity", 0),
                            "current_topic": stats.get("current_topic"),
                            "context_items": stats.get("context_items_count", 0)
                        })

                return {
                    "success": True,
                    "active_sessions": len(sessions),
                    "sessions": session_info
                }

            except Exception as e:
                logger.error(f"Failed to list active sessions: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/v1/models")
        async def get_available_models():
            """Get available AI models for each service."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                available_models = self.ai_orchestrator.get_available_models()
                return {
                    "success": True,
                    "models": available_models,
                    "context_features": {
                        "conversation_memory": True,
                        "multi_turn_context": True,
                        "contextual_search": True,
                        "contextual_image_generation": True
                    }
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
                        "message": f"Switched {service_type} to {model_type}",
                        "note": "Context management continues with new model"
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
            """Get detailed AI services status with context management info."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    return {
                        "success": True,
                        "status": health,
                        "context_management": {
                            "enabled": True,
                            "persistent_storage": True,
                            "memory_cache": True,
                            "session_tracking": True
                        }
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            else:
                return {
                    "success": False,
                    "error": "AI services not initialized",
                    "context_management": {
                        "enabled": False,
                        "persistent_storage": False,
                        "memory_cache": False,
                        "session_tracking": False
                    }
                }

        @self.app.post("/api/v1/context/cleanup")
        async def cleanup_old_contexts():
            """Clean up old/expired contexts."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                # Clean up contexts older than 24 hours
                cleaned_count = self.context_store.cleanup_expired_contexts(max_age_hours=24)

                return {
                    "success": True,
                    "message": f"Cleaned up {cleaned_count} expired contexts",
                    "cleaned_contexts": cleaned_count
                }

            except Exception as e:
                logger.error(f"Context cleanup failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def run(self, host: str = None, port: int = None):
        """Run the FastAPI application."""
        host = host or self.settings.backend_host
        port = port or self.settings.backend_port

        logger.info(f"Starting SONA Backend with Context Management on {host}:{port}")

        uvicorn.run(
            "backend.app:app",
            host=host,
            port=port,
            log_level=self.settings.log_level.lower(),
            reload=False
        )


# Create global app instance
sona_backend = SONABackendWithContext()
app = sona_backend.app

# For backward compatibility, also export the original class name
SONABackend = SONABackendWithContext

if __name__ == "__main__":
    sona_backend.run()
