"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Updated FastAPI backend with memory-persistent conversation context support.
Contexts remain in memory until explicitly deleted or container restart.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, Any, List
import asyncio
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from utils.constants import SONA_PERSONA


class SONABackendWithMemoryPersistentContext:
    """SONA AI Assistant Backend with Memory-Persistent Context Management."""

    def __init__(self):
        """Initialize SONA backend with AI services and memory-persistent context management."""
        self.settings = get_settings()
        self.app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.app_version,
            description="AI-powered assistant with memory-persistent conversation context",
            debug=self.settings.debug
        )

        # Initialize memory-persistent context store
        from utils.context_store import initialize_context_store
        # Increase default context limit since we're keeping everything in memory
        self.context_store = initialize_context_store(max_memory_contexts=2000)

        # Initialize enhanced AI orchestrator with context support
        from ai.orchestrator import EnhancedAIOrchestrator
        self.ai_orchestrator = EnhancedAIOrchestrator()
        self.is_ai_initialized = False

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

        logger.info("SONA Backend with Memory-Persistent Context Management initialized")

    async def _initialize_ai_services(self):
        """Initialize AI services if not already done."""
        if not self.is_ai_initialized:
            try:
                await self.ai_orchestrator.initialize()
                self.is_ai_initialized = True
                logger.info("Enhanced AI orchestrator with memory-persistent context initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced AI orchestrator: {e}")

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
        """Setup API routes with memory-persistent context support."""

        @self.app.get("/")
        async def root():
            """Root endpoint with basic info and memory context statistics."""
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

                    # Get memory context management stats
                    memory_usage = self.context_store.get_memory_usage()
                    context_stats = {
                        "active_sessions": memory_usage.get("contexts_in_memory", 0),
                        "memory_usage_percent": memory_usage.get("memory_usage_percent", 0),
                        "storage_type": "memory_persistent",
                        "persistence_until": "explicit_deletion_or_container_restart"
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
                "ai_services": ai_status,
                "memory_persistent_context": context_stats,
                "features": [
                    "Memory-persistent conversation context",
                    "Context-aware responses",
                    "Conversation memory until explicit deletion",
                    "Enhanced search with context",
                    "Contextual image generation",
                    "No automatic context cleanup"
                ]
            }

        @self.app.get("/health")
        async def health_check():
            """Enhanced health check endpoint with memory-persistent context information."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    memory_usage = self.context_store.get_memory_usage()

                    return {
                        "status": "healthy",
                        "timestamp": asyncio.get_event_loop().time(),
                        "ai_services": health,
                        "memory_persistent_context": {
                            "enabled": True,
                            "active_sessions": memory_usage.get("contexts_in_memory", 0),
                            "memory_usage_percent": memory_usage.get("memory_usage_percent", 0),
                            "storage_model": "memory_persistent_until_container_restart",
                            "automatic_cleanup": False,
                            "warning_threshold_reached": memory_usage.get("warning_threshold_reached", False),
                            "average_session_age_hours": memory_usage.get("average_session_age_hours", 0)
                        }
                    }
                except Exception as e:
                    return {
                        "status": "degraded",
                        "timestamp": asyncio.get_event_loop().time(),
                        "ai_services": {"error": str(e)},
                        "memory_persistent_context": {
                            "enabled": False,
                            "error": str(e)
                        }
                    }
            else:
                return {
                    "status": "starting",
                    "timestamp": asyncio.get_event_loop().time(),
                    "ai_services": {"status": "initializing"},
                    "memory_persistent_context": {
                        "enabled": False,
                        "status": "initializing"
                    }
                }

        @self.app.post("/api/v1/chat")
        async def chat_endpoint_with_memory_context(
                message: str = Form(...),
                session_id: Optional[str] = Form(None)
        ):
            """Enhanced chat endpoint with memory-persistent conversation context support."""
            try:
                logger.info(f"Processing chat message for memory-persistent session {session_id}: {message[:50]}...")

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

                # Use the enhanced AI orchestrator with memory-persistent context
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
                    "context_used": result.get("context_used", ""),
                    "persistence_model": "memory_until_deletion_or_restart"
                }

                return response_data

            except Exception as e:
                logger.error(f"Memory-persistent chat processing failed: {e}")
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
        async def upload_audio_endpoint_with_memory_context(
                audio_file: UploadFile = File(...),
                session_id: Optional[str] = Form(None)
        ):
            """Enhanced audio upload and processing endpoint with memory-persistent context support."""
            try:
                logger.info(f"Processing audio upload for memory-persistent session {session_id}: {audio_file.filename}")

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

                # Use enhanced AI orchestrator for speech-to-text with memory-persistent context
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
                    "context_used": result.get("context_used", ""),
                    "persistence_model": "memory_until_deletion_or_restart"
                }

            except Exception as e:
                logger.error(f"Memory-persistent audio processing failed: {e}")
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
                        "context": context_stats,
                        "persistence_model": "memory_until_deletion_or_restart"
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
                        "message": f"Context cleared for session {session_id}",
                        "note": "Context removed from memory permanently"
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
                all_stats = self.context_store.get_all_sessions_stats()

                session_info = []
                for session_id in sessions:
                    stats = all_stats.get(session_id, {})
                    if stats:
                        session_info.append({
                            "session_id": session_id,
                            "turn_count": stats.get("turn_count", 0),
                            "last_activity": stats.get("last_activity", 0),
                            "created_at": stats.get("created_at", 0),
                            "current_topic": stats.get("current_topic"),
                            "context_items": stats.get("context_items_count", 0),
                            "access_count": stats.get("access_count", 0),
                            "age_hours": round((asyncio.get_event_loop().time() - stats.get("created_at", 0)) / 3600, 2)
                        })

                memory_usage = self.context_store.get_memory_usage()

                return {
                    "success": True,
                    "active_sessions": len(sessions),
                    "sessions": session_info,
                    "memory_usage": memory_usage,
                    "persistence_model": "memory_until_deletion_or_restart",
                    "automatic_cleanup": False
                }

            except Exception as e:
                logger.error(f"Failed to list active sessions: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/v1/context/memory-usage")
        async def get_memory_usage():
            """Get detailed memory usage statistics."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                memory_usage = self.context_store.get_memory_usage()
                system_info = self.context_store.get_system_info()

                return {
                    "success": True,
                    "memory_usage": memory_usage,
                    "system_info": system_info,
                    "recommendations": self._get_memory_recommendations(memory_usage)
                }

            except Exception as e:
                logger.error(f"Failed to get memory usage: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/v1/context/sessions/old")
        async def get_old_sessions(older_than_hours: int = 24):
            """Get sessions older than specified hours (for manual cleanup consideration)."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                old_sessions = self.context_store.get_sessions_by_age(older_than_hours)

                session_details = []
                for session_id in old_sessions:
                    stats = self.context_store.get_context_stats(session_id)
                    if stats:
                        age_hours = (asyncio.get_event_loop().time() - stats.get("created_at", 0)) / 3600
                        session_details.append({
                            "session_id": session_id,
                            "age_hours": round(age_hours, 2),
                            "turn_count": stats.get("turn_count", 0),
                            "last_activity": stats.get("last_activity", 0),
                            "context_items": stats.get("context_items_count", 0)
                        })

                return {
                    "success": True,
                    "older_than_hours": older_than_hours,
                    "old_sessions_count": len(old_sessions),
                    "old_sessions": session_details,
                    "note": "These sessions are candidates for manual cleanup"
                }

            except Exception as e:
                logger.error(f"Failed to get old sessions: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.post("/api/v1/context/manual-cleanup")
        async def manual_cleanup_old_contexts(older_than_hours: int = Form(default=48)):
            """Manually clean up contexts older than specified hours."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                # Get sessions to be cleaned before cleanup
                sessions_to_cleanup = self.context_store.get_sessions_by_age(older_than_hours)

                # Perform manual cleanup
                cleaned_count = self.context_store.manual_cleanup_old_sessions(older_than_hours)

                return {
                    "success": True,
                    "message": f"Manually cleaned up {cleaned_count} old contexts",
                    "cleaned_contexts": cleaned_count,
                    "older_than_hours": older_than_hours,
                    "cleaned_sessions": sessions_to_cleanup[:cleaned_count],
                    "note": "This was a manual cleanup operation"
                }

            except Exception as e:
                logger.error(f"Manual context cleanup failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.post("/api/v1/context/clear-all")
        async def clear_all_contexts():
            """Clear all contexts from memory (nuclear option)."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                # Get count before clearing
                sessions_before = len(self.context_store.list_sessions())

                # Clear all contexts
                cleared_count = self.context_store.clear_all_contexts()

                return {
                    "success": True,
                    "message": f"Cleared all {cleared_count} contexts from memory",
                    "sessions_before": sessions_before,
                    "cleared_contexts": cleared_count,
                    "warning": "All conversation contexts have been permanently deleted"
                }

            except Exception as e:
                logger.error(f"Failed to clear all contexts: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        @self.app.get("/api/v1/context/{session_id}/export")
        async def export_session_context(session_id: str, format: str = "json"):
            """Export context for a specific session."""
            try:
                await self._initialize_ai_services()

                if not self.is_ai_initialized:
                    return {"success": False, "error": "AI services not initialized"}

                exported_data = self.context_store.export_context(session_id, format)

                if exported_data:
                    return {
                        "success": True,
                        "session_id": session_id,
                        "export_format": format,
                        "data": exported_data,
                        "exported_at": asyncio.get_event_loop().time()
                    }
                else:
                    return {
                        "success": False,
                        "error": "Session context not found or export failed"
                    }

            except Exception as e:
                logger.error(f"Failed to export session context: {e}")
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
                    "memory_persistent_context_features": {
                        "conversation_memory": True,
                        "multi_turn_context": True,
                        "contextual_search": True,
                        "contextual_image_generation": True,
                        "persistence_model": "memory_until_deletion_or_restart",
                        "automatic_cleanup": False
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
                        "note": "Memory-persistent context management continues with new model"
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
            """Get detailed AI services status with memory-persistent context management info."""
            await self._initialize_ai_services()

            if self.is_ai_initialized:
                try:
                    health = await self.ai_orchestrator.get_health_status()
                    memory_usage = self.context_store.get_memory_usage()
                    system_info = self.context_store.get_system_info()

                    return {
                        "success": True,
                        "status": health,
                        "memory_persistent_context_management": {
                            "enabled": True,
                            "storage_model": "memory_persistent_until_container_restart",
                            "memory_usage": memory_usage,
                            "system_info": system_info,
                            "automatic_cleanup": False,
                            "manual_cleanup_available": True
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
                    "memory_persistent_context_management": {
                        "enabled": False,
                        "status": "not_initialized"
                    }
                }

    def _get_memory_recommendations(self, memory_usage: Dict[str, Any]) -> list[str]:
        """Get recommendations based on memory usage."""
        recommendations = []

        usage_percent = memory_usage.get("memory_usage_percent", 0)

        if usage_percent > 90:
            recommendations.append("⚠️ Memory usage is very high. Consider manual cleanup of old sessions.")
        elif usage_percent > 70:
            recommendations.append("🔔 Memory usage is getting high. Monitor and consider cleanup if needed.")
        elif usage_percent < 20:
            recommendations.append("✅ Memory usage is healthy.")

        if memory_usage.get("warning_threshold_reached", False):
            recommendations.append("⚠️ Warning threshold reached. Manual cleanup recommended.")

        if memory_usage.get("emergency_threshold_reached", False):
            recommendations.append("🚨 Emergency threshold reached. Immediate cleanup required.")

        avg_age = memory_usage.get("average_session_age_hours", 0)
        if avg_age > 24:
            recommendations.append(f"📅 Average session age is {avg_age:.1f} hours. Consider cleanup of old sessions.")

        contexts_count = memory_usage.get("contexts_in_memory", 0)
        if contexts_count > 1000:
            recommendations.append(f"📊 High number of active sessions ({contexts_count}). Monitor performance.")

        return recommendations

    def run(self, host: str = None, port: int = None):
        """Run the FastAPI application."""
        host = host or self.settings.backend_host
        port = port or self.settings.backend_port

        logger.info(f"Starting SONA Backend with Memory-Persistent Context Management on {host}:{port}")
        logger.info("Context Persistence Model: Memory-only until explicit deletion or container restart")

        uvicorn.run(
            "backend.app:app",
            host=host,
            port=port,
            log_level=self.settings.log_level.lower(),
            reload=False
        )


# Create global app instance
sona_backend = SONABackendWithMemoryPersistentContext()
app = sona_backend.app

# For backward compatibility, also export the original class name
SONABackend = SONABackendWithMemoryPersistentContext

if __name__ == "__main__":
    sona_backend.run()
