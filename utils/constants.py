"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Constants used throughout the SONA AI Assistant application.
"""

from enum import Enum


class IntentType(Enum):
    """Enumeration of supported intent types."""
    WEB_SEARCH = "web_search"
    IMAGE_GENERATION = "image_generation"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Enumeration of supported AI model types."""
    # Speech-to-Text Models
    WHISPER = "whisper"
    DEEPSPEECH = "deepspeech"

    # Intent Detection Models
    OPENAI = "openai"
    LOCAL_TRANSFORMER = "local_transformer"

    # Image Generation Models
    GEMINI = "gemini"

    # Web Search Models
    SERP = "serp"


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"


class ResponseType(Enum):
    """Types of responses from AI services."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    ERROR = "error"

class ContextType(Enum):
    """Types of context information."""
    USER_PREFERENCE = "user_preference"
    TOPIC = "topic"
    ENTITY = "entity"
    TASK = "task"
    SEARCH_HISTORY = "search_history"
    IMAGE_HISTORY = "image_history"
    CONVERSATION_FLOW = "conversation_flow"


# SONA Persona Configuration
SONA_PERSONA = {
    "name": "SONA: The Mobile Assistant",
    "personality": "friendly, efficient, and always helpful",
    "greeting": "Hi! I'm SONA, your AI assistant. How can I help you today?",
    "capabilities": [
        "Answer questions by searching the web",
        "Generate images based on your descriptions",
        "Process both text and voice commands",
        "Maintain helpful conversations"
    ]
}

# Error Messages
ERROR_MESSAGES = {
    "api_key_missing": "API key is missing for the requested service",
    "model_not_supported": "The specified model is not supported",
    "audio_processing_failed": "Failed to process audio input",
    "network_error": "Network error occurred while processing request",
    "invalid_input": "Invalid input provided",
    "service_unavailable": "Service temporarily unavailable",
    "file_too_large": "File size exceeds maximum limit",
    "unsupported_format": "Unsupported file format"
}

# Success Messages
SUCCESS_MESSAGES = {
    "audio_processed": "Audio processed successfully",
    "intent_detected": "Intent detected successfully",
    "search_completed": "Web search completed",
    "image_generated": "Image generated successfully"
}

# API Endpoints
API_ENDPOINTS = {
    "health": "/health",
    "chat": "/api/v1/chat",
    "upload_audio": "/api/v1/upload-audio",
    "process_text": "/api/v1/process-text"
}

# File Size Limits (in bytes)
FILE_SIZE_LIMITS = {
    "audio": 10 * 1024 * 1024,  # 10MB
    "image": 5 * 1024 * 1024,  # 5MB
}

# Timeout Settings (in seconds)
TIMEOUT_SETTINGS = {
    "openai_api": 30,
    "gemini_api": 45,
    "serp_api": 15,
    "whisper_processing": 60,
    "deepspeech_processing": 45
}

# Cache Keys
CACHE_KEYS = {
    "intent_detection": "intent_detection_{user_id}_{text_hash}",
    "web_search": "web_search_{query_hash}",
    "image_generation": "image_generation_{prompt_hash}"
}

# Log Format
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

# Audio Configuration
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "wav"
}

# Model Availability
MODEL_AVAILABILITY = {
    ModelType.WHISPER: True,
    ModelType.DEEPSPEECH: False,  # Set to True when implemented
    ModelType.OPENAI: True,
    ModelType.LOCAL_TRANSFORMER: False,  # Set to True when implemented
    ModelType.GEMINI: True,
    ModelType.SERP: True
}