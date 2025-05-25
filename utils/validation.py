"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Validation utilities for SONA AI Assistant.
Handles input validation and data sanitization.
"""

import re
import os
from typing import Any, Dict, Optional, Union
from loguru import logger

from config.settings import get_settings
from utils.constants import AudioFormat, FILE_SIZE_LIMITS


def validate_audio_file(file_path: str) -> bool:
    """
    Validate audio file.

    Args:
        file_path: Path to audio file

    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > FILE_SIZE_LIMITS["audio"]:
            logger.error(f"Audio file too large: {file_size} bytes")
            return False

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        settings = get_settings()
        allowed_formats = settings.get_allowed_audio_formats()

        if file_ext not in allowed_formats:
            logger.error(f"Unsupported audio format: {file_ext}")
            return False

        # Try to load audio file (basic check)
        try:
            import librosa
            audio_data, sample_rate = librosa.load(file_path, sr=None,
                                                   duration=1.0)  # Load only 1 second for validation
            if len(audio_data) == 0:
                logger.error("Audio file is empty")
                return False

            logger.info(f"Audio file validated: {file_ext}, {sample_rate}Hz")
            return True

        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return False

    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        return False


def validate_text_input(text: str, max_length: int = 1000, min_length: int = 1) -> bool:
    """
    Validate text input.

    Args:
        text: Input text
        max_length: Maximum allowed length
        min_length: Minimum required length

    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(text, str):
            logger.warning(f"Text input is not a string: {type(text)}")
            return False

        # Check length constraints
        text_stripped = text.strip()
        if len(text_stripped) < min_length:
            logger.warning(f"Text too short: {len(text_stripped)} < {min_length}")
            return False

        if len(text) > max_length:
            logger.warning(f"Text too long: {len(text)} > {max_length}")
            return False

        # Check for basic content (not just whitespace/special chars)
        if not re.search(r'[a-zA-Z0-9]', text):
            logger.warning("Text contains no alphanumeric characters")
            return False

        # Check for potentially malicious content
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:text/html',  # Data URLs
            r'vbscript:',  # VBScript
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Suspicious content detected: {pattern}")
                return False

        return True

    except Exception as e:
        logger.error(f"Text validation failed: {e}")
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    try:
        if not filename:
            return "unknown_file"

        # Remove path components
        filename = os.path.basename(filename)

        # Replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)

        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)

        # Replace multiple underscores with single underscore
        filename = re.sub(r'_{2,}', '_', filename)

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Ensure reasonable length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            max_name_length = 255 - len(ext)
            filename = name[:max_name_length] + ext

        # Ensure filename is not empty after sanitization
        if not filename or filename == '_':
            return "sanitized_file"

        return filename

    except Exception as e:
        logger.error(f"Filename sanitization failed: {e}")
        return "unknown_file"


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate required API keys are present.

    Returns:
        Dictionary with validation results
    """
    try:
        settings = get_settings()
        return settings.validate_api_keys()
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return {"openai": False, "gemini": False, "serp": False}


def validate_model_configuration() -> Dict[str, bool]:
    """
    Validate model configuration is valid.

    Returns:
        Dictionary with validation results
    """
    try:
        settings = get_settings()

        valid_models = {
            "speech_to_text": settings.speech_to_text_model in ["whisper", "deepspeech"],
            "intent_detection": settings.intent_detection_model in ["openai", "local_transformer"],
            "image_generation": settings.image_generation_model in ["gemini"],
            "web_search": settings.web_search_model in ["serp"]
        }

        return valid_models

    except Exception as e:
        logger.error(f"Model configuration validation failed: {e}")
        return {
            "speech_to_text": False,
            "intent_detection": False,
            "image_generation": False,
            "web_search": False
        }


def validate_file_upload(file_data: bytes, filename: str, file_type: str) -> Dict[str, Any]:
    """
    Validate uploaded file.

    Args:
        file_data: File content as bytes
        filename: Original filename
        file_type: Expected file type ('audio', 'image', etc.)

    Returns:
        Validation result dictionary
    """
    try:
        result = {
            "valid": False,
            "error": None,
            "sanitized_filename": sanitize_filename(filename),
            "file_size": len(file_data)
        }

        # Check if file data is provided
        if not file_data:
            result["error"] = "No file data provided"
            return result

        # Check file size
        max_size = FILE_SIZE_LIMITS.get(file_type, FILE_SIZE_LIMITS["audio"])
        if len(file_data) > max_size:
            result["error"] = f"File too large. Maximum size: {max_size} bytes ({max_size / (1024 * 1024):.1f} MB)"
            return result

        # Minimum file size check
        if len(file_data) < 100:  # Less than 100 bytes is likely invalid
            result["error"] = "File too small or empty"
            return result

        # Check file extension
        if not filename:
            result["error"] = "Filename not provided"
            return result

        file_ext = os.path.splitext(filename)[1].lower().lstrip('.')

        if file_type == "audio":
            settings = get_settings()
            allowed_formats = settings.get_allowed_audio_formats()
            if file_ext not in allowed_formats:
                result["error"] = f"Unsupported audio format '{file_ext}'. Allowed: {', '.join(allowed_formats)}"
                return result

        # Basic file signature validation for audio files
        if file_type == "audio":
            if not _validate_audio_signature(file_data, file_ext):
                result["error"] = f"File does not appear to be a valid {file_ext} audio file"
                return result

        result["valid"] = True
        return result

    except Exception as e:
        logger.error(f"File upload validation failed: {e}")
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "sanitized_filename": sanitize_filename(filename) if filename else "unknown_file",
            "file_size": len(file_data) if file_data else 0
        }


def _validate_audio_signature(file_data: bytes, file_ext: str) -> bool:
    """
    Validate audio file signature (magic bytes).

    Args:
        file_data: File content as bytes
        file_ext: File extension

    Returns:
        True if signature matches expected format
    """
    try:
        if len(file_data) < 12:
            return False

        # Check magic bytes for common audio formats
        signatures = {
            'wav': [b'RIFF', b'WAVE'],
            'mp3': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
            'flac': [b'fLaC'],
            'm4a': [b'ftypM4A'],
        }

        if file_ext not in signatures:
            # If we don't have signature info, allow it through
            return True

        # Check if any of the signatures match
        for signature in signatures[file_ext]:
            if file_data.startswith(signature):
                return True

            # For some formats, check at different positions
            if file_ext == 'wav' and signature == b'WAVE':
                if signature in file_data[:20]:  # WAVE should be near the beginning
                    return True

        return False

    except Exception as e:
        logger.error(f"File signature validation failed: {e}")
        return True  # Allow through if signature check fails


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session identifier

    Returns:
        True if valid format
    """
    try:
        if not session_id or not isinstance(session_id, str):
            return False

        # Basic format validation
        if len(session_id) < 8 or len(session_id) > 64:
            return False

        # Allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            return False

        return True

    except Exception as e:
        logger.error(f"Session ID validation failed: {e}")
        return False


def validate_search_query(query: str) -> bool:
    """
    Validate web search query.

    Args:
        query: Search query string

    Returns:
        True if valid
    """
    try:
        # Use text validation with search-specific constraints
        if not validate_text_input(query, max_length=500, min_length=1):
            return False

        # Additional search-specific validations
        query_stripped = query.strip()

        # Check for too many special characters (potential spam)
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', query_stripped)) / len(query_stripped)
        if special_char_ratio > 0.5:  # More than 50% special characters
            logger.warning(f"Search query has too many special characters: {special_char_ratio:.2%}")
            return False

        return True

    except Exception as e:
        logger.error(f"Search query validation failed: {e}")
        return False


def validate_image_prompt(prompt: str) -> bool:
    """
    Validate image generation prompt.

    Args:
        prompt: Image generation prompt

    Returns:
        True if valid
    """
    try:
        # Use text validation with image-specific constraints
        if not validate_text_input(prompt, max_length=2000, min_length=3):
            return False

        # Check for inappropriate content keywords (basic filter)
        inappropriate_keywords = [
            'nude', 'naked', 'nsfw', 'adult', 'explicit',
            'violence', 'weapon', 'gore', 'blood'
        ]

        prompt_lower = prompt.lower()
        for keyword in inappropriate_keywords:
            if keyword in prompt_lower:
                logger.warning(f"Image prompt contains inappropriate keyword: {keyword}")
                return False

        return True

    except Exception as e:
        logger.error(f"Image prompt validation failed: {e}")
        return False


def sanitize_text_input(text: str) -> str:
    """
    Sanitize text input by removing/escaping potentially harmful content.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    try:
        if not isinstance(text, str):
            return ""

        # Remove control characters except common whitespace
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')

        # Remove potential HTML/script tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)

        # Remove potential JavaScript
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    except Exception as e:
        logger.error(f"Text sanitization failed: {e}")
        return text  # Return original if sanitization fails
