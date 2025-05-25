"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

OpenAI Whisper speech-to-text service implementation.
Handles audio transcription using OpenAI's Whisper API (not local whisper).
"""
import openai
import tempfile
import os
from typing import Union
from loguru import logger
import asyncio
import functools

from .base import SpeechToTextBase
from config.settings import get_settings
from utils.validation import validate_audio_file
from utils.constants import ERROR_MESSAGES, SUCCESS_MESSAGES


class WhisperService(SpeechToTextBase):
    """OpenAI Whisper API speech-to-text service implementation."""

    def __init__(self, **kwargs):
        """Initialize Whisper service."""
        settings = get_settings()
        super().__init__(
            model_name="whisper",
            api_key=settings.openai_api_key,  # Use OpenAI API key
            model_size="whisper-1",  # OpenAI API model name
            **kwargs
        )
        self.client = None
        self.supported_formats = ['wav', 'mp3', 'm4a', 'flac', 'webm', 'mp4']

    async def initialize(self) -> None:
        """Initialize OpenAI Whisper API client."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("OpenAI API key is required for Whisper service")

            logger.info("Initializing OpenAI Whisper API service")

            # Initialize OpenAI client
            self.client = openai.AsyncOpenAI(api_key=self.config['api_key'])

            # Test the connection with a minimal request (we'll skip this for now as it requires audio)
            logger.info("OpenAI Whisper API client initialized")

            self.is_initialized = True
            logger.info("Whisper service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Whisper service: {e}")
            raise RuntimeError(f"Whisper initialization failed: {e}")

    async def transcribe_audio(self, audio_data: Union[bytes, str]) -> str:
        """
        Transcribe audio data using OpenAI Whisper API.

        Args:
            audio_data: Audio data as bytes or file path

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If transcription fails
            ValueError: If audio data is invalid
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Handle different input types
            if isinstance(audio_data, bytes):
                # Save bytes to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name

                try:
                    result = await self._transcribe_file(temp_file_path)
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

                return result

            elif isinstance(audio_data, str):
                # File path provided
                if not os.path.exists(audio_data):
                    raise ValueError(f"Audio file not found: {audio_data}")

                # Basic validation
                if not self._basic_audio_validation(audio_data):
                    raise ValueError("Invalid audio file format")

                return await self._transcribe_file(audio_data)

            else:
                raise ValueError("Audio data must be bytes or file path string")

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    async def _transcribe_file(self, file_path: str) -> str:
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            file_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio file: {file_path}")

            # Check file size (OpenAI has a 25MB limit)
            file_size = os.path.getsize(file_path)
            max_size = 25 * 1024 * 1024  # 25MB

            if file_size > max_size:
                raise ValueError(f"Audio file too large: {file_size} bytes (max: 25MB)")

            # Open and transcribe the audio file
            with open(file_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )

            # Handle different response formats
            if hasattr(transcript, 'text'):
                transcribed_text = transcript.text.strip()
            elif isinstance(transcript, str):
                transcribed_text = transcript.strip()
            else:
                transcribed_text = str(transcript).strip()

            logger.info(f"Transcription completed: {len(transcribed_text)} characters")

            if not transcribed_text:
                logger.warning("Transcription returned empty text")
                return "No speech detected in audio file"

            return transcribed_text

        except Exception as e:
            logger.error(f"Whisper API transcription failed: {e}")
            raise RuntimeError(f"File transcription failed: {e}")

    def _basic_audio_validation(self, file_path: str) -> bool:
        """
        Basic audio file validation.

        Args:
            file_path: Path to audio file

        Returns:
            True if basic validation passes
        """
        try:
            if not os.path.exists(file_path):
                return False

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # Too small
                logger.warning(f"Audio file too small: {file_size} bytes")
                return False

            if file_size > 25 * 1024 * 1024:  # OpenAI limit
                logger.warning(f"Audio file too large: {file_size} bytes")
                return False

            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if file_ext not in self.supported_formats:
                logger.warning(f"Unsupported audio format: {file_ext}")
                return False

            return True

        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False

    def is_available(self) -> bool:
        """
        Check if Whisper service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"Whisper availability check failed: {e}")
            return False

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported audio formats.

        Returns:
            List of supported audio file extensions
        """
        return self.supported_formats.copy()

    async def get_model_info(self) -> dict:
        """
        Get information about the Whisper model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_version": "whisper-1",
            "service": "OpenAI Whisper API",
            "supported_formats": self.supported_formats,
            "max_file_size": "25MB",
            "initialized": self.is_initialized,
            "available": self.is_available()
        }

    async def health_check(self) -> dict:
        """Perform health check on the service."""
        try:
            is_available = self.is_available()

            # Additional check for API key
            has_api_key = bool(self.config.get('api_key'))

            status = "healthy" if (is_available and has_api_key) else "unhealthy"

            result = {
                "service": f"speech_to_text_{self.model_name}",
                "status": status,
                "initialized": self.is_initialized,
                "has_api_key": has_api_key,
                "model": "whisper-1",
                "service_type": "OpenAI API"
            }

            if not has_api_key:
                result["error"] = "OpenAI API key not configured"

            return result

        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return {
                "service": f"speech_to_text_{self.model_name}",
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized
            }
