"""
OpenAI Whisper speech-to-text service implementation.
Handles audio transcription using Whisper models.
"""
import whisper
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
    """OpenAI Whisper speech-to-text service implementation."""

    def __init__(self, **kwargs):
        """Initialize Whisper service."""
        settings = get_settings()
        super().__init__(
            model_name="whisper",
            model_size=settings.whisper_model_size,
            **kwargs
        )
        self.model = None
        self.supported_formats = ['wav', 'mp3', 'm4a', 'flac']

    async def initialize(self) -> None:
        """Initialize Whisper model."""
        try:
            logger.info(f"Initializing Whisper model: {self.config['model_size']}")

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                functools.partial(whisper.load_model, self.config['model_size'])
            )

            self.is_initialized = True
            logger.info("Whisper model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise RuntimeError(f"Whisper initialization failed: {e}")

    async def transcribe_audio(self, audio_data: Union[bytes, str]) -> str:
        """
        Transcribe audio data using Whisper.

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

                # Validate audio file
                if not validate_audio_file(audio_data):
                    raise ValueError("Invalid audio file format")

                return await self._transcribe_file(audio_data)

            else:
                raise ValueError("Audio data must be bytes or file path string")

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    async def _transcribe_file(self, file_path: str) -> str:
        """
        Transcribe audio file using Whisper model.

        Args:
            file_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio file: {file_path}")

            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    self.model.transcribe,
                    file_path,
                    fp16=False,  # Use fp32 for better compatibility
                    language="en"  # Can be made configurable
                )
            )

            transcribed_text = result["text"].strip()
            logger.info(f"Transcription completed: {len(transcribed_text)} characters")

            return transcribed_text

        except Exception as e:
            logger.error(f"Whisper file transcription failed: {e}")
            raise RuntimeError(f"File transcription failed: {e}")

    def is_available(self) -> bool:
        """
        Check if Whisper service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Check if whisper is importable and model can be loaded
            import whisper
            return True
        except ImportError:
            logger.warning("Whisper package not available")
            return False
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
        Get information about the loaded Whisper model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_size": self.config.get('model_size', 'base'),
            "supported_formats": self.supported_formats,
            "initialized": self.is_initialized,
            "language": "en"
        }
