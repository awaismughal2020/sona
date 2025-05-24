"""
Audio processing utilities for SONA AI Assistant.
Handles audio file validation, conversion, and preprocessing.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from typing import Union, Tuple, Optional
from loguru import logger
import tempfile

from config.settings import get_settings
from utils.constants import AudioFormat, FILE_SIZE_LIMITS


class AudioProcessor:
    """Audio processing utilities."""

    def __init__(self):
        """Initialize audio processor."""
        self.settings = get_settings()
        self.target_sample_rate = self.settings.audio_sample_rate
        self.target_channels = self.settings.audio_channels

    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate audio file format and properties.

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
            allowed_formats = self.settings.get_allowed_audio_formats()

            if file_ext not in allowed_formats:
                logger.error(f"Unsupported audio format: {file_ext}")
                return False

            # Try to load audio file
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=None, duration=1.0)
                if len(audio_data) == 0:
                    logger.error("Audio file is empty")
                    return False

                logger.info(f"Audio file validated: {file_ext}, {sample_rate}Hz, {len(audio_data)} samples")
                return True

            except Exception as e:
                logger.error(f"Failed to load audio file: {e}")
                return False

        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False

    def preprocess_audio(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess audio file for speech recognition.

        Args:
            file_path: Input audio file path
            output_path: Output file path (optional)

        Returns:
            Path to preprocessed audio file
        """
        try:
            # Load audio file
            audio_data, original_sr = librosa.load(file_path, sr=None)

            # Resample if necessary
            if original_sr != self.target_sample_rate:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=original_sr,
                    target_sr=self.target_sample_rate
                )
                logger.info(f"Resampled audio: {original_sr}Hz -> {self.target_sample_rate}Hz")

            # Convert to mono if necessary
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
                logger.info("Converted audio to mono")

            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)

            # Remove silence from beginning and end
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

            # Create output path if not provided
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".wav")

            # Save preprocessed audio
            sf.write(
                output_path,
                audio_data,
                self.target_sample_rate,
                format='WAV',
                subtype='PCM_16'
            )

            logger.info(f"Audio preprocessed and saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise RuntimeError(f"Audio preprocessing failed: {e}")

    def convert_bytes_to_audio(self, audio_bytes: bytes, format_hint: str = "wav") -> str:
        """
        Convert audio bytes to temporary audio file.

        Args:
            audio_bytes: Raw audio data
            format_hint: Audio format hint

        Returns:
            Path to temporary audio file
        """
        try:
            # Create temporary file
            suffix = f".{format_hint.lower()}"
            temp_file = tempfile.mktemp(suffix=suffix)

            # Write bytes to file
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)

            # Validate and preprocess
            if self.validate_audio_file(temp_file):
                processed_file = self.preprocess_audio(temp_file)

                # Clean up original temp file
                os.unlink(temp_file)

                return processed_file
            else:
                raise ValueError("Invalid audio data")

        except Exception as e:
            logger.error(f"Failed to convert audio bytes: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")

    def get_audio_info(self, file_path: str) -> dict:
        """
        Get audio file information.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio information
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None)

            duration = len(audio_data) / sample_rate

            return {
                "file_path": file_path,
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[0],
                "samples": len(audio_data),
                "format": os.path.splitext(file_path)[1].lower().lstrip('.'),
                "file_size": os.path.getsize(file_path)
            }

        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}
