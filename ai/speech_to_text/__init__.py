"""
Speech-to-text services package.
"""

from .whisper_service import WhisperService
from .base import SpeechToTextBase

__all__ = ["WhisperService", "SpeechToTextBase"]
