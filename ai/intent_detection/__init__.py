"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Intent detection services package.
"""

from .openai_service import OpenAIIntentService
from .base import IntentDetectionBase

__all__ = ["OpenAIIntentService", "IntentDetectionBase"]
