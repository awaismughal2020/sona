"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Conversation Context Manager for SONA AI Assistant.
Maintains context across conversation turns for better multi-turn interactions.
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger
import asyncio
from collections import deque

# Import ContextType from constants to avoid circular imports
from utils.constants import ContextType


@dataclass
class ContextItem:
    """Individual context item."""
    type: ContextType
    key: str
    value: Any
    confidence: float
    timestamp: float
    expires_at: Optional[float] = None
    source: str = "system"

    def is_expired(self) -> bool:
        """Check if context item has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary."""
        return cls(
            type=ContextType(data["type"]),
            key=data["key"],
            value=data["value"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            expires_at=data.get("expires_at"),
            source=data.get("source", "system")
        )


@dataclass
class ConversationTurn:
    """Individual conversation turn."""
    turn_id: str
    user_input: str
    assistant_response: str
    intent: str
    confidence: float
    context_used: List[str]
    context_generated: List[str]
    timestamp: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(**data)


class ConversationContext:
    """
    Manages conversation context for a single session.
    Maintains context items, conversation history, and provides context-aware responses.
    """

    def __init__(self, session_id: str, max_history: int = 50, max_context_items: int = 100):
        """
        Initialize conversation context.

        Args:
            session_id: Unique session identifier
            max_history: Maximum number of conversation turns to keep
            max_context_items: Maximum number of context items to maintain
        """
        self.session_id = session_id
        self.max_history = max_history
        self.max_context_items = max_context_items

        # Context storage
        self.context_items: Dict[str, ContextItem] = {}
        self.conversation_history: deque = deque(maxlen=max_history)

        # Metadata
        self.created_at = time.time()
        self.last_activity = time.time()
        self.turn_count = 0

        # Current conversation state
        self.current_topic: Optional[str] = None
        self.current_task: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}

        logger.info(f"Created conversation context for session: {session_id}")

    def add_context_item(self,
                        context_type: ContextType,
                        key: str,
                        value: Any,
                        confidence: float = 1.0,
                        ttl_minutes: Optional[int] = None,
                        source: str = "system") -> None:
        """
        Add or update a context item.

        Args:
            context_type: Type of context
            key: Context key
            value: Context value
            confidence: Confidence score (0.0 to 1.0)
            ttl_minutes: Time to live in minutes
            source: Source of the context item
        """
        try:
            expires_at = None
            if ttl_minutes:
                expires_at = time.time() + (ttl_minutes * 60)

            context_item = ContextItem(
                type=context_type,
                key=key,
                value=value,
                confidence=confidence,
                timestamp=time.time(),
                expires_at=expires_at,
                source=source
            )

            item_id = f"{context_type.value}:{key}"
            self.context_items[item_id] = context_item

            # Maintain size limit
            if len(self.context_items) > self.max_context_items:
                self._cleanup_old_context_items()

            # Update specific tracking based on type
            if context_type == ContextType.TOPIC:
                self.current_topic = value
            elif context_type == ContextType.TASK:
                self.current_task = value
            elif context_type == ContextType.USER_PREFERENCE:
                self.user_preferences[key] = value

            logger.debug(f"Added context item: {item_id} = {value}")

        except Exception as e:
            logger.error(f"Failed to add context item: {e}")

    def get_context_item(self, context_type: ContextType, key: str) -> Optional[ContextItem]:
        """
        Get a specific context item.

        Args:
            context_type: Type of context
            key: Context key

        Returns:
            Context item or None if not found
        """
        try:
            item_id = f"{context_type.value}:{key}"
            item = self.context_items.get(item_id)

            if item and item.is_expired():
                del self.context_items[item_id]
                return None

            return item

        except Exception as e:
            logger.error(f"Failed to get context item: {e}")
            return None

    def get_context_by_type(self, context_type: ContextType) -> List[ContextItem]:
        """
        Get all context items of a specific type.

        Args:
            context_type: Type of context to retrieve

        Returns:
            List of context items
        """
        try:
            items = []
            prefix = f"{context_type.value}:"

            for item_id, item in list(self.context_items.items()):
                if item_id.startswith(prefix):
                    if item.is_expired():
                        del self.context_items[item_id]
                        continue
                    items.append(item)

            return sorted(items, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Failed to get context by type: {e}")
            return []

    def add_conversation_turn(self,
                            user_input: str,
                            assistant_response: str,
                            intent: str,
                            confidence: float,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new conversation turn.

        Args:
            user_input: User's input message
            assistant_response: Assistant's response
            intent: Detected intent
            confidence: Intent confidence
            metadata: Additional metadata

        Returns:
            Turn ID
        """
        try:
            self.turn_count += 1
            turn_id = f"turn_{self.turn_count}_{int(time.time())}"

            # Extract context used and generated
            context_used = self._extract_context_used(user_input, intent)
            context_generated = self._extract_context_generated(assistant_response, intent, metadata or {})

            turn = ConversationTurn(
                turn_id=turn_id,
                user_input=user_input,
                assistant_response=assistant_response,
                intent=intent,
                confidence=confidence,
                context_used=context_used,
                context_generated=context_generated,
                timestamp=time.time(),
                metadata=metadata or {}
            )

            self.conversation_history.append(turn)
            self.last_activity = time.time()

            # Auto-generate context from this turn
            self._auto_generate_context(turn)

            logger.info(f"Added conversation turn: {turn_id}")
            return turn_id

        except Exception as e:
            logger.error(f"Failed to add conversation turn: {e}")
            return ""

    def get_recent_context_for_prompt(self, max_turns: int = 5) -> str:
        """
        Generate context string for AI prompt based on recent conversation.

        Args:
            max_turns: Maximum number of recent turns to include

        Returns:
            Formatted context string
        """
        try:
            context_parts = []

            # Add user preferences
            if self.user_preferences:
                prefs = []
                for key, value in self.user_preferences.items():
                    prefs.append(f"{key}: {value}")
                context_parts.append(f"User preferences: {', '.join(prefs)}")

            # Add current topic/task
            if self.current_topic:
                context_parts.append(f"Current topic: {self.current_topic}")
            if self.current_task:
                context_parts.append(f"Current task: {self.current_task}")

            # Add recent conversation history
            recent_turns = list(self.conversation_history)[-max_turns:]
            if recent_turns:
                history_parts = []
                for turn in recent_turns:
                    history_parts.append(f"User: {turn.user_input}")
                    history_parts.append(f"Assistant: {turn.assistant_response}")

                context_parts.append(f"Recent conversation:\n{chr(10).join(history_parts)}")

            # Add relevant entities
            entities = self.get_context_by_type(ContextType.ENTITY)
            if entities:
                entity_parts = []
                for entity in entities[:5]:  # Top 5 most recent entities
                    entity_parts.append(f"{entity.key}: {entity.value}")
                context_parts.append(f"Relevant entities: {', '.join(entity_parts)}")

            return "\n\n".join(context_parts) if context_parts else ""

        except Exception as e:
            logger.error(f"Failed to generate context for prompt: {e}")
            return ""

    def get_contextual_search_suggestions(self) -> List[str]:
        """Get search suggestions based on conversation context."""
        try:
            suggestions = []

            # From current topic
            if self.current_topic:
                suggestions.append(f"More about {self.current_topic}")

            # From recent searches
            search_history = self.get_context_by_type(ContextType.SEARCH_HISTORY)
            for search in search_history[:3]:
                suggestions.append(f"Related to {search.value}")

            # From entities
            entities = self.get_context_by_type(ContextType.ENTITY)
            for entity in entities[:2]:
                suggestions.append(f"Information about {entity.value}")

            return suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []

    def _extract_context_used(self, user_input: str, intent: str) -> List[str]:
        """Extract context items that were likely used for this input."""
        used_context = []

        # Check if current topic/task was referenced
        user_lower = user_input.lower()

        if self.current_topic and self.current_topic.lower() in user_lower:
            used_context.append(f"topic:{self.current_topic}")

        if self.current_task and self.current_task.lower() in user_lower:
            used_context.append(f"task:{self.current_task}")

        # Check entities
        entities = self.get_context_by_type(ContextType.ENTITY)
        for entity in entities:
            if str(entity.value).lower() in user_lower:
                used_context.append(f"entity:{entity.key}")

        return used_context

    def _extract_context_generated(self, response: str, intent: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract context that should be generated from this response."""
        generated_context = []

        # Based on intent
        if intent == "web_search":
            if metadata.get("data"):
                generated_context.append("search_results")
        elif intent == "image_generation":
            generated_context.append("image_request")

        return generated_context

    def _auto_generate_context(self, turn: ConversationTurn) -> None:
        """Automatically generate context from conversation turn."""
        try:
            # Extract topics from user input
            self._extract_topics(turn.user_input)

            # Extract entities
            self._extract_entities(turn.user_input)

            # Store search queries
            if turn.intent == "web_search":
                self.add_context_item(
                    ContextType.SEARCH_HISTORY,
                    f"query_{int(time.time())}",
                    turn.user_input,
                    confidence=0.8,
                    ttl_minutes=60
                )

            # Store image generation requests
            elif turn.intent == "image_generation":
                self.add_context_item(
                    ContextType.IMAGE_HISTORY,
                    f"prompt_{int(time.time())}",
                    turn.user_input,
                    confidence=0.8,
                    ttl_minutes=120
                )

        except Exception as e:
            logger.error(f"Failed to auto-generate context: {e}")

    def _extract_topics(self, text: str) -> None:
        """Extract potential topics from text."""
        try:
            # Simple topic extraction (can be enhanced with NLP)
            text_lower = text.lower()

            # Common topic indicators
            topic_keywords = [
                "about", "regarding", "concerning", "discuss", "talk about",
                "learn about", "tell me about", "explain", "information on"
            ]

            for keyword in topic_keywords:
                if keyword in text_lower:
                    # Extract potential topic after keyword
                    start = text_lower.find(keyword) + len(keyword)
                    potential_topic = text[start:start+50].strip()

                    if potential_topic:
                        self.add_context_item(
                            ContextType.TOPIC,
                            "current_topic",
                            potential_topic,
                            confidence=0.7,
                            ttl_minutes=30
                        )
                        break

        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")

    def _extract_entities(self, text: str) -> None:
        """Extract named entities from text."""
        try:
            # Simple entity extraction (can be enhanced with NLP)
            import re

            # Extract potential entities (capitalized words/phrases)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

            for entity in entities[:3]:  # Limit to 3 entities per turn
                if len(entity.split()) <= 3:  # Only short entities
                    self.add_context_item(
                        ContextType.ENTITY,
                        entity.lower().replace(' ', '_'),
                        entity,
                        confidence=0.6,
                        ttl_minutes=45
                    )

        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")

    def _cleanup_old_context_items(self) -> None:
        """Clean up expired and old context items."""
        try:
            current_time = time.time()
            items_to_remove = []

            for item_id, item in self.context_items.items():
                if item.is_expired():
                    items_to_remove.append(item_id)

            # Remove expired items
            for item_id in items_to_remove:
                del self.context_items[item_id]

            # If still too many, remove oldest low-confidence items
            if len(self.context_items) > self.max_context_items:
                sorted_items = sorted(
                    self.context_items.items(),
                    key=lambda x: (x[1].confidence, x[1].timestamp)
                )

                excess_count = len(self.context_items) - self.max_context_items
                for i in range(excess_count):
                    item_id = sorted_items[i][0]
                    del self.context_items[item_id]

            logger.debug(f"Cleaned up context items: {len(items_to_remove)} expired, {len(self.context_items)} remaining")

        except Exception as e:
            logger.error(f"Failed to cleanup context items: {e}")

    def serialize(self) -> Dict[str, Any]:
        """Serialize context to dictionary for storage."""
        try:
            return {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "last_activity": self.last_activity,
                "turn_count": self.turn_count,
                "current_topic": self.current_topic,
                "current_task": self.current_task,
                "user_preferences": self.user_preferences,
                "context_items": {k: v.to_dict() for k, v in self.context_items.items()},
                "conversation_history": [turn.to_dict() for turn in self.conversation_history]
            }

        except Exception as e:
            logger.error(f"Failed to serialize context: {e}")
            return {}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Deserialize context from dictionary."""
        try:
            context = cls(data["session_id"])
            context.created_at = data["created_at"]
            context.last_activity = data["last_activity"]
            context.turn_count = data["turn_count"]
            context.current_topic = data.get("current_topic")
            context.current_task = data.get("current_task")
            context.user_preferences = data.get("user_preferences", {})

            # Restore context items
            for item_id, item_data in data.get("context_items", {}).items():
                context.context_items[item_id] = ContextItem.from_dict(item_data)

            # Restore conversation history
            for turn_data in data.get("conversation_history", []):
                turn = ConversationTurn.from_dict(turn_data)
                context.conversation_history.append(turn)

            return context

        except Exception as e:
            logger.error(f"Failed to deserialize context: {e}")
            return cls(data.get("session_id", "unknown"))

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "turn_count": self.turn_count,
            "context_items_count": len(self.context_items),
            "conversation_history_length": len(self.conversation_history),
            "current_topic": self.current_topic,
            "current_task": self.current_task,
            "user_preferences_count": len(self.user_preferences)
        }
