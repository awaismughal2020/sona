"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Context Storage Manager for SONA AI Assistant.
Handles persistent storage and retrieval of conversation contexts.
"""

import json
import os
import time
import asyncio
from typing import Dict, Optional, List, Any
from loguru import logger
import threading
from pathlib import Path


class ContextStore:
    """
    Manages persistent storage of conversation contexts.
    Supports both memory and file-based storage.
    """

    def __init__(self, storage_dir: str = "contexts", max_memory_contexts: int = 100):
        """
        Initialize context store.

        Args:
            storage_dir: Directory to store context files
            max_memory_contexts: Maximum contexts to keep in memory
        """
        self.storage_dir = Path(storage_dir)
        self.max_memory_contexts = max_memory_contexts

        # In-memory cache - we'll import ConversationContext when needed
        self.memory_contexts: Dict[str, Any] = {}  # Changed from ConversationContext to Any
        self.access_order: List[str] = []  # For LRU eviction

        # Thread safety
        self.lock = threading.RLock()

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Background cleanup task
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()

        logger.info(f"Initialized context store with storage: {self.storage_dir}")

    def get_context(self, session_id: str) -> Optional[Any]:  # Changed return type
        """
        Get conversation context for session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationContext or None if not found
        """
        try:
            with self.lock:
                # Check memory cache first
                if session_id in self.memory_contexts:
                    self._update_access_order(session_id)
                    return self.memory_contexts[session_id]

                # Try to load from disk
                context = self._load_from_disk(session_id)
                if context:
                    self._add_to_memory(session_id, context)
                    return context

                return None

        except Exception as e:
            logger.error(f"Failed to get context for session {session_id}: {e}")
            return None

    def create_context(self, session_id: str) -> Any:  # Changed return type
        """
        Create new conversation context.

        Args:
            session_id: Session identifier

        Returns:
            New ConversationContext
        """
        try:
            # Import here to avoid circular import
            from utils.conversation_context import ConversationContext

            with self.lock:
                # Check if context already exists
                existing_context = self.get_context(session_id)
                if existing_context:
                    logger.warning(f"Context already exists for session {session_id}")
                    return existing_context

                # Create new context
                context = ConversationContext(session_id)
                self._add_to_memory(session_id, context)

                # Save to disk
                self._save_to_disk(session_id, context)

                logger.info(f"Created new context for session: {session_id}")
                return context

        except Exception as e:
            logger.error(f"Failed to create context for session {session_id}: {e}")
            # Return basic context even if storage fails
            from utils.conversation_context import ConversationContext
            return ConversationContext(session_id)

    def save_context(self, session_id: str, context: Any) -> bool:  # Changed parameter type
        """
        Save conversation context.

        Args:
            session_id: Session identifier
            context: Context to save

        Returns:
            True if successful
        """
        try:
            with self.lock:
                # Update memory cache
                self.memory_contexts[session_id] = context
                self._update_access_order(session_id)

                # Save to disk
                return self._save_to_disk(session_id, context)

        except Exception as e:
            logger.error(f"Failed to save context for session {session_id}: {e}")
            return False

    def delete_context(self, session_id: str) -> bool:
        """
        Delete conversation context.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            with self.lock:
                # Remove from memory
                if session_id in self.memory_contexts:
                    del self.memory_contexts[session_id]

                if session_id in self.access_order:
                    self.access_order.remove(session_id)

                # Remove from disk
                context_file = self._get_context_file_path(session_id)
                if context_file.exists():
                    context_file.unlink()

                logger.info(f"Deleted context for session: {session_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete context for session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[str]:
        """
        List all session IDs with stored contexts.

        Returns:
            List of session identifiers
        """
        try:
            sessions = set()

            # From memory
            sessions.update(self.memory_contexts.keys())

            # From disk
            for file_path in self.storage_dir.glob("context_*.json"):
                session_id = file_path.stem.replace("context_", "")
                sessions.add(session_id)

            return list(sessions)

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def get_context_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get context statistics for session.

        Args:
            session_id: Session identifier

        Returns:
            Statistics dictionary or None
        """
        try:
            context = self.get_context(session_id)
            if context:
                return context.get_stats()
            return None

        except Exception as e:
            logger.error(f"Failed to get context stats for session {session_id}: {e}")
            return None

    def cleanup_expired_contexts(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired contexts.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of contexts cleaned up
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0

            with self.lock:
                # Check memory contexts
                sessions_to_remove = []
                for session_id, context in self.memory_contexts.items():
                    age = current_time - context.last_activity
                    if age > max_age_seconds:
                        sessions_to_remove.append(session_id)

                # Remove expired contexts from memory
                for session_id in sessions_to_remove:
                    del self.memory_contexts[session_id]
                    if session_id in self.access_order:
                        self.access_order.remove(session_id)
                    cleaned_count += 1

                # Check disk contexts
                for file_path in self.storage_dir.glob("context_*.json"):
                    try:
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_path.unlink()
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to check/remove context file {file_path}: {e}")

            logger.info(f"Cleaned up {cleaned_count} expired contexts")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired contexts: {e}")
            return 0

    def _add_to_memory(self, session_id: str, context: Any) -> None:  # Changed parameter type
        """Add context to memory cache with LRU eviction."""
        try:
            # Add/update context
            self.memory_contexts[session_id] = context
            self._update_access_order(session_id)

            # Evict if over limit
            if len(self.memory_contexts) > self.max_memory_contexts:
                # Remove least recently used
                oldest_session = self.access_order[0]

                # Save to disk before evicting
                self._save_to_disk(oldest_session, self.memory_contexts[oldest_session])

                # Remove from memory
                del self.memory_contexts[oldest_session]
                self.access_order.remove(oldest_session)

                logger.debug(f"Evicted context from memory: {oldest_session}")

        except Exception as e:
            logger.error(f"Failed to add context to memory: {e}")

    def _update_access_order(self, session_id: str) -> None:
        """Update access order for LRU tracking."""
        try:
            if session_id in self.access_order:
                self.access_order.remove(session_id)
            self.access_order.append(session_id)

        except Exception as e:
            logger.error(f"Failed to update access order: {e}")

    def _get_context_file_path(self, session_id: str) -> Path:
        """Get file path for context storage."""
        # Sanitize session ID for filename
        safe_session_id = "".join(c for c in session_id if c.isalnum() or c in "._-")
        return self.storage_dir / f"context_{safe_session_id}.json"

    def _save_to_disk(self, session_id: str, context: Any) -> bool:  # Changed parameter type
        """Save context to disk."""
        try:
            context_file = self._get_context_file_path(session_id)

            # Serialize context
            context_data = context.serialize()

            # Write to temporary file first, then rename (atomic operation)
            temp_file = context_file.with_suffix('.tmp')

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.rename(context_file)

            logger.debug(f"Saved context to disk: {context_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save context to disk for session {session_id}: {e}")
            return False

    def _load_from_disk(self, session_id: str) -> Optional[Any]:  # Changed return type
        """Load context from disk."""
        try:
            # Import here to avoid circular import
            from utils.conversation_context import ConversationContext

            context_file = self._get_context_file_path(session_id)

            if not context_file.exists():
                return None

            with open(context_file, 'r', encoding='utf-8') as f:
                context_data = json.load(f)

            context = ConversationContext.deserialize(context_data)

            logger.debug(f"Loaded context from disk: {context_file}")
            return context

        except Exception as e:
            logger.error(f"Failed to load context from disk for session {session_id}: {e}")
            return None

    async def periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)

                current_time = time.time()
                if current_time - self.last_cleanup > self.cleanup_interval:
                    self.cleanup_expired_contexts()
                    self.last_cleanup = current_time

        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            return {
                "contexts_in_memory": len(self.memory_contexts),
                "max_memory_contexts": self.max_memory_contexts,
                "memory_usage_percent": (len(self.memory_contexts) / self.max_memory_contexts) * 100,
                "access_order_length": len(self.access_order),
                "storage_directory": str(self.storage_dir)
            }

        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}

    def export_context(self, session_id: str, format: str = 'json') -> Optional[Dict[str, Any]]:
        """
        Export context in specified format.

        Args:
            session_id: Session identifier
            format: Export format ('json', 'summary')

        Returns:
            Exported data or None
        """
        try:
            context = self.get_context(session_id)
            if not context:
                return None

            if format == 'json':
                return context.serialize()
            elif format == 'summary':
                return {
                    "session_id": session_id,
                    "stats": context.get_stats(),
                    "recent_topics": [item.value for item in context.get_context_by_type(ContextType.TOPIC)[:5]],
                    "user_preferences": context.user_preferences,
                    "recent_searches": [item.value for item in
                                        context.get_context_by_type(ContextType.SEARCH_HISTORY)[:3]]
                }
            else:
                logger.error(f"Unsupported export format: {format}")
                return None

        except Exception as e:
            logger.error(f"Failed to export context for session {session_id}: {e}")
            return None


# Global context store instance
_global_context_store: Optional[ContextStore] = None


def get_context_store() -> ContextStore:
    """Get global context store instance."""
    global _global_context_store
    if _global_context_store is None:
        _global_context_store = ContextStore()
    return _global_context_store


def initialize_context_store(storage_dir: str = "contexts", max_memory_contexts: int = 100) -> ContextStore:
    """Initialize global context store with custom settings."""
    global _global_context_store
    _global_context_store = ContextStore(storage_dir, max_memory_contexts)
    return _global_context_store
