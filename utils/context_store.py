"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Updated Context Storage Manager for SONA AI Assistant.
Supports memory-persistent storage that keeps contexts until explicit deletion or container restart.

REPLACE YOUR EXISTING utils/context_store.py WITH THIS FILE.
"""

import json
import os
import time
import asyncio
from typing import Dict, Optional, List, Any
from loguru import logger
import threading
from pathlib import Path


class MemoryPersistentContextStore:
    """
    Memory-persistent context store that keeps conversations until explicit deletion or container restart.
    No automatic cleanup, no file persistence - pure memory storage.
    """

    def __init__(self, max_memory_contexts: int = 2000):
        """
        Initialize memory-persistent context store.

        Args:
            max_memory_contexts: Maximum contexts to keep in memory
        """
        self.max_memory_contexts = max_memory_contexts

        # Memory storage only
        self.memory_contexts: Dict[str, Any] = {}
        self.access_order: List[str] = []  # For tracking, not LRU eviction
        self.session_metadata: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"Initialized memory-persistent context store (max: {max_memory_contexts})")

    def get_context(self, session_id: str) -> Optional[Any]:
        """Get conversation context for session."""
        try:
            with self.lock:
                if session_id in self.memory_contexts:
                    self._update_access_order(session_id)
                    self._update_session_metadata(session_id, "last_accessed", time.time())
                    return self.memory_contexts[session_id]
                return None
        except Exception as e:
            logger.error(f"Failed to get context for session {session_id}: {e}")
            return None

    def create_context(self, session_id: str) -> Any:
        """Create new conversation context."""
        try:
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
                self._create_session_metadata(session_id)

                logger.info(f"Created new memory-persistent context for session: {session_id}")
                return context

        except Exception as e:
            logger.error(f"Failed to create context for session {session_id}: {e}")
            from utils.conversation_context import ConversationContext
            return ConversationContext(session_id)

    def save_context(self, session_id: str, context: Any) -> bool:
        """Save conversation context to memory."""
        try:
            with self.lock:
                self.memory_contexts[session_id] = context
                self._update_access_order(session_id)
                self._update_session_metadata(session_id, "last_updated", time.time())
                return True
        except Exception as e:
            logger.error(f"Failed to save context for session {session_id}: {e}")
            return False

    def delete_context(self, session_id: str) -> bool:
        """Delete conversation context from memory."""
        try:
            with self.lock:
                if session_id in self.memory_contexts:
                    del self.memory_contexts[session_id]
                if session_id in self.access_order:
                    self.access_order.remove(session_id)
                if session_id in self.session_metadata:
                    del self.session_metadata[session_id]

                logger.info(f"Deleted context from memory for session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete context for session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[str]:
        """List all session IDs with stored contexts."""
        try:
            with self.lock:
                return list(self.memory_contexts.keys())
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def get_context_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get context statistics for session."""
        try:
            context = self.get_context(session_id)
            if context:
                stats = context.get_stats()
                if session_id in self.session_metadata:
                    stats.update(self.session_metadata[session_id])
                return stats
            return None
        except Exception as e:
            logger.error(f"Failed to get context stats for session {session_id}: {e}")
            return None

    def get_all_sessions_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active sessions."""
        try:
            with self.lock:
                all_stats = {}
                for session_id in self.memory_contexts.keys():
                    stats = self.get_context_stats(session_id)
                    if stats:
                        all_stats[session_id] = stats
                return all_stats
        except Exception as e:
            logger.error(f"Failed to get all sessions stats: {e}")
            return {}

    def cleanup_expired_contexts(self, max_age_hours: int = 24) -> int:
        """
        Manual cleanup method - does NOT automatically run.
        Returns number of contexts that WOULD be cleaned up (for info only).
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            would_clean = 0

            with self.lock:
                for session_id, metadata in self.session_metadata.items():
                    last_activity = metadata.get("last_accessed", metadata.get("created_at", 0))
                    age = current_time - last_activity
                    if age > max_age_seconds:
                        would_clean += 1

            logger.info(f"Found {would_clean} contexts older than {max_age_hours} hours (not cleaned automatically)")
            return would_clean
        except Exception as e:
            logger.error(f"Failed to check expired contexts: {e}")
            return 0

    def manual_cleanup_old_sessions(self, older_than_hours: int = 24) -> int:
        """Manually cleanup sessions older than specified hours."""
        try:
            old_sessions = self.get_sessions_by_age(older_than_hours)
            cleaned_count = 0

            for session_id in old_sessions:
                if self.delete_context(session_id):
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Manually cleaned up {cleaned_count} old sessions")
            return cleaned_count
        except Exception as e:
            logger.error(f"Failed to manually cleanup old sessions: {e}")
            return 0

    def get_sessions_by_age(self, older_than_hours: int = 1) -> List[str]:
        """Get sessions older than specified hours."""
        try:
            current_time = time.time()
            age_threshold = older_than_hours * 3600
            old_sessions = []

            with self.lock:
                for session_id, metadata in self.session_metadata.items():
                    last_activity = metadata.get("last_accessed", metadata.get("created_at", 0))
                    age = current_time - last_activity
                    if age > age_threshold:
                        old_sessions.append(session_id)

            return old_sessions
        except Exception as e:
            logger.error(f"Failed to get sessions by age: {e}")
            return []

    def clear_all_contexts(self) -> int:
        """Clear all contexts from memory."""
        try:
            with self.lock:
                count = len(self.memory_contexts)
                self.memory_contexts.clear()
                self.access_order.clear()
                self.session_metadata.clear()
                logger.info(f"Cleared all {count} contexts from memory")
                return count
        except Exception as e:
            logger.error(f"Failed to clear all contexts: {e}")
            return 0

    def _add_to_memory(self, session_id: str, context: Any) -> None:
        """Add context to memory with soft limit warnings."""
        try:
            self.memory_contexts[session_id] = context
            self._update_access_order(session_id)

            # Warn when approaching limit
            if len(self.memory_contexts) > self.max_memory_contexts * 0.9:
                logger.warning(f"Memory usage high: {len(self.memory_contexts)}/{self.max_memory_contexts}")

            # Emergency protection (only if way over limit)
            if len(self.memory_contexts) > self.max_memory_contexts * 1.5:
                logger.error("Emergency cleanup triggered - memory usage too high")
                sessions_to_remove = int(len(self.memory_contexts) * 0.1)
                oldest_sessions = self.access_order[:sessions_to_remove]
                for old_session in oldest_sessions:
                    self.delete_context(old_session)
                    logger.warning(f"Emergency removed session: {old_session}")

        except Exception as e:
            logger.error(f"Failed to add context to memory: {e}")

    def _update_access_order(self, session_id: str) -> None:
        """Update access order for tracking."""
        try:
            if session_id in self.access_order:
                self.access_order.remove(session_id)
            self.access_order.append(session_id)
        except Exception as e:
            logger.error(f"Failed to update access order: {e}")

    def _create_session_metadata(self, session_id: str) -> None:
        """Create metadata for a new session."""
        try:
            current_time = time.time()
            self.session_metadata[session_id] = {
                "created_at": current_time,
                "last_accessed": current_time,
                "last_updated": current_time,
                "access_count": 1
            }
        except Exception as e:
            logger.error(f"Failed to create session metadata: {e}")

    def _update_session_metadata(self, session_id: str, key: str, value: Any) -> None:
        """Update session metadata."""
        try:
            if session_id not in self.session_metadata:
                self._create_session_metadata(session_id)

            self.session_metadata[session_id][key] = value

            if key == "last_accessed":
                self.session_metadata[session_id]["access_count"] = self.session_metadata[session_id].get("access_count", 0) + 1

        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            with self.lock:
                total_sessions = len(self.memory_contexts)
                memory_usage_percent = (total_sessions / self.max_memory_contexts) * 100

                # Calculate session ages
                current_time = time.time()
                session_ages = []

                for session_id, metadata in self.session_metadata.items():
                    created_at = metadata.get("created_at", current_time)
                    age_hours = (current_time - created_at) / 3600
                    session_ages.append(age_hours)

                avg_age = sum(session_ages) / len(session_ages) if session_ages else 0
                oldest_age = max(session_ages) if session_ages else 0

                return {
                    "contexts_in_memory": total_sessions,
                    "max_memory_contexts": self.max_memory_contexts,
                    "memory_usage_percent": memory_usage_percent,
                    "access_order_length": len(self.access_order),
                    "storage_type": "memory_persistent",
                    "average_session_age_hours": round(avg_age, 2),
                    "oldest_session_age_hours": round(oldest_age, 2),
                    "warning_threshold_reached": memory_usage_percent > 90,
                    "emergency_threshold_reached": memory_usage_percent > 150
                }

        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}

    def export_context(self, session_id: str, format: str = 'json') -> Optional[Dict[str, Any]]:
        """Export context in specified format."""
        try:
            context = self.get_context(session_id)
            if not context:
                return None

            if format == 'json':
                exported = context.serialize()
                if session_id in self.session_metadata:
                    exported["session_metadata"] = self.session_metadata[session_id]
                return exported

            elif format == 'summary':
                from utils.constants import ContextType
                return {
                    "session_id": session_id,
                    "stats": context.get_stats(),
                    "session_metadata": self.session_metadata.get(session_id, {}),
                    "recent_topics": [item.value for item in context.get_context_by_type(ContextType.TOPIC)[:5]],
                    "user_preferences": context.user_preferences,
                    "recent_searches": [item.value for item in context.get_context_by_type(ContextType.SEARCH_HISTORY)[:3]]
                }
            else:
                logger.error(f"Unsupported export format: {format}")
                return None

        except Exception as e:
            logger.error(f"Failed to export context for session {session_id}: {e}")
            return None

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information about the context store."""
        return {
            "store_type": "MemoryPersistentContextStore",
            "persistence_model": "memory_only_until_container_restart",
            "automatic_cleanup": False,
            "file_persistence": False,
            "max_contexts": self.max_memory_contexts,
            "features": [
                "Memory-only storage",
                "No automatic cleanup",
                "Persists until explicit deletion or container restart",
                "Manual cleanup methods available",
                "Session metadata tracking"
            ]
        }


# For backward compatibility - original ContextStore interface
class ContextStore(MemoryPersistentContextStore):
    """
    Backward compatibility wrapper that implements the original ContextStore interface
    but uses memory-persistent storage underneath.
    """

    def __init__(self, storage_dir: str = "contexts", max_memory_contexts: int = 2000):
        """Initialize with backward compatibility for original parameters."""
        # Ignore storage_dir since we're memory-persistent
        super().__init__(max_memory_contexts=max_memory_contexts)
        logger.info("Using memory-persistent context store (ignoring storage_dir parameter)")


# Global context store instance
_global_context_store: Optional[MemoryPersistentContextStore] = None


def get_context_store() -> MemoryPersistentContextStore:
    """Get global memory-persistent context store instance."""
    global _global_context_store
    if _global_context_store is None:
        _global_context_store = MemoryPersistentContextStore()
    return _global_context_store


def initialize_context_store(storage_dir: str = "contexts", max_memory_contexts: int = 2000) -> MemoryPersistentContextStore:
    """Initialize global context store with memory-persistent settings."""
    global _global_context_store
    # Ignore storage_dir parameter for memory-persistent mode
    _global_context_store = MemoryPersistentContextStore(max_memory_contexts)
    logger.info(f"Initialized memory-persistent context store (ignoring storage_dir: {storage_dir})")
    return _global_context_store
