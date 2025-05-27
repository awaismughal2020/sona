"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Backend package for SONA AI Assistant with Memory-Persistent Context Management.
"""

from .app import SONABackendWithMemoryPersistentContext, SONABackend, app

# For backward compatibility, also export with original name
SONABackendWithContext = SONABackendWithMemoryPersistentContext

__all__ = ["SONABackendWithMemoryPersistentContext", "SONABackendWithContext", "SONABackend", "app"]
