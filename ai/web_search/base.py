"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Base class for web search services.
Provides interface for all web search implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from loguru import logger


class WebSearchBase(ABC):
    """Abstract base class for web search services."""

    def __init__(self, service_name: str, **kwargs):
        """
        Initialize web search service.

        Args:
            service_name: Name of the search service
            **kwargs: Additional configuration parameters
        """
        self.service_name = service_name
        self.config = kwargs
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the web search service."""
        pass

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search with given query.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of search results with title, snippet, url, etc.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available."""
        pass

    async def health_check(self) -> dict:
        """Perform health check on the service."""
        try:
            is_available = self.is_available()
            return {
                "service": f"web_search_{self.service_name}",
                "status": "healthy" if is_available else "unhealthy",
                "initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.service_name}: {e}")
            return {
                "service": f"web_search_{self.service_name}",
                "status": "unhealthy",
                "error": str(e)
            }
