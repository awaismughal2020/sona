"""
SERP API web search service implementation.
Handles web search functionality using SerpAPI.
"""

from serpapi import GoogleSearch
from typing import List, Dict, Any
import asyncio
from loguru import logger

from .base import WebSearchBase
from config.settings import get_settings
from utils.constants import ERROR_MESSAGES, SUCCESS_MESSAGES


class SerpSearchService(WebSearchBase):
    """SERP API web search service implementation."""

    def __init__(self, **kwargs):
        """Initialize SERP search service."""
        settings = get_settings()
        super().__init__(
            service_name="serp",
            api_key=settings.serp_api_key,
            **kwargs
        )
        self.search_engine = "google"  # Default to Google

    async def initialize(self) -> None:
        """Initialize SERP API service."""
        try:
            if not self.config.get('api_key'):
                raise ValueError("SERP API key is required")

            logger.info("Initializing SERP API service")

            # Test the API connection
            await self._test_connection()

            self.is_initialized = True
            logger.info("SERP API service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SERP service: {e}")
            raise RuntimeError(f"SERP initialization failed: {e}")

    async def _test_connection(self) -> None:
        """Test SERP API connection."""
        try:
            # Perform a simple test search
            search = GoogleSearch({
                "q": "test",
                "api_key": self.config['api_key'],
                "num": 1
            })

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                search.get_dict
            )

            if "error" in result:
                raise Exception(f"SERP API error: {result['error']}")

            logger.info("SERP API connection test successful")

        except Exception as e:
            logger.error(f"SERP API connection test failed: {e}")
            raise

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using SERP API.

        Args:
            query: Search query string
            num_results: Number of results to return (max 10)

        Returns:
            List of search results with title, snippet, url, etc.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Performing web search: {query}")

            # Limit num_results to reasonable bounds
            num_results = min(max(num_results, 1), 10)

            # Configure search parameters
            search_params = {
                "q": query,
                "api_key": self.config['api_key'],
                "num": num_results,
                "hl": "en",  # Language
                "gl": "us",  # Country
                "safe": "active"  # Safe search
            }

            # Perform search
            search = GoogleSearch(search_params)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                search.get_dict
            )

            # Check for API errors
            if "error" in results:
                logger.error(f"SERP API error: {results['error']}")
                raise Exception(f"Search failed: {results['error']}")

            # Process and format results
            formatted_results = self._format_search_results(results, query)

            logger.info(f"Search completed: {len(formatted_results)} results found")
            return formatted_results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")

    def _format_search_results(self, raw_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Format raw SERP API results into standardized format.

        Args:
            raw_results: Raw results from SERP API
            query: Original search query

        Returns:
            List of formatted search results
        """
        formatted_results = []

        # Extract organic results
        organic_results = raw_results.get("organic_results", [])

        for result in organic_results:
            formatted_result = {
                "title": result.get("title", "No title"),
                "snippet": result.get("snippet", "No description available"),
                "url": result.get("link", ""),
                "displayed_link": result.get("displayed_link", ""),
                "position": result.get("position", 0),
                "source": "google_search"
            }

            # Add additional metadata if available
            if "date" in result:
                formatted_result["date"] = result["date"]

            if "rich_snippet" in result:
                formatted_result["rich_snippet"] = result["rich_snippet"]

            formatted_results.append(formatted_result)

        # Add search metadata
        search_metadata = {
            "query": query,
            "total_results": len(formatted_results),
            "search_time": raw_results.get("search_metadata", {}).get("total_time_taken", 0),
            "search_id": raw_results.get("search_metadata", {}).get("id", ""),
        }

        # Add metadata to first result if results exist
        if formatted_results:
            formatted_results[0]["search_metadata"] = search_metadata

        return formatted_results

    def is_available(self) -> bool:
        """
        Check if SERP service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            return bool(self.config.get('api_key')) and self.is_initialized
        except Exception as e:
            logger.error(f"SERP availability check failed: {e}")
            return False

    async def get_search_suggestions(self, query: str) -> List[str]:
        """
        Get search suggestions for a query.

        Args:
            query: Partial search query

        Returns:
            List of search suggestions
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            search_params = {
                "q": query,
                "api_key": self.config['api_key'],
                "engine": "google_autocomplete"
            }

            search = GoogleSearch(search_params)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                search.get_dict
            )

            suggestions = []
            if "suggestions" in results:
                for suggestion in results["suggestions"]:
                    if "value" in suggestion:
                        suggestions.append(suggestion["value"])

            return suggestions[:5]  # Return top 5 suggestions

        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []

    async def search_news(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform news search using SERP API.

        Args:
            query: Search query string
            num_results: Number of news results to return

        Returns:
            List of news search results
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            search_params = {
                "q": query,
                "api_key": self.config['api_key'],
                "tbm": "nws",  # News search
                "num": min(num_results, 10)
            }

            search = GoogleSearch(search_params)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                search.get_dict
            )

            return self._format_news_results(results, query)

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []

    def _format_news_results(self, raw_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Format news search results."""
        formatted_results = []

        news_results = raw_results.get("news_results", [])

        for result in news_results:
            formatted_result = {
                "title": result.get("title", "No title"),
                "snippet": result.get("snippet", "No description available"),
                "url": result.get("link", ""),
                "source": result.get("source", "Unknown source"),
                "date": result.get("date", ""),
                "thumbnail": result.get("thumbnail", ""),
                "type": "news"
            }

            formatted_results.append(formatted_result)

        return formatted_results

    async def get_service_info(self) -> dict:
        """
        Get information about the SERP service.

        Returns:
            Dictionary with service information
        """
        return {
            "service_name": self.service_name,
            "search_engine": self.search_engine,
            "capabilities": ["web_search", "news_search", "autocomplete"],
            "max_results_per_query": 10,
            "initialized": self.is_initialized,
            "available": self.is_available()
        }
