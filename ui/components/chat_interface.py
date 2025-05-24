"""
Chat interface components for SONA AI Assistant.
Handles chat message rendering, formatting, and interaction.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import time
import json
from datetime import datetime
from loguru import logger

from utils.constants import SONA_PERSONA


class ChatInterface:
    """Chat interface component for SONA."""

    def __init__(self):
        """Initialize chat interface."""
        self.message_container = None
        self.input_container = None

    def render_message(self, message: Dict[str, Any], show_metadata: bool = False):
        """
        Render a single chat message.

        Args:
            message: Message dictionary containing role, content, timestamp, etc.
            show_metadata: Whether to show message metadata (intent, confidence, etc.)
        """
        try:
            role = message.get("role", "user")
            content = message.get("content", "")
            timestamp = message.get("timestamp", time.time())

            # Format timestamp
            formatted_time = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

            with st.chat_message(role):
                # Main message content
                st.write(content)

                # Show metadata if requested
                if show_metadata and role == "assistant":
                    self._render_message_metadata(message)

                # Show timestamp in small text
                st.caption(f"⏰ {formatted_time}")

        except Exception as e:
            logger.error(f"Error rendering message: {e}")
            st.error("Error displaying message")

    def _render_message_metadata(self, message: Dict[str, Any]):
        """
        Render message metadata (intent, confidence, etc.).

        Args:
            message: Message dictionary
        """
        try:
            metadata_items = []

            # Intent information
            if "intent" in message:
                intent = message["intent"]
                metadata_items.append(f"🎯 Intent: {intent}")

            # Confidence score
            if "confidence" in message:
                confidence = message["confidence"]
                confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                metadata_items.append(f"{confidence_emoji} Confidence: {confidence:.2f}")

            # Input type (for voice inputs)
            if "input_type" in message and message["input_type"] == "audio":
                metadata_items.append("🎤 Voice Input")

            # Response type
            if "response_type" in message:
                response_type = message["response_type"]
                type_emoji = {"text": "💬", "image": "🖼️", "audio": "🎵"}.get(response_type, "📝")
                metadata_items.append(f"{type_emoji} Type: {response_type}")

            # Display metadata if any exists
            if metadata_items:
                st.caption(" | ".join(metadata_items))

        except Exception as e:
            logger.error(f"Error rendering metadata: {e}")

    def render_message_list(self, messages: List[Dict[str, Any]], show_metadata: bool = False):
        """
        Render a list of chat messages.

        Args:
            messages: List of message dictionaries
            show_metadata: Whether to show metadata for messages
        """
        try:
            if not messages:
                st.info("No messages yet. Start a conversation!")
                return

            for message in messages:
                self.render_message(message, show_metadata)

        except Exception as e:
            logger.error(f"Error rendering message list: {e}")
            st.error("Error displaying chat history")

    def render_typing_indicator(self, message: str = "Thinking..."):
        """
        Render typing indicator for AI response.

        Args:
            message: Loading message to display
        """
        with st.chat_message("assistant"):
            with st.spinner(message):
                # Small delay for visual effect
                time.sleep(0.5)

    def format_search_results(self, results: List[Dict[str, Any]], query: str = "") -> str:
        """
        Format search results for display in chat.

        Args:
            results: List of search result dictionaries
            query: Original search query

        Returns:
            Formatted search results string
        """
        try:
            if not results:
                query_part = f' for "{query}"' if query else ''
                return f"No search results found{query_part}."

            query_part = f' for "{query}"' if query else ''
            formatted = f"🔍 **Search Results{query_part}:**\n\n"

            for i, result in enumerate(results[:5], 1):  # Show top 5 results
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description available")
                url = result.get("url", "")
                date = result.get("date", "")

                formatted += f"**{i}. {title}**\n"
                formatted += f"{snippet}\n"

                if url:
                    formatted += f"🔗 [Read more]({url})\n"

                if date:
                    formatted += f"📅 {date}\n"

                formatted += "\n"

            if len(results) > 5:
                formatted += f"... and {len(results) - 5} more results available.\n"

            return formatted

        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return "Error formatting search results."

    def render_search_results_detailed(self, results: List[Dict[str, Any]], query: str = ""):
        """
        Render search results with detailed formatting and expandable sections.

        Args:
            results: List of search result dictionaries
            query: Original search query
        """
        try:
            if not results:
                query_part = f' for "{query}"' if query else ''
                st.info(f"No search results found{query_part}.")
                return

            query_part = f' for "{query}"' if query else ''
            st.subheader(f"🔍 Search Results{query_part}")
            st.caption(f"Found {len(results)} result(s)")

            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description available")
                url = result.get("url", "")
                date = result.get("date", "")
                source = result.get("source", "")

                # Use expander for each result
                with st.expander(f"{i}. {title}", expanded=(i <= 2)):  # Expand first 2 results
                    st.write(snippet)

                    # Additional metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        if url:
                            st.markdown(f"🔗 [Read full article]({url})")
                        if source:
                            st.caption(f"📰 Source: {source}")

                    with col2:
                        if date:
                            st.caption(f"📅 {date}")
                        st.caption(f"🏷️ Result #{i}")

        except Exception as e:
            logger.error(f"Error rendering detailed search results: {e}")
            st.error("Error displaying search results")

    def render_image_result(self, image_data: Dict[str, Any], prompt: str = ""):
        """
        Render image generation result.

        Args:
            image_data: Image generation result dictionary
            prompt: Original image prompt
        """
        try:
            if not image_data.get("success"):
                error_msg = image_data.get("error", "Unknown error occurred")
                st.error(f"Image generation failed: {error_msg}")
                return

            query_part = f' for "{prompt}"' if prompt else ''
            st.subheader(f"🎨 Generated Image{query_part}")

            # Display image if available
            if image_data.get("image_data"):
                import base64
                import io
                from PIL import Image

                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data["image_data"])
                    image = Image.open(io.BytesIO(image_bytes))

                    st.image(image, caption=f"Generated: {prompt}", use_column_width=True)

                    # Image metadata
                    metadata = image_data.get("metadata", {})
                    if metadata:
                        with st.expander("Image Details"):
                            st.json(metadata)

                except Exception as e:
                    logger.error(f"Error displaying image: {e}")
                    st.error("Error displaying generated image")

            elif image_data.get("image_url"):
                st.image(image_data["image_url"], caption=f"Generated: {prompt}", use_column_width=True)
            else:
                st.warning("Image was generated but no image data is available for display")

        except Exception as e:
            logger.error(f"Error rendering image result: {e}")
            st.error("Error displaying image result")

    def render_welcome_message(self):
        """Render welcome message and capabilities."""
        st.markdown(f"""
        ### 👋 Welcome to {SONA_PERSONA['name']}!
        
        I'm {SONA_PERSONA['personality']} and ready to help you with:
        
        """)

        # Display capabilities as cards
        cols = st.columns(2)

        capabilities = [
            ("💬", "Text Chat", "Ask me questions in natural language"),
            ("🎤", "Voice Input", "Upload audio files for speech recognition"),
            ("🔍", "Web Search", "Get current information from the internet"),
            ("🎨", "Image Generation", "Create images from text descriptions")
        ]

        for i, (icon, title, description) in enumerate(capabilities):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem; margin: 0.5rem 0;">
                    <h4>{icon} {title}</h4>
                    <p style="margin: 0; color: #666;">{description}</p>
                </div>
                """, unsafe_allow_html=True)

    def render_error_message(self, error: str, details: Optional[str] = None):
        """
        Render error message with optional details.

        Args:
            error: Main error message
            details: Optional error details
        """
        st.error(f"❌ {error}")

        if details:
            with st.expander("Error Details"):
                st.code(details)

    def render_success_message(self, message: str):
        """
        Render success message.

        Args:
            message: Success message to display
        """
        st.success(f"✅ {message}")

    def render_info_message(self, message: str):
        """
        Render informational message.

        Args:
            message: Info message to display
        """
        st.info(f"ℹ️ {message}")

    def clear_chat_history(self):
        """Clear chat history from session state."""
        if "messages" in st.session_state:
            # Keep only the welcome message
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": SONA_PERSONA["greeting"],
                    "timestamp": time.time()
                }
            ]
            st.success("Chat history cleared!")
            st.rerun()

    def export_chat_history(self, messages: List[Dict[str, Any]], format: str = "json") -> str:
        """
        Export chat history in specified format.

        Args:
            messages: List of messages to export
            format: Export format ('json', 'txt', 'md')

        Returns:
            Formatted chat history string
        """
        try:
            if format == "json":
                return json.dumps(messages, indent=2, default=str)

            elif format == "txt":
                formatted = f"SONA Chat History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                formatted += "=" * 50 + "\n\n"

                for message in messages:
                    role = message.get("role", "unknown").upper()
                    content = message.get("content", "")
                    timestamp = message.get("timestamp", time.time())
                    time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

                    formatted += f"[{time_str}] {role}:\n{content}\n\n"

                return formatted

            elif format == "md":
                formatted = f"# SONA Chat History\n\n"
                formatted += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                for message in messages:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    timestamp = message.get("timestamp", time.time())
                    time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

                    icon = "🤖" if role == "assistant" else "👤"
                    formatted += f"## {icon} {role.title()} - {time_str}\n\n{content}\n\n"

                return formatted

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Error exporting chat history: {e}")
            return f"Error exporting chat history: {e}"

    def render_chat_stats(self, messages: List[Dict[str, Any]]):
        """
        Render chat statistics.

        Args:
            messages: List of messages to analyze
        """
        try:
            if not messages:
                return

            # Calculate stats
            total_messages = len(messages)
            user_messages = len([m for m in messages if m.get("role") == "user"])
            assistant_messages = len([m for m in messages if m.get("role") == "assistant"])

            # Intent breakdown
            intents = [m.get("intent") for m in messages if m.get("intent")]
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            # Display stats
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Messages", total_messages)

            with col2:
                st.metric("Your Messages", user_messages)

            with col3:
                st.metric("SONA Responses", assistant_messages)

            # Intent breakdown
            if intent_counts:
                st.subheader("Intent Breakdown")
                for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {intent}: {count}")

        except Exception as e:
            logger.error(f"Error rendering chat stats: {e}")
            st.error("Error displaying chat statistics")
