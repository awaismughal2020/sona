"""
Streamlit UI for SONA AI Assistant with proper voice processing reset.
Provides web-based interface for text and voice interactions.
"""

import streamlit as st
import requests
import json
import io
import base64
from PIL import Image
import asyncio
import sys
import os
from typing import Optional, Dict, Any
import time

# IMPORTANT: set_page_config must be the FIRST Streamlit command
st.set_page_config(
    page_title="SONA AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from utils.constants import SONA_PERSONA
from ui.components.chat_interface import ChatInterface
from ui.components.voice_input import VoiceInputComponent


class SONAStreamlitApp:
    """SONA Streamlit Application."""

    def __init__(self):
        """Initialize Streamlit app."""
        self.settings = get_settings()
        self.backend_url = self.settings.get_backend_url()
        self.chat_interface = ChatInterface()
        self.voice_component = VoiceInputComponent()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": SONA_PERSONA["greeting"],
                    "timestamp": time.time()
                }
            ]

        if "session_id" not in st.session_state:
            st.session_state.session_id = f"session_{int(time.time())}"

        if "backend_health" not in st.session_state:
            st.session_state.backend_health = None

        if "available_models" not in st.session_state:
            st.session_state.available_models = {}

        # Voice processing state
        if "voice_processing" not in st.session_state:
            st.session_state.voice_processing = False

        if "last_processed_file" not in st.session_state:
            st.session_state.last_processed_file = None

        if "voice_file_key" not in st.session_state:
            st.session_state.voice_file_key = 0

    def check_backend_health(self) -> bool:
        """Check if backend is healthy."""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                st.session_state.backend_health = response.json()
                return True
            else:
                st.session_state.backend_health = None
                return False
        except Exception as e:
            st.session_state.backend_health = None
            return False

    def send_message(self, message: str) -> Optional[Dict[str, Any]]:
        """Send text message to backend."""
        try:
            data = {
                "message": message,
                "session_id": st.session_state.session_id
            }

            response = requests.post(
                f"{self.backend_url}/api/v1/chat",
                data=data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Backend error: {response.status_code}")
                return None

        except Exception as e:
            st.error(f"Failed to send message: {str(e)}")
            return None

    def upload_audio(self, audio_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """Upload audio file to backend."""
        try:
            files = {"audio_file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
            data = {"session_id": st.session_state.session_id}

            response = requests.post(
                f"{self.backend_url}/api/v1/upload-audio",
                files=files,
                data=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Audio processing error: {response.status_code}")
                return None

        except Exception as e:
            st.error(f"Failed to process audio: {str(e)}")
            return None

    def get_available_models(self):
        """Get available models from backend."""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/models", timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.session_state.available_models = result.get("models", {})
        except Exception as e:
            st.warning(f"Could not fetch available models: {str(e)}")

    def switch_model(self, service_type: str, model_type: str) -> bool:
        """Switch AI model for a service."""
        try:
            data = {
                "service_type": service_type,
                "model_type": model_type
            }

            response = requests.post(
                f"{self.backend_url}/api/v1/switch-model",
                data=data,
                timeout=15
            )

            return response.status_code == 200

        except Exception as e:
            st.error(f"Failed to switch model: {str(e)}")
            return False

    def render_sidebar(self):
        """Render sidebar with configuration options."""
        with st.sidebar:
            st.title(f"⚙️ {SONA_PERSONA['name']} Settings")

            # Backend health status
            st.subheader("🔍 System Status")
            if self.check_backend_health():
                st.success("✅ Backend Connected")

                # Show service health
                if st.session_state.backend_health:
                    health_info = st.session_state.backend_health.get("ai_services", {})
                    if isinstance(health_info, dict) and "summary" in health_info:
                        summary = health_info["summary"]
                        st.info(f"Services: {summary['healthy']}/{summary['total']} healthy")
            else:
                st.error("❌ Backend Disconnected")
                st.warning("Please ensure the backend server is running.")

            st.divider()

            # Model configuration
            st.subheader("🤖 AI Models")

            # Get available models
            if st.button("🔄 Refresh Models"):
                self.get_available_models()

            if st.session_state.available_models:
                for service_type, models in st.session_state.available_models.items():
                    if models:
                        st.write(f"**{service_type.replace('_', ' ').title()}:**")

                        current_model = getattr(self.settings, f"{service_type}_model", models[0])

                        selected_model = st.selectbox(
                            f"Select {service_type} model:",
                            models,
                            index=models.index(current_model) if current_model in models else 0,
                            key=f"model_{service_type}"
                        )

                        if selected_model != current_model:
                            if st.button(f"Switch {service_type}", key=f"switch_{service_type}"):
                                if self.switch_model(service_type, selected_model):
                                    st.success(f"Switched to {selected_model}")
                                    st.rerun()

            st.divider()

            # Capabilities
            st.subheader("💡 Capabilities")
            for capability in SONA_PERSONA["capabilities"]:
                st.write(f"• {capability}")

            st.divider()

            # Session info
            st.subheader("📊 Session Info")
            st.write(f"**Session ID:** {st.session_state.session_id[:12]}...")
            st.write(f"**Messages:** {len(st.session_state.messages) - 1}")  # Exclude greeting

            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": SONA_PERSONA["greeting"],
                        "timestamp": time.time()
                    }
                ]
                st.rerun()

            # Voice processing controls
            st.divider()
            st.subheader("🎤 Voice Controls")

            if st.button("🔄 Reset Voice Input"):
                self._reset_voice_input()
                st.success("Voice input reset!")
                st.rerun()

    def _reset_voice_input(self):
        """Reset voice input state."""
        st.session_state.voice_processing = False
        st.session_state.last_processed_file = None
        st.session_state.voice_file_key += 1  # This will reset the file uploader

    def render_main_chat(self):
        """Render main chat interface."""
        st.title(f"💬 {SONA_PERSONA['name']}")
        st.write(f"*{SONA_PERSONA['personality']}*")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # Show additional data if available
                if "data" in message and message["data"]:
                    self._render_message_data(message["data"], message.get("response_type"))

        # Input section
        st.divider()

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": time.time()
            })

            # Display user message immediately
            with st.chat_message("user"):
                st.write(prompt)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.send_message(prompt)

                if response and response.get("success"):
                    response_content = response["response"]
                    st.write(response_content)

                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": time.time(),
                        "intent": response.get("intent"),
                        "confidence": response.get("confidence"),
                        "response_type": response.get("response_type"),
                        "data": response.get("data")
                    })

                    # Show additional data
                    if response.get("data"):
                        self._render_message_data(response["data"], response.get("response_type"))
                else:
                    error_msg = "Sorry, I couldn't process your request right now."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": time.time()
                    })

        # Voice input section
        st.markdown("---")
        st.markdown("### 🎤 Voice Input")
        self._render_voice_input_section()

    def _render_voice_input_section(self):
        """Render voice input section with proper reset functionality."""
        st.markdown("Upload an audio file to interact with SONA using your voice.")

        # Create columns for better layout
        col1, col2 = st.columns([3, 1])

        with col1:
            # File upload with dynamic key for reset functionality
            uploaded_file = st.file_uploader(
                "Upload audio file",
                type=self.settings.get_allowed_audio_formats(),
                help="Supported formats: " + ", ".join(self.settings.get_allowed_audio_formats()),
                key=f"audio_uploader_{st.session_state.voice_file_key}"
            )

        with col2:
            # Reset button
            if st.button("🔄 Reset", help="Clear audio upload"):
                self._reset_voice_input()
                st.rerun()

        # Only show file info and process button if file is uploaded and not currently processing
        if uploaded_file is not None and not st.session_state.voice_processing:
            # Check if this is a new file
            file_info = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"

            # Show audio file info
            st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📁 **File:** {uploaded_file.name}")
            with col2:
                file_size = len(uploaded_file.getvalue())
                st.info(f"📊 **Size:** {file_size / 1024:.1f} KB")

            # Process button
            if st.button("🎯 Process Audio", key="process_upload_btn"):
                self._process_audio_file(uploaded_file)

        elif st.session_state.voice_processing:
            st.info("🔄 Processing audio... Please wait.")

        # Tips section
        with st.expander("💡 Tips for Better Voice Recognition"):
            st.markdown(f"""
            **For best results:**
            - 🎯 Speak clearly and at normal pace
            - 🔇 Record in a quiet environment
            - ⏱️ Keep recordings under 2 minutes
            - 🎤 Use good quality microphone if possible

            **Supported formats:** {', '.join(self.settings.get_allowed_audio_formats())}

            **File size limit:** {self.settings.max_file_size // (1024 * 1024)}MB

            **Example commands:**
            - "What's the weather in Islamabad?"
            - "Generate an image of a sunset"
            - "Tell me about cryptocurrency prices"
            """)

    def _process_audio_file(self, uploaded_file):
        """Process uploaded audio file with proper state management."""
        try:
            # Set processing state
            st.session_state.voice_processing = True

            with st.spinner("Processing audio..."):
                # Read audio file
                audio_bytes = uploaded_file.read()

                # Process with backend
                response = self.upload_audio(audio_bytes, uploaded_file.name)

            # Reset processing state
            st.session_state.voice_processing = False

            if response and response.get("success"):
                transcription = response.get("transcription", "")

                # Add transcription as user message
                if transcription:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"🎤 {transcription}",
                        "timestamp": time.time(),
                        "input_type": "audio"
                    })

                # Add assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": response["response"],
                    "timestamp": time.time(),
                    "intent": response.get("intent"),
                    "confidence": response.get("confidence"),
                    "response_type": response.get("response_type"),
                    "data": response.get("data")
                }

                st.session_state.messages.append(assistant_message)

                # Show success and results
                st.success(f"**Transcription:** {transcription}")
                st.write(f"**SONA's Response:** {response['response']}")

                # Show additional data
                if response.get("data"):
                    self._render_message_data(response["data"], response.get("response_type"))

                # Store processed file info to prevent reprocessing
                st.session_state.last_processed_file = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"

                # Auto-reset after successful processing
                st.session_state.voice_file_key += 1

                # Small delay and rerun to show results and reset interface
                time.sleep(1)
                st.rerun()

            else:
                st.session_state.voice_processing = False
                error_msg = response.get("error", "Unknown error") if response else "No response from server"
                st.error(f"Failed to process audio: {error_msg}")

        except Exception as e:
            st.session_state.voice_processing = False
            st.error(f"Audio processing error: {str(e)}")

    def _render_message_data(self, data: Any, response_type: str):
        """Render additional message data based on response type with smaller images."""
        if response_type == "image" and data:
            if data.get("success") and data.get("image_data"):
                try:
                    # Decode base64 image data if present
                    image_data = data["image_data"]
                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))

                        # Display image with fixed width of 400px
                        st.image(image, caption="Generated Image Description", width=400)

                        # Show enhanced description if available
                        if "metadata" in data and "detailed_description" in data["metadata"]:
                            enhanced_desc = data["metadata"]["detailed_description"]
                            if enhanced_desc and len(enhanced_desc) > 50:
                                with st.expander("📝 View Full Description"):
                                    st.write(enhanced_desc)

                except Exception as e:
                    st.error(f"Could not display image: {e}")
            elif data.get("image_url"):
                st.image(data["image_url"], caption="Generated Image", width=400)
            else:
                st.warning("Image generation failed or no image data available")

        elif response_type == "text" and isinstance(data, list):
            # Web search results
            if data:
                st.subheader("🔍 Search Results")

                for i, result in enumerate(data[:3], 1):  # Show top 3 results
                    with st.expander(f"{i}. {result.get('title', 'No title')}", expanded=i == 1):
                        st.write(result.get('snippet', 'No description available'))
                        if result.get('url'):
                            st.markdown(f"🔗 [Read more]({result['url']})")
                        if result.get('date'):
                            st.caption(f"Published: {result['date']}")

    def run(self):
        """Run the Streamlit application."""
        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
        }
        .user-message {
            background-color: #e8f4f8;
        }
        .assistant-message {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

        # Render components
        self.render_sidebar()
        self.render_main_chat()


# Main application entry point
def main():
    """Main application entry point."""
    app = SONAStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
