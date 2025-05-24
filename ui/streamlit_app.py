"""
Streamlit UI for SONA AI Assistant.
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
                    health_info = st.session_state.backend_health.get("services", {})
                    if "summary" in health_info:
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
        """Render voice input section with both upload and real-time recording."""
        st.markdown("Upload an audio file to interact with SONA using your voice.")

        # Create tabs for different voice input methods
        upload_tab, record_tab = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])

        with upload_tab:
            self._render_file_upload()

        with record_tab:
            self._render_real_time_recording()

    def _render_file_upload(self):
        """Render file upload interface."""
        # File upload
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=self.settings.get_allowed_audio_formats(),
            help="Supported formats: " + ", ".join(self.settings.get_allowed_audio_formats()),
            key="audio_uploader"
        )

        if uploaded_file is not None:
            # Show audio file info
            st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📁 **File:** {uploaded_file.name}")
            with col2:
                file_size = len(uploaded_file.getvalue())
                st.info(f"📊 **Size:** {file_size / 1024:.1f} KB")

            if st.button("🎯 Process Audio", key="process_upload_btn"):
                self._process_audio_file(uploaded_file)

    def _render_real_time_recording(self):
        """Render real-time audio recording interface."""
        st.markdown("### 🎙️ Real-Time Voice Recording")

        # Recording controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔴 Start Recording", key="start_record"):
                st.session_state.recording = True
                st.success("Recording started! Speak now...")

        with col2:
            if st.button("⏹️ Stop Recording", key="stop_record"):
                if st.session_state.get('recording'):
                    st.session_state.recording = False
                    st.info("Recording stopped.")

        with col3:
            if st.button("🎯 Process Recording", key="process_record"):
                st.info("Processing recorded audio...")

        # Recording status
        if st.session_state.get('recording'):
            st.warning("🔴 **Recording in progress...** Speak clearly into your microphone.")
        else:
            st.info("💡 **Ready to record.** Click 'Start Recording' to begin.")

        # Instructions
        with st.expander("📱 Recording Instructions"):
            st.markdown("""
            **How to use real-time recording:**

            1. **Click 'Start Recording'** - Grant microphone permission if asked
            2. **Speak clearly** - Talk at normal pace, avoid background noise
            3. **Click 'Stop Recording'** - When you're done speaking
            4. **Click 'Process Recording'** - To transcribe and get AI response

            **Tips for better results:**
            - Use a quiet environment
            - Speak directly towards your microphone
            - Keep recordings under 30 seconds for best results
            - Make sure your browser has microphone permissions
            """)

        # Browser-based recording note
        st.info("""
        🚧 **Real-time recording is in development!** 

        For now, you can:
        1. Use your phone's voice recorder app
        2. Record a message and save it
        3. Upload the file using the 'Upload Audio' tab above

        Browser-based recording will be available in the next update!
        """)

    def _process_audio_file(self, uploaded_file):
        """Process uploaded audio file."""
        try:
            with st.spinner("Processing audio..."):
                # Read audio file
                audio_bytes = uploaded_file.read()

                # Process with backend
                response = self.upload_audio(audio_bytes, uploaded_file.name)

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

                st.success(f"**Transcription:** {transcription}")
                st.write(f"**SONA's Response:** {response['response']}")

                # Show additional data
                if response.get("data"):
                    self._render_message_data(response["data"], response.get("response_type"))

                # Small delay before rerun to show results
                time.sleep(0.5)

            else:
                error_msg = response.get("error", "Unknown error") if response else "No response from server"
                st.error(f"Failed to process audio: {error_msg}")

        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")

    def _render_text_input(self):
        """This method is no longer used - integrated into render_main_chat."""
        pass

    def _render_voice_input(self):
        """This method is no longer used - integrated into render_main_chat."""
        pass

    def _render_message_data(self, data: Any, response_type: str):
        """Render additional message data based on response type."""
        if response_type == "image" and data:
            if data.get("success") and data.get("image_data"):
                try:
                    # Decode base64 image data if present
                    image_data = data["image_data"]
                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption="Generated Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {e}")
            elif data.get("image_url"):
                st.image(data["image_url"], caption="Generated Image", use_column_width=True)
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


# UI Components
# ui/components/chat_interface.py
"""
Chat interface components for SONA AI Assistant.
"""

import streamlit as st
from typing import Dict, Any, List
import time


class ChatInterface:
    """Chat interface component for SONA."""

    def __init__(self):
        """Initialize chat interface."""
        pass

    def render_message(self, message: Dict[str, Any], show_metadata: bool = False):
        """
        Render a single chat message.

        Args:
            message: Message dictionary
            show_metadata: Whether to show message metadata
        """
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", time.time())

        with st.chat_message(role):
            st.write(content)

            if show_metadata:
                # Show metadata
                metadata = []

                if "intent" in message:
                    metadata.append(f"Intent: {message['intent']}")

                if "confidence" in message:
                    confidence = message['confidence']
                    metadata.append(f"Confidence: {confidence:.2f}")

                if "input_type" in message:
                    metadata.append(f"Input: {message['input_type']}")

                if metadata:
                    st.caption(" | ".join(metadata))

    def render_typing_indicator(self):
        """Render typing indicator."""
        with st.chat_message("assistant"):
            st.write("🤔 Thinking...")

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."

        formatted = "Here's what I found:\n\n"

        for i, result in enumerate(results[:3], 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description")
            url = result.get("url", "")

            formatted += f"**{i}. {title}**\n"
            formatted += f"{snippet}\n"
            if url:
                formatted += f"🔗 [Read more]({url})\n"
            formatted += "\n"

        return formatted


# ui/components/voice_input.py
"""
Voice input components for SONA AI Assistant.
"""

import streamlit as st
import io
from typing import Optional, Tuple


class VoiceInputComponent:
    """Voice input component for SONA."""

    def __init__(self):
        """Initialize voice input component."""
        self.supported_formats = ['wav', 'mp3', 'm4a', 'flac']

    def render_file_upload(self) -> Optional[Tuple[bytes, str]]:
        """
        Render audio file upload interface.

        Returns:
            Tuple of (audio_bytes, filename) if file uploaded, None otherwise
        """
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=self.supported_formats,
            help=f"Supported formats: {', '.join(self.supported_formats)}"
        )

        if uploaded_file is not None:
            # Show audio player
            st.audio(uploaded_file, format='audio/wav')

            # Show file info
            file_size = len(uploaded_file.getvalue())
            st.caption(f"File: {uploaded_file.name} ({file_size} bytes)")

            return uploaded_file.getvalue(), uploaded_file.name

        return None

    def render_recording_interface(self):
        """
        Render audio recording interface.
        Note: Browser-based recording requires additional JavaScript components.
        """
        st.info("🎤 **Audio Recording**")
        st.write("Browser-based recording will be available in a future update.")
        st.write("For now, please upload an audio file using the file uploader above.")

    def validate_audio_format(self, filename: str) -> bool:
        """Validate audio file format."""
        if not filename:
            return False

        extension = filename.split('.')[-1].lower()
        return extension in self.supported_formats


# Main application entry point
def main():
    """Main application entry point."""
    app = SONAStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
