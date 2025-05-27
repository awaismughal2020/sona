"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.

Complete Streamlit UI for SONA AI Assistant with conversation context display.
"""

import streamlit as st
import requests
import json
import io
import base64
from PIL import Image
import time
from typing import Optional, Dict, Any
import numpy as np

# Audio imports (conditional)
try:
    import sounddevice as sd
    import wave
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

st.set_page_config(
    page_title="SONA AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from utils.constants import SONA_PERSONA
from ui.components.chat_interface import ChatInterface


class SONAStreamlitAppWithContext:
    """SONA Streamlit Application with Context Management Display."""

    def __init__(self):
        """Initialize Streamlit app with context features."""
        self.settings = get_settings()
        self.backend_url = self.settings.get_backend_url()
        self.chat_interface = ChatInterface()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables with context support."""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": SONA_PERSONA["greeting"],
                    "timestamp": time.time(),
                    "context_aware": False
                }
            ]

        if "session_id" not in st.session_state:
            st.session_state.session_id = f"session_{int(time.time())}"

        if "backend_health" not in st.session_state:
            st.session_state.backend_health = None

        if "context_info" not in st.session_state:
            st.session_state.context_info = None

        # Voice processing state
        if "voice_processing" not in st.session_state:
            st.session_state.voice_processing = False

        if "voice_tab_key" not in st.session_state:
            st.session_state.voice_tab_key = 0

        # Context management state
        if "show_context_details" not in st.session_state:
            st.session_state.show_context_details = False

        if "context_stats" not in st.session_state:
            st.session_state.context_stats = {}

    def check_backend_health(self) -> bool:
        """Check if backend is healthy and get context information."""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                st.session_state.backend_health = health_data

                # Extract context management info
                if "context_features" in health_data:
                    st.session_state.context_info = health_data["context_features"]

                return True
            else:
                st.session_state.backend_health = None
                st.session_state.context_info = None
                return False
        except Exception as e:
            st.session_state.backend_health = None
            st.session_state.context_info = None
            return False

    def get_session_context(self) -> Optional[Dict[str, Any]]:
        """Get context information for current session."""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/context/{st.session_state.session_id}",
                timeout=10
            )

            if response.status_code == 200:
                context_data = response.json()
                if context_data.get("success"):
                    st.session_state.context_stats = context_data.get("context", {})
                    return context_data.get("context")

            return None

        except Exception as e:
            st.error(f"Failed to get context information: {str(e)}")
            return None

    def clear_session_context(self) -> bool:
        """Clear context for current session."""
        try:
            response = requests.delete(
                f"{self.backend_url}/api/v1/context/{st.session_state.session_id}",
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)

            return False

        except Exception as e:
            st.error(f"Failed to clear context: {str(e)}")
            return False

    def send_message(self, message: str) -> Optional[Dict[str, Any]]:
        """Send text message to backend with context support."""
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
        """Upload audio file to backend with context support."""
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

    def render_context_sidebar(self):
        """Render context management sidebar."""
        with st.sidebar:
            st.title(f"⚙️ {SONA_PERSONA['name']} Settings")

            # Backend health status
            st.subheader("🔍 System Status")
            if self.check_backend_health():
                st.success("✅ Backend Connected")

                # Context management status
                if st.session_state.context_info:
                    context_info = st.session_state.context_info

                    if context_info.get("context_persistence"):
                        st.success("🧠 Context Management: Active")
                    else:
                        st.warning("🧠 Context Management: Inactive")

                    if context_info.get("multi_turn_awareness"):
                        st.info("💬 Multi-turn Conversations: Enabled")

                    if context_info.get("session_management"):
                        st.info("📝 Session Tracking: Enabled")

                # Show service health
                if st.session_state.backend_health:
                    health_info = st.session_state.backend_health.get("ai_services", {})
                    if isinstance(health_info, dict) and "summary" in health_info:
                        summary = health_info["summary"]
                        st.info(f"AI Services: {summary['healthy']}/{summary['total']} healthy")
            else:
                st.error("❌ Backend Disconnected")
                st.warning("Please ensure the backend server is running.")

            st.divider()

            # Context Information
            st.subheader("🧠 Conversation Context")

            # Get current context stats
            context_stats = self.get_session_context()

            if context_stats:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Turns", context_stats.get("turn_count", 0))
                    st.metric("Context Items", context_stats.get("context_items_count", 0))

                with col2:
                    st.metric("Messages", len(st.session_state.messages) - 1)

                    # Show current topic if available
                    current_topic = context_stats.get("current_topic")
                    if current_topic:
                        st.caption(f"**Topic:** {current_topic[:30]}...")

                # Context details toggle
                if st.toggle("Show Context Details", key="context_toggle"):
                    st.session_state.show_context_details = True

                    with st.expander("📊 Context Details", expanded=True):
                        st.write(f"**Session ID:** {st.session_state.session_id[:12]}...")

                        if context_stats.get("created_at"):
                            created_time = time.strftime(
                                "%H:%M:%S",
                                time.localtime(context_stats["created_at"])
                            )
                            st.write(f"**Started:** {created_time}")

                        if context_stats.get("last_activity"):
                            last_activity = time.strftime(
                                "%H:%M:%S",
                                time.localtime(context_stats["last_activity"])
                            )
                            st.write(f"**Last Activity:** {last_activity}")

                        # User preferences
                        prefs_count = context_stats.get("user_preferences_count", 0)
                        if prefs_count > 0:
                            st.write(f"**Preferences Learned:** {prefs_count}")

            else:
                st.info("No context information available")

            st.divider()

            # Context Management Actions
            st.subheader("🔧 Context Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Refresh Context"):
                    self.get_session_context()
                    st.success("Context refreshed!")
                    st.rerun()

            with col2:
                if st.button("🗑️ Clear Context"):
                    if self.clear_session_context():
                        st.success("Context cleared!")
                        # Also clear UI messages except welcome
                        st.session_state.messages = [
                            {
                                "role": "assistant",
                                "content": SONA_PERSONA["greeting"],
                                "timestamp": time.time(),
                                "context_aware": False
                            }
                        ]
                        st.rerun()
                    else:
                        st.error("Failed to clear context")

            # New session button
            if st.button("🆕 New Session"):
                # Generate new session ID
                st.session_state.session_id = f"session_{int(time.time())}"
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": SONA_PERSONA["greeting"],
                        "timestamp": time.time(),
                        "context_aware": False
                    }
                ]
                st.session_state.context_stats = {}
                st.success("New session started!")
                st.rerun()

            st.divider()

            # Voice system status
            st.subheader("🎤 Voice System")
            if AUDIO_LIBS_AVAILABLE:
                try:
                    devices = sd.query_devices()
                    input_devices = [d for d in devices if d['max_input_channels'] > 0]

                    if input_devices:
                        st.success("✅ Real-time recording available")
                        st.info(f"Input devices: {len(input_devices)}")
                    else:
                        st.warning("⚠️ No input devices found")

                except Exception as e:
                    st.warning(f"⚠️ Audio system issue: {str(e)}")
            else:
                st.error("❌ Real-time recording disabled")
                st.caption("Install sounddevice: pip install sounddevice")

            st.divider()

            # Capabilities
            st.subheader("💡 Capabilities")
            capabilities = [
                "🧠 Context-aware conversations",
                "💬 Multi-turn dialogue memory",
                "🔍 Contextual web search",
                "🎨 Enhanced image generation",
                "🎤 Voice input processing"
            ]

            for capability in capabilities:
                st.write(f"• {capability}")

    def render_main_chat(self):
        """Render main chat interface with context indicators."""
        st.title(f"💬 {SONA_PERSONA['name']}")
        st.write(f"*{SONA_PERSONA['personality']} with conversation memory*")

        # Show context status banner
        if st.session_state.context_info and st.session_state.context_info.get("context_persistence"):
            st.info("🧠 **Context Management Active** - I remember our conversation and provide context-aware responses!")

        # Display chat messages with context indicators
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # Show context indicators
                if message["role"] == "assistant" and message.get("context_aware"):
                    st.caption("🧠 Context-aware response")

                if message.get("input_type") == "audio":
                    st.caption("🎤 Voice input")

                # Show additional data if available
                if "data" in message and message["data"]:
                    self._render_message_data(message["data"], message.get("response_type"))

        # Text input section
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
                with st.spinner("Thinking with context..."):
                    response = self.send_message(prompt)

                if response and response.get("success"):
                    response_content = response["response"]
                    st.write(response_content)

                    # Show context usage indicator
                    if response.get("context_aware"):
                        if response.get("context_used"):
                            st.caption("🧠 Used conversation context")
                        else:
                            st.caption("🧠 Context-aware processing")

                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": time.time(),
                        "intent": response.get("intent"),
                        "confidence": response.get("confidence"),
                        "response_type": response.get("response_type"),
                        "data": response.get("data"),
                        "context_aware": response.get("context_aware", False),
                        "turn_id": response.get("turn_id")
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
                        "timestamp": time.time(),
                        "context_aware": False
                    })

        # Voice input section
        st.markdown("---")
        self._render_enhanced_voice_section()

    def _render_enhanced_voice_section(self):
        """Render enhanced voice input section with context support."""
        st.markdown("### 🎤 Voice Input")

        # Check if currently processing
        if st.session_state.voice_processing:
            st.info("🔄 Processing voice input with context... Please wait.")
            return

        # Create tabs for different voice input methods
        tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Real-time Recording"])

        with tab1:
            self._render_file_upload_tab()

        with tab2:
            self._render_realtime_recording_tab()

    def _render_file_upload_tab(self):
        """Render file upload tab with context awareness."""
        st.markdown("Upload an audio file to transcribe with context awareness:")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=self.settings.get_allowed_audio_formats(),
            help=f"Supported: {', '.join(self.settings.get_allowed_audio_formats())}",
            key=f"file_upload_{st.session_state.voice_tab_key}"
        )

        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue())

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📁 **File:** {uploaded_file.name}")
            with col2:
                st.info(f"📊 **Size:** {file_size / 1024:.1f} KB")

            # Audio player
            st.audio(uploaded_file.getvalue())

            # Process button with context indicator
            if st.button("🎯 Process Audio with Context", key=f"process_file_{st.session_state.voice_tab_key}"):
                self.process_audio_callback(uploaded_file.getvalue(), uploaded_file.name)

    def _render_realtime_recording_tab(self):
        """Render real-time recording tab with context features."""
        if not AUDIO_LIBS_AVAILABLE:
            st.error("❌ Real-time recording not available")
            st.markdown("""
            Install required dependencies:
            ```bash
            pip install sounddevice numpy
            ```
            """)
            return

        # Show available devices
        try:
            devices = sd.query_devices()
            input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]

            if not input_devices:
                st.error("❌ No input audio devices found")
                return

            st.success(f"✅ Found {len(input_devices)} input devices")
            st.info("🧠 Voice input will be processed with conversation context")

            # Device selection (optional)
            with st.expander("🎛️ Device Selection (Optional)"):
                st.markdown("The app will automatically use the best available device, but you can choose a specific one:")
                self._render_device_selector()

            # Recording interface
            st.markdown("**Quick Recording Options:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🔴 Record 5s", key=f"record_5s_{st.session_state.voice_tab_key}"):
                    self._quick_record(5)

            with col2:
                if st.button("🔴 Record 10s", key=f"record_10s_{st.session_state.voice_tab_key}"):
                    self._quick_record(10)

            with col3:
                if st.button("🔴 Record 15s", key=f"record_15s_{st.session_state.voice_tab_key}"):
                    self._quick_record(15)

            # Custom duration
            st.markdown("**Custom Duration:**")

            col1, col2 = st.columns([3, 1])

            with col1:
                duration = st.slider(
                    "Duration (seconds)",
                    min_value=1,
                    max_value=30,
                    value=10,
                    key=f"duration_slider_{st.session_state.voice_tab_key}"
                )

            with col2:
                if st.button(f"🔴 Record", key=f"record_custom_{st.session_state.voice_tab_key}"):
                    self._quick_record(duration)

            # Tips specific to your setup
            with st.expander("💡 Tips for Your Setup"):
                st.markdown("""
                **For best results:**
                - Speak clearly and at normal volume
                - Record in a quiet environment
                - The app will process your voice with conversation context

                **Troubleshooting:**
                - If recording fails, try refreshing the page
                - Check microphone permissions in browser settings
                - The app will automatically try different devices if one fails
                """)

        except Exception as e:
            st.error(f"❌ Audio system error: {str(e)}")
            return

    def _render_device_selector(self):
        """Render device selection interface."""
        if not AUDIO_LIBS_AVAILABLE:
            return

        try:
            devices = sd.query_devices()
            input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

            if input_devices:
                st.markdown("**🎤 Available Microphones:**")

                device_options = []
                device_mapping = {}

                for device_id, device in input_devices:
                    device_name = device['name']
                    device_options.append(f"[{device_id}] {device_name}")
                    device_mapping[f"[{device_id}] {device_name}"] = device_id

                # Default to first device
                selected_device = st.selectbox(
                    "Choose your preferred microphone:",
                    device_options,
                    index=0,
                    key="device_selector"
                )

                selected_device_id = device_mapping[selected_device]

                # Store selected device in session state
                st.session_state.preferred_device_id = selected_device_id
                st.session_state.preferred_device_name = selected_device.split('] ')[1]

                # Show device info
                selected_device_info = devices[selected_device_id]
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Channels:** {selected_device_info['max_input_channels']}")
                with col2:
                    st.info(f"**Sample Rate:** {selected_device_info['default_samplerate']} Hz")

                return selected_device_id

        except Exception as e:
            st.error(f"Error listing devices: {e}")
            return None

    def _quick_record(self, duration: float):
        """Perform quick recording with context processing."""
        if not AUDIO_LIBS_AVAILABLE:
            st.error("❌ Audio recording libraries not available")
            return

        try:
            # Countdown
            countdown_placeholder = st.empty()
            progress_bar = st.progress(0.0)

            # Use preferred device if selected, otherwise use default
            device_id = getattr(st.session_state, 'preferred_device_id', None)

            # Countdown
            for i in range(3, 0, -1):
                countdown_placeholder.warning(f"🔴 Recording starts in {i}...")
                time.sleep(1)

            countdown_placeholder.info("🔴 Recording now! Speak clearly...")

            sample_rate = self.settings.audio_sample_rate

            try:
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype=np.float32,
                    device=device_id
                )

                # Show progress
                start_time = time.time()
                while not recording.flags.writeable:
                    elapsed = time.time() - start_time
                    progress = min(elapsed / duration, 1.0)
                    progress_bar.progress(progress)

                    remaining = duration - elapsed
                    if remaining > 0:
                        countdown_placeholder.info(f"🔴 Recording... {remaining:.1f}s left")

                    time.sleep(0.1)

                sd.wait()
                progress_bar.progress(1.0)
                countdown_placeholder.success("✅ Recording completed! Processing with context...")

            except Exception as e:
                st.error(f"Recording failed with device {device_id}: {e}")
                return

            # Check if we actually got audio
            max_amplitude = np.max(np.abs(recording))
            rms_level = np.sqrt(np.mean(recording ** 2))

            if max_amplitude < 0.001:
                countdown_placeholder.warning("⚠️ Very low audio detected")
                st.warning("Audio level is very low. Try speaking louder or closer to the microphone.")

            # Convert to WAV format
            wav_data = self._numpy_to_wav(recording, sample_rate)

            if wav_data:
                # Show audio player
                st.audio(wav_data, format="audio/wav")

                # Show audio quality info
                with st.expander("📊 Recording Info"):
                    st.write(f"**Duration:** {duration} seconds")
                    st.write(f"**Max amplitude:** {max_amplitude:.4f}")
                    st.write(f"**RMS level:** {rms_level:.4f}")
                    st.write(f"**Sample rate:** {sample_rate} Hz")
                    st.write(f"**File size:** {len(wav_data)} bytes")

                # Process the recording with context
                filename = f"recording_{int(time.time())}.wav"
                self.process_audio_callback(wav_data, filename)
            else:
                st.error("❌ Failed to process recording")

        except Exception as e:
            st.error(f"❌ Recording failed: {str(e)}")

    def process_audio_callback(self, audio_bytes: bytes, filename: str):
        """Callback function to process audio data with context support."""
        try:
            st.session_state.voice_processing = True

            # Show processing status with context indicator
            with st.spinner("🔄 Processing voice input with conversation context..."):
                # Upload and process audio
                response = self.upload_audio(audio_bytes, filename)

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

                # Add assistant response with context info
                assistant_message = {
                    "role": "assistant",
                    "content": response["response"],
                    "timestamp": time.time(),
                    "intent": response.get("intent"),
                    "confidence": response.get("confidence"),
                    "response_type": response.get("response_type"),
                    "data": response.get("data"),
                    "context_aware": response.get("context_aware", False),
                    "turn_id": response.get("turn_id")
                }

                st.session_state.messages.append(assistant_message)

                # Show success message with context info
                success_msg = f"✅ **Transcription:** {transcription}"
                if response.get("context_aware"):
                    success_msg += "\n🧠 **Processed with conversation context**"

                st.success(success_msg)

                # Display response immediately
                with st.chat_message("assistant"):
                    st.write(response["response"])

                    if response.get("context_aware"):
                        st.caption("🧠 Context-aware response")

                    # Show additional data if available
                    if response.get("data"):
                        self._render_message_data(response["data"], response.get("response_type"))

                # Increment tab key to refresh interface
                st.session_state.voice_tab_key += 1

                # Rerun to update chat
                time.sleep(1)
                st.rerun()

            else:
                st.session_state.voice_processing = False
                error_msg = response.get("error", "Unknown error") if response else "No response from server"
                st.error(f"❌ Failed to process audio: {error_msg}")

        except Exception as e:
            st.session_state.voice_processing = False
            st.error(f"❌ Audio processing error: {str(e)}")

    def _numpy_to_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        try:
            import wave
            import io

            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Convert to 16-bit PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Create WAV in memory
            wav_io = io.BytesIO()

            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            wav_io.seek(0)
            return wav_io.read()

        except Exception as e:
            st.error(f"Failed to convert audio: {e}")
            return b""

    def _render_message_data(self, data: Any, response_type: str):
        """Render additional message data with context awareness."""
        if response_type == "image" and data:
            if data.get("success") and data.get("image_data"):
                try:
                    image_data = data["image_data"]
                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption="Generated Image Description", width=400)

                        if "metadata" in data and "detailed_description" in data["metadata"]:
                            enhanced_desc = data["metadata"]["detailed_description"]
                            if enhanced_desc and len(enhanced_desc) > 50:
                                with st.expander("📝 View Full Description"):
                                    st.write(enhanced_desc)

                except Exception as e:
                    st.error(f"Could not display image: {e}")
            else:
                st.warning("Image generation failed")

        elif response_type == "text" and isinstance(data, list):
            if data:
                st.subheader("🔍 Search Results")
                for i, result in enumerate(data[:3], 1):
                    with st.expander(f"{i}. {result.get('title', 'No title')}", expanded=i == 1):
                        st.write(result.get('snippet', 'No description available'))
                        if result.get('url'):
                            st.markdown(f"🔗 [Read more]({result['url']})")
                        if result.get('date'):
                            st.caption(f"Published: {result['date']}")

    def render_context_insights(self):
        """Render context insights panel (optional feature)."""
        if st.session_state.show_context_details and st.session_state.context_stats:
            st.markdown("---")
            st.subheader("🧠 Conversation Insights")

            context_stats = st.session_state.context_stats

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Conversation Turns",
                    context_stats.get("turn_count", 0),
                    help="Number of back-and-forth exchanges"
                )

            with col2:
                st.metric(
                    "Context Items",
                    context_stats.get("context_items_count", 0),
                    help="Pieces of information remembered"
                )

            with col3:
                preferences = context_stats.get("user_preferences_count", 0)
                st.metric(
                    "Preferences Learned",
                    preferences,
                    help="User preferences discovered"
                )

            # Show current topic if available
            if context_stats.get("current_topic"):
                st.info(f"**Current Topic:** {context_stats['current_topic']}")

    def render_conversation_summary(self):
        """Render conversation summary panel."""
        if len(st.session_state.messages) > 3:  # Only show if there's a meaningful conversation
            with st.expander("📋 Conversation Summary"):
                # Count message types
                user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
                assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
                context_aware_responses = [m for m in assistant_messages if m.get("context_aware")]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Your Messages", len(user_messages))

                with col2:
                    st.metric("SONA Responses", len(assistant_messages) - 1)  # Exclude greeting

                with col3:
                    st.metric("Context-Aware", len(context_aware_responses))

                # Show conversation timeline
                st.markdown("**Recent Activity:**")
                recent_messages = st.session_state.messages[-6:]  # Last 6 messages

                for msg in recent_messages:
                    if msg["role"] != "assistant" or msg["content"] != SONA_PERSONA["greeting"]:
                        timestamp = time.strftime("%H:%M", time.localtime(msg["timestamp"]))
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        context_icon = "🧠" if msg.get("context_aware") else ""
                        audio_icon = "🎤" if msg.get("input_type") == "audio" else ""

                        preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                        st.caption(f"{timestamp} {role_icon}{context_icon}{audio_icon} {preview}")

    def render_tips_panel(self):
        """Render tips and help panel."""
        with st.expander("💡 Tips for Better Conversations"):
            st.markdown("""
            **Getting the Most from Context-Aware Conversations:**
            
            🧠 **Context Features:**
            - I remember our previous discussions
            - I can reference earlier topics naturally
            - I learn your preferences over time
            
            💬 **Conversation Tips:**
            - Ask follow-up questions - I'll remember the context
            - Reference previous topics - "Tell me more about that"
            - Build on our conversation naturally
            
            🎤 **Voice Input:**
            - Use voice for natural, flowing conversations
            - I process audio with full conversation context
            - Great for complex, multi-part questions
            
            🔍 **Enhanced Search:**
            - My web searches use our conversation context
            - Ask related questions for better results
            - I remember what we've searched for before
            
            🎨 **Image Generation:**
            - I maintain style consistency across requests
            - Reference previous images: "Make another like that"
            - Build on themes from our conversation
            """)

    def render_session_management(self):
        """Render session management panel."""
        with st.expander("📊 Session Management"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Current Session:** {st.session_state.session_id[:12]}...")
                if st.session_state.context_stats:
                    created_at = st.session_state.context_stats.get("created_at")
                    if created_at:
                        start_time = time.strftime("%H:%M:%S", time.localtime(created_at))
                        st.write(f"**Started:** {start_time}")

            with col2:
                # Export conversation
                if st.button("📥 Export Chat", help="Download conversation history"):
                    self._export_conversation()

                # Session settings
                if st.button("⚙️ Session Settings", help="Advanced session options"):
                    self._show_session_settings()

    def _export_conversation(self):
        """Export conversation to downloadable format."""
        try:
            # Create conversation export
            export_data = {
                "session_id": st.session_state.session_id,
                "export_time": time.time(),
                "messages": st.session_state.messages,
                "context_stats": st.session_state.context_stats
            }

            # Convert to JSON
            json_str = json.dumps(export_data, indent=2, default=str)

            # Create download
            st.download_button(
                label="📥 Download Conversation",
                data=json_str,
                file_name=f"sona_conversation_{st.session_state.session_id[:8]}.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Failed to export conversation: {e}")

    def _show_session_settings(self):
        """Show session settings modal."""
        with st.container():
            st.markdown("**Session Settings:**")

            # Context settings
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔄 Reload Context"):
                    self.get_session_context()
                    st.success("Context reloaded!")

            with col2:
                if st.button("🧹 Cleanup Old Sessions"):
                    self._cleanup_old_sessions()

    def _cleanup_old_sessions(self):
        """Trigger cleanup of old sessions."""
        try:
            response = requests.post(f"{self.backend_url}/api/v1/context/cleanup", timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    cleaned = result.get("cleaned_contexts", 0)
                    st.success(f"Cleaned up {cleaned} old sessions!")
                else:
                    st.error("Cleanup failed")
            else:
                st.error("Cleanup request failed")

        except Exception as e:
            st.error(f"Cleanup failed: {e}")

    def render_debug_panel(self):
        """Render debug information panel (for development)."""
        if st.session_state.get("debug_mode", False):
            with st.expander("🔧 Debug Information"):
                st.markdown("**Session State:**")
                st.json({
                    "session_id": st.session_state.session_id,
                    "message_count": len(st.session_state.messages),
                    "voice_processing": st.session_state.voice_processing,
                    "context_stats": st.session_state.context_stats,
                    "backend_health": st.session_state.backend_health is not None
                })

                if st.button("Toggle Debug Mode"):
                    st.session_state.debug_mode = False
                    st.rerun()
        else:
            # Hidden debug toggle
            if st.button("🔧", help="Enable debug mode"):
                st.session_state.debug_mode = True
                st.rerun()

    def run(self):
        """Run the Streamlit application with context features."""
        # Custom CSS for better context visualization
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .context-indicator {
            background-color: #e8f4f8;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            color: #1f77b4;
            margin-top: 0.25rem;
        }
        .voice-indicator {
            background-color: #f0e8ff;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            color: #6a0dad;
            margin-top: 0.25rem;
        }
        .session-banner {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007bff;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Main layout
        self.render_context_sidebar()

        # Main content area
        self.render_main_chat()

        # Additional panels
       # self.render_context_insights()


# Main application entry point
def main():
    """Main application entry point with context management."""
    try:
        app = SONAStreamlitAppWithContext()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.markdown("Please refresh the page or check the backend connection.")


if __name__ == "__main__":
    main()
