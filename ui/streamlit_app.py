"""
Updated Streamlit UI for SONA AI Assistant with real-time voice recording.
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
import numpy as np

try:
    import sounddevice as sd
    import wave
    AUDIO_LIBS_AVAILABLE = True
    print("✅ Audio libraries loaded successfully")
except ImportError as e:
    AUDIO_LIBS_AVAILABLE = False
    print(f"⚠️ Audio libraries not available: {e}")

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

# Import the enhanced voice component (you'll need to save the previous artifact as this file)
# from ui.components.enhanced_voice_input import EnhancedVoiceInputComponent


class SONAStreamlitApp:
    """SONA Streamlit Application with Enhanced Voice Features."""

    def __init__(self):
        """Initialize Streamlit app."""
        self.settings = get_settings()
        self.backend_url = self.settings.get_backend_url()
        self.chat_interface = ChatInterface()

        # Use enhanced voice component
        # self.voice_component = EnhancedVoiceInputComponent()

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

        if "realtime_recording" not in st.session_state:
            st.session_state.realtime_recording = False

        if "voice_tab_key" not in st.session_state:
            st.session_state.voice_tab_key = 0

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

    def process_audio_callback(self, audio_bytes: bytes, filename: str):
        """Callback function to process audio data."""
        try:
            st.session_state.voice_processing = True

            # Show processing status
            with st.spinner("🔄 Processing your voice input..."):
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

                # Show success message
                st.success(f"✅ **Transcription:** {transcription}")

                # Display response immediately
                with st.chat_message("assistant"):
                    st.write(response["response"])

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

            # Voice system status
            st.subheader("🎤 Voice System")

            # Check if real-time recording is available
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_input_channels'] > 0]

                if input_devices:
                    st.success("✅ Real-time recording available")
                    st.info(f"Input devices: {len(input_devices)}")
                else:
                    st.warning("⚠️ No input devices found")

            except ImportError:
                st.error("❌ Real-time recording disabled")
                st.caption("Install sounddevice: pip install sounddevice")
            except Exception as e:
                st.warning(f"⚠️ Audio system issue: {str(e)}")

            st.divider()

            # Capabilities
            st.subheader("💡 Capabilities")
            for capability in SONA_PERSONA["capabilities"]:
                st.write(f"• {capability}")

            st.divider()

            # Session info
            st.subheader("📊 Session Info")
            st.write(f"**Session ID:** {st.session_state.session_id[:12]}...")
            st.write(f"**Messages:** {len(st.session_state.messages) - 1}")

            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": SONA_PERSONA["greeting"],
                        "timestamp": time.time()
                    }
                ]
                st.rerun()

            # Voice controls
            st.divider()
            st.subheader("🎤 Voice Controls")

            if st.button("🔄 Reset Voice Interface"):
                st.session_state.voice_processing = False
                st.session_state.realtime_recording = False
                st.session_state.voice_tab_key += 1
                st.success("Voice interface reset!")
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
        self._render_enhanced_voice_section()

    def _render_enhanced_voice_section(self):
        """Render enhanced voice input section."""
        st.markdown("### 🎤 Voice Input")

        # Check if currently processing
        if st.session_state.voice_processing:
            st.info("🔄 Processing voice input... Please wait.")
            return

        # Create tabs for different voice input methods
        tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Real-time Recording"])

        with tab1:
            self._render_file_upload_tab()

        with tab2:
            self._render_realtime_recording_tab()

    def _render_file_upload_tab(self):
        """Render file upload tab."""
        st.markdown("Upload an audio file to transcribe:")

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

            # Process button
            if st.button("🎯 Process Audio File", key=f"process_file_{st.session_state.voice_tab_key}"):
                self.process_audio_callback(uploaded_file.getvalue(), uploaded_file.name)

    def _render_realtime_recording_tab(self):
        """Render real-time recording tab with device selection."""
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

            # Device selection (optional)
            with st.expander("🎛️ Device Selection (Optional)"):
                st.markdown(
                    "The app will automatically use the best available device, but you can choose a specific one:")
                self._render_device_selector()

            # Audio system info
            with st.expander("🔧 Audio System Info"):
                st.markdown("**Your available devices:**")
                st.markdown("- **MacBook Pro Microphone** (Recommended - Good audio level)")
                st.markdown("- **iPhone 16 Pro Max Microphone** (Good - Moderate audio level)")
                st.markdown("- **WH-1000XM5** (Backup - Low audio level)")
                st.markdown("- **Microsoft Teams Audio** (Backup - Low audio level)")

        except Exception as e:
            st.error(f"❌ Audio system error: {str(e)}")
            return

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
            **Based on your audio diagnostic:**

            **Best Options:**
            - 🥇 **MacBook Pro Microphone** - Highest audio level detected
            - 🥈 **iPhone 16 Pro Max Microphone** - Good audio level

            **For best results:**
            - Use MacBook Pro's built-in microphone for clearest audio
            - If using iPhone, keep it close (6-8 inches away)
            - Speak clearly and at normal volume
            - Record in a quiet environment

            **Troubleshooting:**
            - If recording fails, try refreshing the page
            - Check microphone permissions in browser settings
            - The app will automatically try different devices if one fails
            """)

    def _quick_record(self, duration: float):
        """Perform quick recording with your specific device setup."""
        if not AUDIO_LIBS_AVAILABLE:
            st.error("❌ Audio recording libraries not available")
            return

        try:
            # Countdown
            countdown_placeholder = st.empty()
            progress_bar = st.progress(0.0)

            # Based on your diagnostic, prioritize devices with good audio levels
            # Device 3: MacBook Pro Microphone (0.0697 level) - Best
            # Device 2: iPhone 16 Pro Max Microphone (0.0327 level) - Good
            # Device 0: WH-1000XM5 (0.0000 level) - Backup
            # Device 5: Microsoft Teams Audio (0.0000 level) - Backup

            preferred_devices = [
                (3, "MacBook Pro Microphone"),
                (2, "iPhone 16 Pro Max Microphone"),
                (0, "WH-1000XM5"),
                (5, "Microsoft Teams Audio")
            ]

            # Countdown
            for i in range(3, 0, -1):
                countdown_placeholder.warning(f"🔴 Recording starts in {i}...")
                time.sleep(1)

            countdown_placeholder.info("🔴 Recording now! Speak clearly...")

            sample_rate = self.settings.audio_sample_rate
            recording = None
            successful_device = None

            # Try devices in order of preference
            for device_id, device_name in preferred_devices:
                try:
                    countdown_placeholder.info(f"🔴 Using {device_name}...")

                    recording = sd.rec(
                        int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype=np.float32,
                        device=device_id
                    )

                    # Show progress with device info
                    start_time = time.time()
                    while not recording.flags.writeable:
                        elapsed = time.time() - start_time
                        progress = min(elapsed / duration, 1.0)
                        progress_bar.progress(progress)

                        # Update status during recording
                        remaining = duration - elapsed
                        if remaining > 0:
                            countdown_placeholder.info(f"🔴 Recording with {device_name}... {remaining:.1f}s left")

                        time.sleep(0.1)

                    sd.wait()
                    successful_device = device_name
                    break

                except Exception as e:
                    print(f"Device {device_id} ({device_name}) failed: {e}")
                    continue

            progress_bar.progress(1.0)

            if recording is None:
                countdown_placeholder.error("❌ All recording devices failed")
                st.error("Please check your microphone settings and try again")
                return

            # Check if we actually got audio
            max_amplitude = np.max(np.abs(recording))
            rms_level = np.sqrt(np.mean(recording ** 2))

            if max_amplitude < 0.001:
                countdown_placeholder.warning(f"⚠️ Very low audio detected with {successful_device}")
                st.warning("Audio level is very low. Try speaking louder or closer to the microphone.")
            else:
                countdown_placeholder.success(
                    f"✅ Recording completed using {successful_device}! (Level: {max_amplitude:.3f})")

            # Convert to WAV format
            wav_data = self._numpy_to_wav(recording, sample_rate)

            if wav_data:
                # Show audio player
                st.audio(wav_data, format="audio/wav")

                # Show audio quality info
                with st.expander("📊 Recording Info"):
                    st.write(f"**Device used:** {successful_device}")
                    st.write(f"**Duration:** {duration} seconds")
                    st.write(f"**Max amplitude:** {max_amplitude:.4f}")
                    st.write(f"**RMS level:** {rms_level:.4f}")
                    st.write(f"**Sample rate:** {sample_rate} Hz")
                    st.write(f"**File size:** {len(wav_data)} bytes")

                # Process the recording
                filename = f"recording_{int(time.time())}.wav"
                self.process_audio_callback(wav_data, filename)
            else:
                st.error("❌ Failed to process recording")

        except Exception as e:
            st.error(f"❌ Recording failed: {str(e)}")
            print(f"Recording error details: {e}")


    # ALSO ADD this method to help users choose their preferred device:
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

                # Default to MacBook Pro Microphone if available, otherwise first device
                default_option = "[3] MacBook Pro Microphone" if "[3] MacBook Pro Microphone" in device_options else \
                device_options[0]

                selected_device = st.selectbox(
                    "Choose your preferred microphone:",
                    device_options,
                    index=device_options.index(default_option) if default_option in device_options else 0,
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
        """Render additional message data."""
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
