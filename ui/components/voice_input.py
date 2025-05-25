"""
Enhanced Voice input components for SONA AI Assistant with real-time recording.
Handles both audio file upload and live microphone recording.
"""
import streamlit as st
import io
import os
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from loguru import logger

# Try to import audio recording libraries
try:
    import sounddevice as sd
    import wave

    REALTIME_AUDIO_AVAILABLE = True
except ImportError:
    REALTIME_AUDIO_AVAILABLE = False
    logger.warning("sounddevice not available - real-time recording disabled")

from config.settings import get_settings


class EnhancedVoiceInputComponent:
    """Enhanced voice input component with real-time recording capabilities."""

    def __init__(self):
        """Initialize enhanced voice input component."""
        self.settings = get_settings()
        self.supported_formats = self.settings.get_allowed_audio_formats()
        self.max_file_size = self.settings.max_file_size
        self.sample_rate = self.settings.audio_sample_rate
        self.channels = self.settings.audio_channels

        # Real-time recording state
        self.chunk_duration = 0.1  # 100ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.max_recording_duration = 30  # 30 seconds max

    def check_audio_permissions(self) -> Dict[str, Any]:
        """Check audio system and permissions."""
        if not REALTIME_AUDIO_AVAILABLE:
            return {
                "available": False,
                "error": "sounddevice library not installed",
                "install_command": "pip install sounddevice"
            }

        try:
            # Check available devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]

            if not input_devices:
                return {
                    "available": False,
                    "error": "No input audio devices found"
                }

            # Quick test recording
            test_duration = 0.1
            test_recording = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()

            return {
                "available": True,
                "input_devices": len(input_devices),
                "default_device": sd.query_devices(kind='input')['name'],
                "sample_rate": self.sample_rate
            }

        except Exception as e:
            return {
                "available": False,
                "error": f"Audio system error: {str(e)}"
            }

    def record_audio_realtime(self, duration: float) -> Optional[bytes]:
        """
        Record audio in real-time for specified duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            WAV audio data as bytes or None if failed
        """
        if not REALTIME_AUDIO_AVAILABLE:
            return None

        try:
            logger.info(f"Starting real-time recording for {duration} seconds")

            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )

            # Wait for recording to complete
            sd.wait()

            # Convert to WAV format
            wav_data = self._numpy_to_wav(recording)

            logger.info(f"Recording completed: {len(wav_data)} bytes")
            return wav_data

        except Exception as e:
            logger.error(f"Real-time recording failed: {e}")
            return None

    def _numpy_to_wav(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes."""
        try:
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Convert to 16-bit PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Create WAV in memory
            wav_io = io.BytesIO()

            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            wav_io.seek(0)
            return wav_io.read()

        except Exception as e:
            logger.error(f"Failed to convert numpy to WAV: {e}")
            return b""

    def get_microphone_level(self) -> float:
        """Get current microphone input level."""
        if not REALTIME_AUDIO_AVAILABLE:
            return 0.0

        try:
            # Quick sample to check current level
            sample_duration = 0.1
            recording = sd.rec(
                int(sample_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()

            # Calculate RMS level
            rms = np.sqrt(np.mean(recording ** 2))
            return min(rms * 10, 1.0)  # Scale and clamp

        except:
            return 0.0

    def render_enhanced_voice_interface(self, process_audio_callback: Callable[[bytes, str], None]):
        """
        Render enhanced voice interface with both file upload and real-time recording.

        Args:
            process_audio_callback: Function to call when audio is ready for processing
        """
        st.markdown("### 🎤 Voice Input Options")

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Real-time Recording"])

        with tab1:
            self._render_file_upload_tab(process_audio_callback)

        with tab2:
            self._render_realtime_tab(process_audio_callback)

    def _render_file_upload_tab(self, process_audio_callback: Callable[[bytes, str], None]):
        """Render file upload tab."""
        st.markdown("Upload an audio file to transcribe:")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=self.supported_formats,
            help=f"Supported: {', '.join(self.supported_formats)}. Max: {self.max_file_size // (1024 * 1024)}MB",
            key="file_upload_tab"
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
            if st.button("🎯 Process Audio File", key="process_file_btn"):
                is_valid, error_msg = self.validate_audio_file(uploaded_file.getvalue(), uploaded_file.name)

                if is_valid:
                    process_audio_callback(uploaded_file.getvalue(), uploaded_file.name)
                else:
                    st.error(f"❌ {error_msg}")

    def _render_realtime_tab(self, process_audio_callback: Callable[[bytes, str], None]):
        """Render real-time recording tab."""
        # Check audio system
        audio_status = self.check_audio_permissions()

        if not audio_status["available"]:
            st.error(f"❌ **Real-time recording not available:** {audio_status['error']}")

            if "install_command" in audio_status:
                st.code(audio_status["install_command"])
                st.markdown("After installation, restart the application.")

            return

        # Audio system info
        with st.expander("🔧 Audio System Info"):
            st.success(f"✅ Audio system ready")
            st.write(f"**Input devices:** {audio_status['input_devices']}")
            st.write(f"**Default device:** {audio_status['default_device']}")
            st.write(f"**Sample rate:** {audio_status['sample_rate']} Hz")

        # Recording controls
        st.markdown("**Quick Recording:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔴 Record 5s", key="record_5s"):
                self._quick_record(5, process_audio_callback)

        with col2:
            if st.button("🔴 Record 10s", key="record_10s"):
                self._quick_record(10, process_audio_callback)

        with col3:
            if st.button("🔴 Record 15s", key="record_15s"):
                self._quick_record(15, process_audio_callback)

        # Custom duration recording
        st.markdown("**Custom Duration:**")

        col1, col2 = st.columns([2, 1])

        with col1:
            duration = st.slider(
                "Recording duration (seconds)",
                min_value=1,
                max_value=self.max_recording_duration,
                value=10,
                key="custom_duration"
            )

        with col2:
            if st.button(f"🔴 Record {duration}s", key="record_custom"):
                self._quick_record(duration, process_audio_callback)

        # Microphone level indicator
        st.markdown("**Microphone Level:**")
        if st.button("🔍 Check Mic Level", key="check_mic"):
            level = self.get_microphone_level()
            level_percent = int(level * 100)

            if level_percent > 50:
                st.success(f"🟢 Good level: {level_percent}%")
            elif level_percent > 20:
                st.warning(f"🟡 Moderate level: {level_percent}%")
            else:
                st.error(f"🔴 Low level: {level_percent}% - Speak louder or move closer to mic")

            st.progress(level)

    def _quick_record(self, duration: float, process_audio_callback: Callable[[bytes, str], None]):
        """Perform quick recording for specified duration."""
        try:
            # Show countdown
            countdown_placeholder = st.empty()
            progress_bar = st.progress(0.0)

            with st.spinner(f"🔴 Recording for {duration} seconds..."):
                # Countdown
                for i in range(3, 0, -1):
                    countdown_placeholder.warning(f"Starting in {i}...")
                    time.sleep(1)

                countdown_placeholder.info("🔴 Recording now!")

                # Record with progress updates
                start_time = time.time()
                recording = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32
                )

                # Update progress
                while not recording.flags.writeable:
                    elapsed = time.time() - start_time
                    progress = min(elapsed / duration, 1.0)
                    progress_bar.progress(progress)
                    time.sleep(0.1)

                sd.wait()
                progress_bar.progress(1.0)
                countdown_placeholder.success("✅ Recording completed!")

            # Convert to WAV
            wav_data = self._numpy_to_wav(recording)

            if wav_data:
                filename = f"recording_{int(time.time())}.wav"

                # Show audio player
                st.audio(wav_data, format="audio/wav")

                # Process the recording
                process_audio_callback(wav_data, filename)
            else:
                st.error("❌ Failed to process recording")

        except Exception as e:
            logger.error(f"Quick recording failed: {e}")
            st.error(f"❌ Recording failed: {str(e)}")

    def validate_audio_file(self, audio_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file.

        Args:
            audio_bytes: Audio file content
            filename: Original filename

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            if len(audio_bytes) > self.max_file_size:
                return False, f"File too large. Max: {self.max_file_size // (1024 * 1024)}MB"

            if len(audio_bytes) < 100:
                return False, "File too small or empty"

            # Check extension
            if not filename:
                return False, "Filename not provided"

            file_ext = os.path.splitext(filename)[1].lower().lstrip('.')
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format '{file_ext}'. Allowed: {', '.join(self.supported_formats)}"

            return True, None

        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False, f"Validation error: {str(e)}"

    def render_tips_and_troubleshooting(self):
        """Render tips and troubleshooting section."""
        with st.expander("💡 Tips & Troubleshooting"):
            st.markdown("""
            **For Best Results:**
            - 🎯 Speak clearly at normal pace
            - 🔇 Record in quiet environment
            - 🎤 Keep microphone 6-12 inches away
            - 🔊 Check microphone level before recording

            **Troubleshooting:**
            - **No audio detected:** Check microphone permissions in browser
            - **Poor quality:** Ensure microphone is working and not muted
            - **Recording fails:** Try refreshing page and allowing microphone access
            - **No input devices:** Check if microphone is connected and recognized by system

            **File Upload Issues:**
            - **Unsupported format:** Convert to WAV, MP3, M4A, or FLAC
            - **File too large:** Compress audio or reduce duration
            - **Upload fails:** Check internet connection and try again
            """)

    def render_audio_settings(self):
        """Render audio settings panel."""
        with st.expander("⚙️ Audio Settings"):
            st.write(f"**Sample Rate:** {self.sample_rate} Hz")
            st.write(f"**Channels:** {self.channels} (Mono)")
            st.write(f"**Supported Formats:** {', '.join(self.supported_formats)}")
            st.write(f"**Max File Size:** {self.max_file_size // (1024 * 1024)} MB")
            st.write(f"**Max Recording Duration:** {self.max_recording_duration} seconds")

            if REALTIME_AUDIO_AVAILABLE:
                st.success("✅ Real-time recording available")
            else:
                st.warning("⚠️ Real-time recording not available")
