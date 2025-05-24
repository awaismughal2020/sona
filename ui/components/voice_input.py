"""
Complete Voice input components for SONA AI Assistant.
Handles audio file upload, validation, and voice input interface.
"""
import streamlit as st
import io
import os
from typing import Optional, Tuple, Dict, Any
from loguru import logger
import time

from config.settings import get_settings


class VoiceInputComponent:
    """Voice input component for SONA."""

    def __init__(self):
        """Initialize voice input component."""
        self.settings = get_settings()
        self.supported_formats = self.settings.get_allowed_audio_formats()
        self.max_file_size = self.settings.max_file_size

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
                return False, f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"

            if len(audio_bytes) < 100:
                return False, "File too small or empty"

            # Check file extension
            if not filename:
                return False, "Filename not provided"

            file_ext = os.path.splitext(filename)[1].lower().lstrip('.')
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format '{file_ext}'. Allowed: {', '.join(self.supported_formats)}"

            return True, None

        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False, f"Validation error: {str(e)}"

    def render_file_upload(self) -> Optional[Tuple[bytes, str]]:
        """
        Render audio file upload interface.

        Returns:
            Tuple of (audio_bytes, filename) if file uploaded, None otherwise
        """
        try:
            st.markdown("Upload an audio file to interact with SONA using your voice.")

            # File upload widget
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=self.supported_formats,
                help=f"Supported formats: {', '.join(self.supported_formats)}. Max size: {self.max_file_size // (1024 * 1024)}MB",
                key="voice_file_upload"
            )

            if uploaded_file is not None:
                # Get file info
                file_size = len(uploaded_file.getvalue())

                # Validate file size
                if file_size > self.max_file_size:
                    st.error(f"❌ File too large! Maximum size: {self.max_file_size // (1024 * 1024)}MB")
                    return None

                # Display file info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📁 **File:** {uploaded_file.name}")
                with col2:
                    st.info(f"📊 **Size:** {file_size / 1024:.1f} KB")

                # Audio player
                st.audio(uploaded_file.getvalue(), format='audio/wav')

                return uploaded_file.getvalue(), uploaded_file.name

            return None

        except Exception as e:
            logger.error(f"Error in file upload interface: {e}")
            st.error(f"Error with file upload: {str(e)}")
            return None

    def render_complete_voice_interface(self, app_instance):
        """
        Render complete voice input interface.

        Args:
            app_instance: Reference to the main app instance
        """
        st.markdown("### 🎤 Voice Input")

        # File upload section
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=self.supported_formats,
            help=f"Supported formats: {', '.join(self.supported_formats)}. Max size: {self.max_file_size // (1024 * 1024)}MB"
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
            st.audio(uploaded_file.getvalue(), format='audio/wav')

            # Process button
            if st.button("🎤 Process Audio", key="process_audio_button"):
                # Validate file
                is_valid, error_msg = self.validate_audio_file(uploaded_file.getvalue(), uploaded_file.name)

                if not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    # Call the app's audio processing method
                    if hasattr(app_instance, '_process_audio_file'):
                        app_instance._process_audio_file(uploaded_file.getvalue(), uploaded_file.name)
                    else:
                        st.error("Audio processing not available")

        # Tips section
        self.render_tips()

    def render_tips(self):
        """Render tips for better audio experience."""
        with st.expander("💡 Tips for Better Voice Recognition"):
            st.markdown(f"""
            **For best results:**
            - 🎯 Speak clearly and at normal pace
            - 🔇 Record in a quiet environment
            - ⏱️ Keep recordings under 2 minutes
            - 🎤 Use good quality microphone if possible
            
            **Supported formats:** {', '.join(self.supported_formats)}
            
            **File size limit:** {self.max_file_size // (1024 * 1024)}MB
            
            **Example commands:**
            - "What's the weather in Islamabad?"
            - "Generate an image of a sunset"
            - "Tell me about cryptocurrency prices"
            """)

    def render_processing_status(self, status: str = "processing"):
        """Render processing status."""
        status_messages = {
            "processing": "🔄 Processing audio file...",
            "transcribing": "🎯 Converting speech to text...",
            "analyzing": "🧠 Understanding your message...",
            "complete": "✅ Processing complete!"
        }

        message = status_messages.get(status, f"🔄 {status}...")

        if status == "complete":
            st.success(message)
        else:
            with st.spinner(message):
                time.sleep(0.5)  # Small delay for visual effect

    def render_transcription_result(self, transcription: str, confidence: float = None):
        """
        Render transcription result.

        Args:
            transcription: Transcribed text
            confidence: Optional confidence score
        """
        try:
            st.markdown("### 📝 Transcription Result")

            # Confidence indicator
            if confidence is not None:
                conf_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                st.markdown(f"**Confidence:** {conf_color} {confidence:.1%}")

            # Transcription text
            if transcription.strip():
                st.markdown(f"**You said:** *\"{transcription}\"*")
            else:
                st.warning("⚠️ No speech detected in the audio file.")

        except Exception as e:
            logger.error(f"Error rendering transcription result: {e}")
            st.error("Error displaying transcription result")

    def render_error_recovery(self, error_type: str = "general"):
        """
        Render error recovery suggestions.

        Args:
            error_type: Type of error
        """
        error_messages = {
            "upload": {
                "title": "Upload Error",
                "suggestions": [
                    "Check if file format is supported",
                    "Ensure file size is under the limit",
                    "Try uploading a different audio file",
                    "Check your internet connection"
                ]
            },
            "processing": {
                "title": "Processing Error",
                "suggestions": [
                    "The audio file might be corrupted",
                    "Try uploading the file again",
                    "Use a different audio format",
                    "Check if the file contains actual speech"
                ]
            },
            "general": {
                "title": "Error Occurred",
                "suggestions": [
                    "Try refreshing the page",
                    "Upload a different audio file",
                    "Check your internet connection",
                    "Contact support if the problem persists"
                ]
            }
        }

        error_info = error_messages.get(error_type, error_messages["general"])

        st.error(f"❌ {error_info['title']}")

        with st.expander("💡 Troubleshooting Suggestions"):
            for suggestion in error_info["suggestions"]:
                st.write(f"• {suggestion}")

    def get_supported_formats_info(self) -> Dict[str, str]:
        """Get information about supported audio formats."""
        format_info = {
            "wav": "WAV - Uncompressed, best quality (recommended)",
            "mp3": "MP3 - Compressed, good compatibility",
            "m4a": "M4A - Apple's audio format, good quality",
            "flac": "FLAC - Lossless compression, high quality"
        }

        return {fmt: format_info.get(fmt, f"{fmt.upper()} format")
                for fmt in self.supported_formats}