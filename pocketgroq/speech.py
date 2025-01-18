import os
import mimetypes
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """Handles speech-related functionality for the GroqProvider class."""
    
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    SUPPORTED_FORMATS = {'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm'}
    MIN_LENGTH = 0.01  # seconds
    MIN_BILLED_LENGTH = 10  # seconds
    
    @classmethod
    def get_speech_models(cls, groq_provider) -> List[str]:
        """
        Get list of available speech models from Groq API.
        
        Args:
            groq_provider: Instance of GroqProvider to use for API calls
            
        Returns:
            List[str]: List of model IDs that support speech processing
        """
        try:
            all_models = groq_provider.get_available_models()
            speech_models = [
                model['id'] for model in all_models 
                if 'whisper' in model['id'].lower()
            ]
            return speech_models
        except Exception as e:
            logger.error(f"Failed to fetch speech models: {str(e)}")
            return []
            
    @staticmethod
    def validate_audio_file(file_path: str) -> None:
        """
        Validate audio file meets requirements.
        
        Args:
            file_path: Path to audio file
            
        Raises:
            ValueError: If file is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > SpeechProcessor.MAX_FILE_SIZE:
            raise ValueError(
                f"Audio file size ({file_size} bytes) exceeds maximum allowed size "
                f"({SpeechProcessor.MAX_FILE_SIZE} bytes)"
            )
            
        # Check file format
        mime_type = mimetypes.guess_type(file_path)[0]
        if not mime_type:
            raise ValueError(f"Could not determine file type for: {file_path}")
            
        file_ext = mime_type.split('/')[-1]
        if file_ext not in SpeechProcessor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. Must be one of: "
                f"{', '.join(SpeechProcessor.SUPPORTED_FORMATS)}"
            )