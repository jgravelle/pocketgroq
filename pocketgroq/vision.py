# pocketgroq/vision.py

import base64
import os
from typing import Dict, Any, List, Union
import mimetypes
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Handles vision-related functionality for the GroqProvider class.
    """
    
    @staticmethod
    def get_vision_models(groq_provider) -> List[str]:
        """
        Get list of available vision models from Groq API.
        
        Args:
            groq_provider: Instance of GroqProvider to use for API calls
            
        Returns:
            List[str]: List of model IDs that support vision capabilities
        """
        try:
            all_models = groq_provider.get_available_models()
            vision_models = [
                model['id'] for model in all_models 
                if 'vision' in model['id'].lower()
            ]
            return vision_models
        except Exception as e:
            logger.error(f"Failed to fetch vision models: {str(e)}")
            return []
    
    MAX_URL_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    MAX_BASE64_IMAGE_SIZE = 4 * 1024 * 1024  # 4MB
    
    SUPPORTED_IMAGE_FORMATS = {
        'image/jpeg', 'image/png', 'image/gif', 
        'image/webp', 'image/bmp', 'image/tiff'
    }

    @classmethod
    def validate_vision_model(cls, model: str, groq_provider) -> bool:
        """
        Validate if the provided model supports vision capabilities.
        
        Args:
            model (str): Model ID to validate
            groq_provider: Instance of GroqProvider to use for API calls
            
        Returns:
            bool: True if model supports vision, False otherwise
        """
        vision_models = cls.get_vision_models(groq_provider)
        return model in vision_models

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encode a local image file to base64.
        
        Args:
            image_path (str): Path to the local image file.
            
        Returns:
            str: Base64 encoded image string with mime type.
            
        Raises:
            ValueError: If image size exceeds limits or format is unsupported.
            FileNotFoundError: If image file doesn't exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check file size before processing
        file_size = os.path.getsize(image_path)
        if file_size > VisionProcessor.MAX_BASE64_IMAGE_SIZE:
            raise ValueError(
                f"Image file size ({file_size} bytes) exceeds maximum allowed size "
                f"({VisionProcessor.MAX_BASE64_IMAGE_SIZE} bytes)"
            )

        # Validate image format
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type not in VisionProcessor.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {mime_type}")

        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf8')
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

    @staticmethod
    def prepare_vision_messages(
        prompt: str,
        image_source: Union[str, Dict[str, str]],
        system_message: str = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for vision API calls.
        
        Args:
            prompt (str): Text prompt to accompany the image
            image_source (Union[str, Dict[str, str]]): Either an image path or URL
            system_message (str, optional): System message to include
            
        Returns:
            List[Dict[str, Any]]: Formatted messages for the API call
        """
        messages = []
        
        # Don't add system message if image is included (per API requirements)
        if system_message and not image_source:
            messages.append({
                "role": "system",
                "content": system_message
            })

        # Prepare user message with text and image
        user_content = [{"type": "text", "text": prompt}]
        
        # Handle image source
        if isinstance(image_source, str):
            if urlparse(image_source).scheme in ['http', 'https']:
                # URL image
                image_url = image_source
            else:
                # Local file path
                image_url = VisionProcessor.encode_image(image_source)
        else:
            # Already formatted image dict
            image_url = image_source.get('url')

        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    @staticmethod
    def validate_image_url(url: str) -> bool:
        """
        Validate if the provided URL is acceptable for vision processing.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)