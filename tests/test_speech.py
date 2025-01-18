#!/usr/bin/env python
import pytest
import os
import tempfile
import wave
from pocketgroq import GroqProvider

def create_test_wav(filename: str, duration_secs: int = 1, sample_rate: int = 44100):
    """Create a test WAV file with silence."""
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = b'\x00' * (duration_secs * sample_rate * 2)
        wav_file.writeframes(frames)

def test_basic_transcription():
    """Test basic audio transcription with a sample file."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment")
    
    provider = GroqProvider(api_key=api_key)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        create_test_wav(temp_path)
    
    try:
        # Test transcription with English-optimized model
        response = provider.transcribe_audio(
            audio_file=temp_path,
            language="en",
            model="distil-whisper-large-v3-en"  # Explicitly use English model
        )
        
        print("\nTranscription Response:")
        print(response)
        
        assert isinstance(response, str)
        assert len(response) > 0
    finally:
        os.unlink(temp_path)

def test_audio_translation():
    """Test audio translation to English."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment")
    
    provider = GroqProvider(api_key=api_key)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        create_test_wav(temp_path)
    
    try:
        # Get available speech models
        speech_models = provider.speech_processor.get_speech_models(provider)
        print("\nAvailable speech models:", speech_models)
        
        # Look specifically for whisper-large-v3 which supports translation
        translation_model = next((m for m in speech_models if m == 'whisper-large-v3'), None)
        
        if not translation_model:
            pytest.skip("whisper-large-v3 model not available")
            
        # Test translation using the correct model
        response = provider.translate_audio(
            audio_file=temp_path,
            model="whisper-large-v3",  # Explicitly use translation-capable model
            prompt="This is a French conversation about cooking."
        )
        
        print("\nTranslation Response:")
        print(response)
        
        assert isinstance(response, str)
        assert len(response) > 0
    finally:
        os.unlink(temp_path)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])