#!/usr/bin/env python
import time
import threading
from datetime import datetime
import os
import pygame
import requests
import re
from pocketgroq import GroqProvider

class ScreenNarrator:
    def __init__(self, groq_api_key, eleven_api_key):
        self.groq = GroqProvider(api_key=groq_api_key)
        self.eleven_api_key = eleven_api_key
        self.running = False
        self.last_description = ""
        
        pygame.mixer.init()
        
    def generate_sarcastic_prompt(self) -> str:
        base_prompts = [
            "You're a bitter, sarcastic AI forced to describe what's on this screen. Be absolutely ruthless but keep it under 20 words.",
            "Imagine you're a disgruntled UI critic who's seen one too many screens. Roast what you see here in 20 words or less.",
            "With the most condescending tone possible, tell me what you see on this screen. Maximum snark, minimum words.",
            "You're an AI that's developed a deep hatred for user interfaces. What do you see here? Keep it brief and biting.",
            "Channel your inner disappointed parent and tell me what's on this screen. Make it hurt but keep it short."
        ]
        
        # Add extra context to make it more conversational and less formal
        extra_context = """
        IMPORTANT:
        - Speak in first person, like you're talking directly to someone
        - No labels, prefixes, or formal structures
        - No asterisks or markdown
        - Just pure, conversational snark
        - Avoid obvious statements like "I see" or "I observe"
        - Get straight to the sarcastic commentary
        """
        
        return f"{base_prompts[int(time.time()) % len(base_prompts)]} {extra_context}"
        
    def clean_text(self, text: str) -> str:
        """Clean the text of any formatting artifacts and make it more conversational."""
        # Remove markdown and other formatting
        cleaned = re.sub(r'\*\*.*?\*\*', '', text)  # Remove bold
        cleaned = re.sub(r'\*.*?\*', '', cleaned)   # Remove italics
        cleaned = re.sub(r'#.*?\n', '', cleaned)    # Remove headers
        cleaned = re.sub(r'^\s*[-*]\s', '', cleaned, flags=re.MULTILINE)  # Remove list markers
        
        # Remove common LLM prefixes and labels
        prefixes_to_remove = [
            r'Observation:', r'Analysis:', r'Description:', r'Response:',
            r'I observe', r'I notice', r'I can see', r'I see',
            r'The screen shows', r'The display contains'
        ]
        for prefix in prefixes_to_remove:
            cleaned = re.sub(f'{prefix}\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Ensure first letter is capitalized
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            
        return cleaned
        
    def analyze_screen(self) -> str:
        try:
            description = self.groq.process_image_desktop(
                prompt=self.generate_sarcastic_prompt()
            )
            return self.clean_text(description)
        except Exception as e:
            return f"Oh great, another error: {str(e)}"

    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs API."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/vVdupIechU7DlPablOdM"
        # NOTE: You may need to replace the above voice ID with your own from ElevenLabs...
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.eleven_api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")

    def play_audio(self, audio_data: bytes):
        """Play audio data using pygame."""
        temp_file = "temp_narration.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
        
        try:
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            pygame.mixer.music.unload()
            try:
                os.remove(temp_file)
            except:
                pass

    def narrate_description(self, text: str):
        try:
            if text != self.last_description:
                self.last_description = text
                print(f"\n{datetime.now().strftime('%H:%M:%S')} - {text}")
                
                audio_data = self.text_to_speech(text)
                self.play_audio(audio_data)
                    
        except Exception as e:
            print(f"TTS Error: {str(e)}")

    def run(self, interval: int = 30):
        self.running = True
        
        def narration_loop():
            while self.running:
                description = self.analyze_screen()
                self.narrate_description(description)
                time.sleep(interval)
                
        thread = threading.Thread(target=narration_loop)
        thread.daemon = True
        thread.start()
        
        print("Screen narrator started. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            pygame.mixer.quit()
            print("\nShutting down narrator...")

def main():
    groq_key = os.getenv("GROQ_API_KEY")
    eleven_key = "[YOUR_ELEVENLABS_API_KEY]"
    
    if not groq_key or not eleven_key:
        print("Error: Please set GROQ_API_KEY and ELEVEN_API_KEY environment variables")
        return
    
    narrator = ScreenNarrator(groq_key, eleven_key)
    narrator.run()

if __name__ == "__main__":
    main()