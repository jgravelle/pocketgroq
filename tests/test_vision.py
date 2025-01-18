#!/usr/bin/env python
import pytest
import os
from pocketgroq import GroqProvider

def test_desktop_vision():
    """Test desktop screen capture and analysis."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment")
    
    provider = GroqProvider(api_key=api_key)
    
    # Full screen analysis
    response = provider.process_image_desktop(
        prompt="Describe what you see on the screen. Focus on any visible windows, text, or UI elements."
    )
    
    print("\nDesktop Analysis Response:")
    print(response)
    
    # Verify meaningful response for full screen
    assert len(response) > 50
    assert any(word in response.lower() for word in ['window', 'screen', 'text', 'interface', 'display'])
    
    # Test specific region capture
    region_response = provider.process_image_desktop_region(
        prompt="What do you see in this region of the screen?",
        x1=0,    # Top-left corner
        y1=0,    # Top-left corner
        x2=400,  # Width
        y2=300   # Height
    )
    
    print("\nScreen Region Analysis Response:")
    print(region_response)
    
    # Verify meaningful response for region
    assert len(region_response) > 50
    # The content assertions should be appropriate for what's actually in that region
    assert any(word in region_response.lower() for word in ['area', 'section', 'portion', 'region', 'part'])

def test_real_vision():
    """Test actual vision processing with a real image."""
    # Use a real Groq API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment")
    
    provider = GroqProvider(api_key=api_key)
    
    # Use a real, stable image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    # Process the actual image
    response = provider.process_image(
        prompt="Describe what you see in this image. Be specific.",
        image_source=image_url
    )
    
    # Verify we got a meaningful response
    assert len(response) > 50  # Response should be substantial
    assert "boardwalk" in response.lower() or "path" in response.lower()
    assert "nature" in response.lower() or "trees" in response.lower()
    
    print("\nVision API Response:")
    print(response)

def test_multi_turn_conversation():
    """Test a multi-turn conversation about an image."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in environment")
    
    provider = GroqProvider(api_key=api_key)
    
    # Use same image for consistency
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    # First turn: Ask about the image
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        }
    ]
    
    response1 = provider.process_image_conversation(messages=conversation)
    print("\nFirst Response:")
    print(response1)
    
    # Add the assistant's response to the conversation
    conversation.append({
        "role": "assistant",
        "content": response1
    })
    
    # Second turn: Ask a follow-up question
    conversation.append({
        "role": "user",
        "content": "What materials do you think the boardwalk is made of?"
    })
    
    response2 = provider.process_image_conversation(messages=conversation)
    print("\nSecond Response:")
    print(response2)
    
    # Verify meaningful responses
    assert len(response1) > 50
    assert len(response2) > 50
    assert "wood" in response2.lower() or "wooden" in response2.lower() or "material" in response2.lower()

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])