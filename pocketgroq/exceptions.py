# pocketgroq/exceptions.py

class GroqAPIKeyMissingError(Exception):
    """Raised when the Groq API key is missing."""
    pass

class GroqAPIError(Exception):
    """Raised when there's an error with the Groq API."""
    pass

class OllamaServerNotRunningError(Exception):
    """Raised when the Ollama server is not running but is required for an operation."""
    pass