from .utils import load_environment, get_env_variable
from .exceptions import GroqAPIKeyMissingError

def get_api_key() -> str:
    """
    Retrieve the Groq API key from the environment.
    
    Returns:
        str: The Groq API key.
    
    Raises:
        GroqAPIKeyMissingError: If the API key is not found in the environment.
    """
    load_environment()
    api_key = get_env_variable('GROQ_API_KEY')
    if not api_key:
        raise GroqAPIKeyMissingError("GROQ_API_KEY not found in environment variables or .env file")
    return api_key