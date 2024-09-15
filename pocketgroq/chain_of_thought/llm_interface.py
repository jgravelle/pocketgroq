# pocketgroq/chain_of_thought/llm_interface.py

from abc import ABC, abstractmethod
from typing import List

class LLMInterface(ABC):
    """
    Abstract base class for LLM integrations.
    """
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Generate a response from the LLM based on the prompt.
        """
        pass

    @abstractmethod
    def set_api_key(self, api_key: str):
        """
        Set the API key for the LLM service.
        """
        pass