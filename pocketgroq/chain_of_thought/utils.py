# pocketgroq/chain_of_thought/utils.py

import re

def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks or unwanted content.
    """
    # Remove potentially harmful characters or patterns
    sanitized = re.sub(r'[<>]', '', text)
    return sanitized.strip()