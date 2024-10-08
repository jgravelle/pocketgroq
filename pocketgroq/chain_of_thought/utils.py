# pocketgroq/chain_of_thought/utils.py

import re
from typing import List

def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks or unwanted content.
    """
    # Remove potentially harmful characters or patterns
    sanitized = re.sub(r'[<>]', '', text)
    return sanitized.strip()

def validate_cot_steps(steps: List[str], min_steps: int = 3) -> bool:
    """
    Validates the extracted Chain-of-Thought steps.

    Args:
        steps (List[str]): The list of reasoning steps.
        min_steps (int, optional): Minimum number of steps required. Defaults to 3.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    if len(steps) < min_steps:
        return False
    for step in steps:
        if not step or len(step) < 5:  # Example criteria
            return False
    return True