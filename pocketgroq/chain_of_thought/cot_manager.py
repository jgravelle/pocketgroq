# pocketgroq/chain_of_thought/cot_manager.py

from typing import List
from .llm_interface import LLMInterface
from .utils import sanitize_input

class ChainOfThoughtManager:
    """
    Manages the Chain-of-Thought reasoning process.
    """
    def __init__(self, llm: LLMInterface, cot_prompt_template: str = None):
        """
        Initialize with an LLM instance and an optional CoT prompt template.
        """
        self.llm = llm
        self.cot_prompt_template = cot_prompt_template or (
            "Solve the following problem step by step:\n\n{problem}\n\nSolution:"
        )

    def generate_cot(self, problem: str) -> List[str]:
        """
        Generate intermediate reasoning steps (Chain-of-Thought) for the given problem.
        """
        sanitized_problem = sanitize_input(problem)
        prompt = self.cot_prompt_template.format(problem=sanitized_problem)
        response = self.llm.generate(prompt)
        cot_steps = self._parse_cot(response)
        return cot_steps

    def synthesize_response(self, cot_steps: List[str]) -> str:
        """
        Synthesize the final answer from the Chain-of-Thought steps.
        """
        synthesis_prompt = "Based on the following reasoning steps, provide a concise answer:\n\n"
        synthesis_prompt += "\n".join(cot_steps) + "\n\nAnswer:"
        final_response = self.llm.generate(synthesis_prompt)
        return final_response.strip()

    def solve_problem(self, problem: str) -> str:
        """
        Complete process to solve a problem using Chain-of-Thought.
        """
        cot = self.generate_cot(problem)
        answer = self.synthesize_response(cot)
        return answer

    def _parse_cot(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract individual reasoning steps.
        This method can be customized based on how the LLM formats its output.
        """
        # Simple split by newline for demonstration; can be enhanced.
        steps = [line.strip() for line in response.split('\n') if line.strip()]
        return steps