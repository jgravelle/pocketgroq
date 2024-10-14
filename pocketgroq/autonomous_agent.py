import time
import logging
import re
from typing import List, Dict, Any, Generator
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AutonomousAgent:
    def __init__(self, groq_provider: GroqProvider, max_sources: int = 5, search_delay: float = 2.0, model: str = "llama3-8b-8192", temperature: float = 0.0):
        self.groq = groq_provider
        self.max_sources = max_sources
        self.search_delay = search_delay
        self.model = model
        self.temperature = temperature

    def process_request(self, request: str, max_sources: int = None, verify: bool = False) -> Generator[Dict[str, str], None, None]:
        if max_sources is not None:
            self.max_sources = max_sources

        self._inform_user(f"Processing request: '{request}'")
        yield {"type": "research", "content": f"Processing request: '{request}'"}

        initial_response = self.groq.generate(prompt=request, model=self.model, temperature=self.temperature)
        self._inform_user(f"Initial response: {initial_response}")
        yield {"type": "research", "content": f"Initial response: {initial_response}"}

        self._inform_user("Searching for information online.")
        yield {"type": "research", "content": "Searching for information online."}

        search_query = self._generate_search_query(request)
        self._inform_user(f"Generated search query: '{search_query}'")
        yield {"type": "research", "content": f"Generated search query: '{search_query}'"}

        search_results = self.groq.web_search(search_query)
        self._inform_user(f"Found {len(search_results)} search results.")
        yield {"type": "research", "content": f"Found {len(search_results)} search results."}

        verified_sources = []
        for i, result in enumerate(search_results):
            if i >= self.max_sources:
                break

            if i > 0:
                time.sleep(self.search_delay)

            self._inform_user(f"Checking source {i+1}: {result['url']}")
            yield {"type": "research", "content": f"Checking source {i+1}: {result['url']}"}

            try:
                content = self.groq.get_web_content(result['url'])
                self._inform_user(f"Retrieved content from {result['url']} (length: {len(content)} characters)")
                yield {"type": "research", "content": f"Retrieved content from {result['url']} (length: {len(content)} characters)"}

                response = self._generate_response_from_content(request, content)
                self._inform_user(f"Generated response from content: {response}")
                yield {"type": "research", "content": f"Generated response from content: {response}"}

                if self._evaluate_response(request, response):
                    verified_sources.append((result['url'], response))
                    if not verify or len(verified_sources) >= 3:  # Check at least 3 sources
                        break
                else:
                    self._inform_user("This response was not satisfactory. Checking another source.")
                    yield {"type": "research", "content": "This response was not satisfactory. Checking another source."}
            except GroqAPIError as e:
                if e.status_code == 429:
                    self._inform_user("Rate limit encountered. Waiting for a minute before retrying.")
                    yield {"type": "research", "content": "Rate limit encountered. Waiting for a minute before retrying."}
                    time.sleep(60)
                else:
                    self._inform_user(f"Error processing {result['url']}: {str(e)}")
                    yield {"type": "research", "content": f"Error processing {result['url']}: {str(e)}"}
            except Exception as e:
                self._inform_user(f"Unexpected error processing {result['url']}: {str(e)}")
                yield {"type": "research", "content": f"Unexpected error processing {result['url']}: {str(e)}"}

        if verified_sources:
            final_response = self._select_best_response(verified_sources, verify)
            self._inform_user(final_response)
            yield {"type": "response", "content": final_response}
        else:
            final_message = "After checking multiple sources, I couldn't find a satisfactory answer to your request."
            self._inform_user(final_message)
            yield {"type": "response", "content": final_message}

    def _select_best_response(self, verified_sources: List[tuple], verify: bool) -> str:
        if verify and len(verified_sources) >= 2:
            # Compare the responses and select the most consistent or recent one
            consistent_response = self._find_consistent_response(verified_sources)
            if consistent_response:
                sources = ", ".join([source[0] for source in verified_sources[:3]])
                return f"Based on verification from multiple sources ({sources}), here's the answer:\n\n{consistent_response}"
            else:
                # If no consistent response, use the most recent one
                return f"Based on the most recent source {verified_sources[-1][0]}, here's the answer:\n\n{verified_sources[-1][1]}"
        else:
            # If verification is not required or we don't have enough sources, use the most recent one
            return f"Based on the source {verified_sources[-1][0]}, here's the answer:\n\n{verified_sources[-1][1]}"

    def _find_consistent_response(self, verified_sources: List[tuple]) -> str:
        # Implement logic to find the most consistent response among the verified sources
        # This could involve comparing key elements of the responses or using a similarity metric
        # For simplicity, we'll use a basic comparison here
        responses = [source[1] for source in verified_sources]
        for response in responses:
            if responses.count(response) > 1:
                return response
        return None

    def _generate_search_query(self, request: str) -> str:
        prompt = f"Generate a single, concise search query (no more than 8 words) to find current, specific information for: '{request}'. Include words like 'current' or 'today' to emphasize recency. Respond with only the search query, no other text."
        query = self.groq.generate(prompt=prompt, model=self.model, temperature=self.temperature).strip()
        query = query.replace('"', '').replace('`', '').strip()
        logger.debug(f"Generated search query: {query}")
        return query

    def _generate_response_from_content(self, request: str, content: str) -> str:
        prompt = f"Based on the following content, provide a concise and accurate answer to this request: '{request}'. Include only current, specific information. Do not use placeholders or generic responses. If the information is not available or current, state that clearly.\n\nContent: {content[:4000]}"
        response = self.groq.generate(prompt=prompt, model=self.model, temperature=self.temperature)
        logger.debug(f"Generated response from content: {response}")
        return response

    def _evaluate_response(self, request: str, response: str) -> bool:
        # Check for placeholder text
        if re.search(r'\[.*?\]', response):
            return False

        # Check for generic or non-specific responses
        generic_phrases = ["I'm sorry", "I don't have access to real-time data", "I cannot provide current information"]
        if any(phrase in response for phrase in generic_phrases):
            return False

        # Use the existing evaluate_response method as an additional check
        return self.groq.evaluate_response(request, response)

    def _inform_user(self, message: str):
        print(f"Agent: {message}")
        logger.debug(message)