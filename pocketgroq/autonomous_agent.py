import time
import logging
from typing import List, Dict, Any, Generator
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AutonomousAgent:
    def __init__(self, groq_provider: GroqProvider, max_sources: int = 5, search_delay: float = 2.0):
        self.groq = groq_provider
        self.max_sources = max_sources
        self.search_delay = search_delay

    def process_request(self, request: str, max_sources: int = None) -> Generator[Dict[str, str], None, None]:
        if max_sources is not None:
            self.max_sources = max_sources

        self._inform_user(f"Processing request: '{request}'")
        yield {"type": "research", "content": f"Processing request: '{request}'"}

        initial_response = self.groq.generate(request)
        self._inform_user(f"Initial response: {initial_response}")
        yield {"type": "research", "content": f"Initial response: {initial_response}"}

        if self.groq.evaluate_response(request, initial_response):
            self._inform_user("Initial response was satisfactory.")
            yield {"type": "research", "content": "Initial response was satisfactory."}
            yield {"type": "response", "content": initial_response}
            return

        self._inform_user("Initial response was not satisfactory. I'll search for information online.")
        yield {"type": "research", "content": "Initial response was not satisfactory. I'll search for information online."}

        search_query = self._generate_search_query(request)
        self._inform_user(f"Generated search query: '{search_query}'")
        yield {"type": "research", "content": f"Generated search query: '{search_query}'"}

        search_results = self.groq.web_search(search_query)
        self._inform_user(f"Found {len(search_results)} search results.")
        yield {"type": "research", "content": f"Found {len(search_results)} search results."}

        for i, result in enumerate(search_results[:self.max_sources]):
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

                if self.groq.evaluate_response(request, response):
                    self._inform_user("This response is satisfactory.")
                    yield {"type": "research", "content": "This response is satisfactory."}
                    yield {"type": "response", "content": response}
                    return
                else:
                    self._inform_user("This response was not satisfactory. I'll check another source.")
                    yield {"type": "research", "content": "This response was not satisfactory. I'll check another source."}
            except GroqAPIError as e:
                if e.status_code == 429:
                    self._inform_user("I've encountered a rate limit. I'll wait for a minute before trying again.")
                    yield {"type": "research", "content": "I've encountered a rate limit. I'll wait for a minute before trying again."}
                    time.sleep(60)
                else:
                    self._inform_user(f"I encountered an error while processing {result['url']}: {str(e)}")
                    yield {"type": "research", "content": f"I encountered an error while processing {result['url']}: {str(e)}"}
            except Exception as e:
                self._inform_user(f"An unexpected error occurred while processing {result['url']}: {str(e)}")
                yield {"type": "research", "content": f"An unexpected error occurred while processing {result['url']}: {str(e)}"}

        final_message = "I'm sorry, but after checking multiple sources, I couldn't find a satisfactory answer to your request."
        self._inform_user(final_message)
        yield {"type": "response", "content": final_message}

    def _generate_search_query(self, request: str) -> str:
        prompt = f"Generate a single, concise search query (no more than 6 words) to find information for: '{request}'. Respond with only the search query, no other text."
        query = self.groq.generate(prompt).strip()
        
        # Remove any quotes or backticks that might have been added
        query = query.replace('"', '').replace('`', '').strip()
        
        logger.debug(f"Generated search query: {query}")
        return query

    def _generate_response_from_content(self, request: str, content: str) -> str:
        prompt = f"Based on the following content, provide a concise and accurate answer to this request: '{request}'\n\nContent: {content[:4000]}"
        response = self.groq.generate(prompt)
        logger.debug(f"Generated response from content: {response}")
        return response

    def _inform_user(self, message: str):
        print(f"Agent: {message}")
        logger.debug(message)