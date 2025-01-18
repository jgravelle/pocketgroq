# setup.py

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pocketgroq",
    version="0.5.6",
    author="PocketGroq Team",
    author_email="j@gravelle.us",
    description="A library for easy integration with Groq API, including web scraping, image handling, and Chain of Thought reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgravelle/pocketgroq",
    project_urls={
        "Bug Tracker": "https://github.com/jgravelle/pocketgroq/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(include=['pocketgroq', 'pocketgroq.*']),
    python_requires=">=3.7",
    install_requires=[
        "bs4>=0.0.2",
        "groq>=0.8.0",
        "python-dotenv>=0.19.1",
        "pytest>=7.3.1",
        "pytest-asyncio>=0.21.0",
        "requests>=2.32.3",
        "langchain>=0.3.1",
        "langchain-groq>=0.2.0",
        "langchain-community>=0.3.1",
        "markdown2>=2.5.0",
        "faiss-cpu>=1.8.0.post1",
        "ollama>=0.3.3",
        "html2text>=2024.2.26",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-asyncio>=0.21.0",
        ],
    },
)
```

# test.py

```python
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import uuid

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError
from pocketgroq.autonomous_agent import AutonomousAgent

DEBUG = False
logging.basicConfig(level=logging.FATAL)
logger = logging.getLogger(__name__)

if DEBUG:
    logger.setLevel(logging.DEBUG)

groq = GroqProvider(rag_persistent=True, rag_index_path="faiss_persistent_index.pkl")

PERSISTENT_SESSION_ID = str(uuid.uuid4())
DISPOSABLE_SESSION_ID = str(uuid.uuid4())

def start_conversations():
    groq.start_conversation(PERSISTENT_SESSION_ID)
    groq.start_conversation(DISPOSABLE_SESSION_ID)

def test_get_available_models():
    print("Testing Get Available Models...")
    models = groq.get_available_models()
    print("Available Models:", models)
    assert isinstance(models, list) and len(models) > 0

def test_basic_chat_completion():
    print("Testing Basic Chat Completion...")
    response = groq.generate(
        prompt="Explain the importance of fast language models in one sentence.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    print("Response:", response)
    assert isinstance(response, str) and len(response) > 0

def test_streaming_chat_completion():
    print("\nTesting Streaming Chat Completion...")
    stream = groq.generate(
        prompt="Count from 1 to 5.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True
    )
    full_response = ""
    for chunk in stream:
        if chunk is not None:
            print(chunk, end="")
            full_response += chunk
        else:
            print("<None>", end="")
    print()
    assert isinstance(full_response, str) and len(full_response) > 0
    print(f"Full response: '{full_response}'")

def test_override_default_model():
    print("\nTesting Override Default Model...")
    selected_model = 'llama3-groq-8b-8192-tool-use-preview'
    response = groq.generate("Explain quantum computing in one sentence.", model=selected_model)
    print("Response with Selected Model:", response)
    assert isinstance(response, str) and len(response) > 0

def test_chat_completion_with_stop_sequence():
    print("\nTesting Chat Completion with Stop Sequence...")
    response = groq.generate(
        prompt="Count to 10. Your response must begin with \"1, \". Example: 1, 2, 3, ...",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=", 6",
        stream=False
    )
    print("Response:", response)
    assert isinstance(response, str) and "5" in response and "6" not in response

def test_response_evaluation():
    print("\nTesting Response Evaluation...")
    request = "What is the capital of France?"
    good_response = "The capital of France is Paris."
    bad_response = "I'm not sure, but I think it might be London or Paris."

    is_good = groq.evaluate_response(request, good_response)
    is_bad = groq.evaluate_response(request, bad_response)

    print(f"Good response evaluation: {is_good}")
    print(f"Bad response evaluation: {is_bad}")

    assert is_good == True, "Good response should be evaluated as satisfactory"
    assert is_bad == False, "Bad response should be evaluated as unsatisfactory"

async def test_async_generation():
    print("\nTesting Asynchronous Generation...")
    response = await groq.generate(
        prompt="Explain the theory of relativity in one sentence.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        async_mode=True
    )
    print("Response:", response)
    assert isinstance(response, str) and len(response) > 0

async def test_streaming_async_chat_completion():
    print("\nTesting Streaming Async Chat Completion...")
    stream = await groq.generate(
        prompt="Count from 1 to 5.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
        async_mode=True
    )
    full_response = ""
    async for chunk in stream:
        if chunk is not None:
            print(chunk, end="")
            full_response += chunk
        else:
            print("<None>", end="")
    print()
    assert isinstance(full_response, str) and len(full_response) > 0
    print(f"Full response: '{full_response}'")

def test_json_mode():
    print("\nTesting JSON Mode...")
    class Ingredient(BaseModel):
        name: str
        quantity: Union[int, str]
        quantity_unit: Optional[str]

        @field_validator('quantity', mode='before')
        @classmethod
        def quantity_to_string(cls, v):
            return str(v)

    class Recipe(BaseModel):
        recipe_name: str = Field(..., alias="name")
        ingredients: List[Ingredient]
        directions: List[str] = Field(..., alias="instructions")

    def get_recipe(recipe_name: str) -> Recipe:
        response = groq.generate(
            prompt=f"Create a recipe for {recipe_name} and return it in JSON format. Include 'name', 'ingredients' (each with 'name', 'quantity', and 'quantity_unit'), and 'instructions' fields.",
            model="llama3-8b-8192",
            temperature=0,
            stream=False,
            json_mode=True
        )
        logger.debug(f"Raw JSON response: {response}")
        
        try:
            json_data = json.loads(response)
            if 'recipe' in json_data:
                json_data = json_data['recipe']
            recipe = Recipe.model_validate(json_data)
            return recipe
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise
        except ValueError as e:
            logger.error(f"Failed to validate Recipe model: {e}")
            raise

    def print_recipe(recipe: Recipe):
        print("Recipe:", recipe.recipe_name)
        print("\nIngredients:")
        for ingredient in recipe.ingredients:
            print(f"- {ingredient.name}: {ingredient.quantity} {ingredient.quantity_unit or ''}")
        print("\nDirections:")
        for step, direction in enumerate(recipe.directions, start=1):
            print(f"{step}. {direction}")

    try:
        recipe = get_recipe("simple pancakes")
        print_recipe(recipe)
        assert isinstance(recipe, Recipe)
        assert len(recipe.ingredients) > 0
        assert len(recipe.directions) > 0
    except Exception as e:
        logger.error(f"Error in JSON mode test: {e}")
        raise

def test_tool_usage():
    print("\nTesting Tool Usage...")
    def reverse_string(input_string: str) -> dict:
        return {"reversed_string": input_string[::-1]}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "reverse_string",
                "description": "Reverse the given string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_string": {
                            "type": "string",
                            "description": "The string to be reversed",
                        }
                    },
                    "required": ["input_string"],
                }
            }
        }
    ]

    groq.register_tool("reverse_string", reverse_string)

    response = groq.generate("Please reverse the string 'hello world'", tools=tools)
    print("Response:", response)
    assert "dlrow olleh" in response.lower()

def test_vision():
    print("\nTesting Vision...")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    response = groq._create_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        model="llama-3.2-11b-vision-preview",
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    print("Response:", response)
    assert isinstance(response, str) and len(response) > 0

def test_cot_problem_solving():
    print("\nTesting Chain of Thought Problem Solving...")
    complex_problem = """
    A farmer has a rectangular field that is 100 meters long and 50 meters wide. 
    He wants to increase the area of the field by 20% by increasing both the length and the width by the same percentage. 
    What should be the new length and width of the field? 
    Round your answer to the nearest centimeter.
    """
    answer = groq.solve_problem_with_cot(complex_problem)
    print("Problem:", complex_problem)
    print("Answer:", answer)
    assert isinstance(answer, str) and len(answer) > 0

def test_cot_step_generation():
    print("\nTesting Chain of Thought Step Generation...")
    problem = "What is the sum of the first 10 prime numbers?"
    cot_steps = groq.generate_cot(problem)
    print("Problem:", problem)
    print("Chain of Thought Steps:")
    for i, step in enumerate(cot_steps, 1):
        print(f"{i}. {step}")
    assert isinstance(cot_steps, list) and len(cot_steps) > 0

def test_cot_synthesis():
    print("\nTesting Chain of Thought Synthesis...")
    cot_steps = [
        "The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        "To find the sum, we add these numbers: 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29",
        "Calculating the sum: 129"
    ]
    final_answer = groq.synthesize_cot(cot_steps)
    print("Chain of Thought Steps:", cot_steps)
    print("Synthesized Answer:", final_answer)
    assert isinstance(final_answer, str) and len(final_answer) > 0

def test_rag_initialization():
    print("\nTesting RAG Initialization...")
    try:
        try:
            subprocess.run(["ollama", "list"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Ollama is not running. Please start Ollama service.")
            return

        groq.initialize_rag()
        print("RAG initialized successfully.")
        assert groq.rag_manager is not None
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
        raise

def test_document_loading(persistent: bool = True):
    mode = "Persistent" if persistent else "Disposable"
    print(f"\nTesting Document Loading in {mode} Mode...")
    try:
        def progress_callback(current, total):
            print(f"Processing document chunks: {current}/{total}")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write("This is a test document about artificial intelligence and machine learning.")
            temp_file_path = temp_file.name

        groq.load_documents(temp_file_path, progress_callback=progress_callback, timeout=60, persistent=persistent)
        print("Local document loaded successfully.")
        assert groq.rag_manager.vector_store is not None

        try:
            groq.load_documents("https://en.wikipedia.org/wiki/Artificial_intelligence", 
                                progress_callback=progress_callback, timeout=120, persistent=persistent)
            print("Web document loaded successfully.")
        except TimeoutError:
            print("Web document loading timed out, but local document was processed successfully.")
        
        assert groq.rag_manager.vector_store is not None
    except Exception as e:
        print(f"Failed to load document: {e}")
        raise
    finally:
        os.unlink(temp_file_path)

def test_document_querying():
    print("\nTesting Document Querying...")
    query = "What is the main topic of the document?"
    try:
        response = groq.query_documents(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        assert isinstance(response, str) and len(response) > 0
    except Exception as e:
        print(f"Failed to query documents: {e}")
        raise

def test_rag_error_handling():
    print("\nTesting RAG Error Handling...")
    global groq
    groq = GroqProvider(rag_persistent=False)
    
    try:
        groq.query_documents("This should fail")
    except ValueError as e:
        print(f"Expected error caught: {e}")
        assert str(e) == "RAG has not been initialized. Call initialize_rag first."
    else:
        raise AssertionError("Expected ValueError was not raised")

    try:
        groq.load_documents("This should also fail")
    except ValueError as e:
        print(f"Expected error caught: {e}")
        assert str(e) == "RAG has not been initialized. Call initialize_rag first."
    else:
        raise AssertionError("Expected ValueError was not raised")
    
def test_persistent_conversation():
    print("\nTesting Persistent Conversation...")
    session_id = PERSISTENT_SESSION_ID
    groq.start_conversation(session_id)
    
    user_message1 = "What is the capital of Ohio?"
    response1 = groq.generate(prompt=user_message1, session_id=session_id)
    print(f"User: {user_message1}")
    print(f"PG: {response1}")
    assert isinstance(response1, str) and len(response1) > 0

    user_message2 = "What is its population?"
    response2 = groq.generate(prompt=user_message2, session_id=session_id)
    print(f"\nUser: {user_message2}")
    print(f"PG: {response2}")
    assert isinstance(response2, str) and len(response2) > 0

def test_disposable_conversation():
    print("\nTesting Disposable Conversation...")
    session_id = DISPOSABLE_SESSION_ID
    groq.start_conversation(session_id)
    
    user_message1 = "What is the capital of Ohio?"
    response1 = groq.generate(prompt=user_message1, session_id=session_id)
    print(f"User: {user_message1}")
    print(f"PG: {response1}")
    assert isinstance(response1, str) and len(response1) > 0

    user_message2 = "What is its population?"
    groq.reset_conversation(session_id)
    response2 = groq.generate(prompt=user_message2, session_id=session_id)
    print(f"\nUser: {user_message2}")
    print(f"PG: {response2}")
    assert isinstance(response2, str) and len(response2) > 0

def test_web_search():
    print("\nTesting Web Search...")
    query = "What is PocketGroq?"
    results = groq.web_search(query)
    print(f"Search query: {query}")
    print(f"Number of results: {len(results)}")
    assert isinstance(results, list) and len(results) > 0
    print("First result:", results[0])

def test_get_web_content():
    print("\nTesting Get Web Content...")
    url = "https://yahoo.com"
    content = groq.get_web_content(url)
    print(f"Content length for {url}: {len(content)} characters")
    assert isinstance(content, str) and len(content) > 0
    print("First 100 characters:", content[:100])

def test_crawl_website():
    print("\nTesting Website Crawling...")
    url = "https://yahoo.com"
    results = groq.crawl_website(url, formats=["markdown", "html"], max_depth=2, max_pages=5)
    print(f"Crawl results for {url}:")
    print(f"Number of pages crawled: {len(results)}")
    assert isinstance(results, list) and len(results) > 0
    print("First page title:", results[0]['metadata']['title'])

def test_scrape_url():
    print("\nTesting URL Scraping...")
    url = "https://yahoo.com"
    result = groq.scrape_url(url, formats=["markdown", "html", "structured_data"])
    print(f"Scrape result for {url}:")
    assert isinstance(result, dict) and 'markdown' in result and 'html' in result
    print("Markdown content length:", len(result['markdown']))
    print("HTML content length:", len(result['html']))
    if 'structured_data' in result:
        print("Structured data:", json.dumps(result['structured_data'], indent=2))

def test_autonomous_agent():
    print("\nTesting Autonomous Agent...")
    agent = AutonomousAgent(groq, max_sources=3)  # Limit to 3 sources for faster testing
    
    request = "What is the current temperature in Sheboygan, Wisconsin?"
    response = agent.process_request(request)
    
    print(f"\nRequest: {request}")
    print(f"Final response: {response}")
    
    is_satisfactory = groq.evaluate_response(request, response)
    print(f"Response is satisfactory: {is_satisfactory}")
    
    assert is_satisfactory, "The autonomous agent should provide a satisfactory response"


def display_menu():
    print("\nPocketGroq Test Menu:")
    print("1. Get Available Models")
    print("2. Basic Chat Completion")
    print("3. Streaming Chat Completion")
    print("4. Override Default Model")
    print("5. Chat Completion with Stop Sequence")
    print("6. Asynchronous Generation")
    print("7. Streaming Async Chat Completion")
    print("8. JSON Mode")
    print("9. Tool Usage")
    print("10. Vision")
    print("11. Chain of Thought Problem Solving")
    print("12. Chain of Thought Step Generation")
    print("13. Chain of Thought Synthesis")
    print("14. Test RAG Initialization")
    print("15. Test Document Loading")
    print("16. Test Document Querying")
    print("17. Test RAG Error Handling")
    print("18. Test Persistent Conversation")
    print("19. Test Disposable Conversation")
    print("20. Test Response Evaluation")
    print("21. Web Search")
    print("22. Get Web Content")
    print("23. Crawl Website")
    print("24. Scrape URL")
    print("25. Test Autonomous Agent")
    print("26. Run All Web Tests")
    print("27. Run All RAG Tests")
    print("28. Run All Conversation Tests")
    print("29. Run All Tests")
    print("0. Exit")

async def main():
    start_conversations()
    
    while True:
        display_menu()
        choice = input("Enter your choice (0-29): ")
        
        try:
            if choice == '0':
                break
            elif choice == '1':
                test_get_available_models()
            elif choice == '2':
                test_basic_chat_completion()
            elif choice == '3':
                test_streaming_chat_completion()
            elif choice == '4':
                test_override_default_model()
            elif choice == '5':
                test_chat_completion_with_stop_sequence()
            elif choice == '6':
                await test_async_generation()
            elif choice == '7':
                await test_streaming_async_chat_completion()
            elif choice == '8':
                test_json_mode()
            elif choice == '9':
                test_tool_usage()
            elif choice == '10':
                test_vision()
            elif choice == '11':
                test_cot_problem_solving()
            elif choice == '12':
                test_cot_step_generation()
            elif choice == '13':
                test_cot_synthesis()
            elif choice == '14':
                test_rag_initialization()
            elif choice == '15':
                test_document_loading(persistent=True)
            elif choice == '16':
                test_document_querying()
            elif choice == '17':
                test_rag_error_handling()
            elif choice == '18':
                test_persistent_conversation()
            elif choice == '19':
                test_disposable_conversation()
            elif choice == '20':
                test_response_evaluation()
            elif choice == '21':
                test_web_search()
            elif choice == '22':
                test_get_web_content()
            elif choice == '23':
                test_crawl_website()
            elif choice == '24':
                test_scrape_url()
            elif choice == '25':
                test_autonomous_agent()
            elif choice == '26':
                test_web_search()
                test_get_web_content()
                test_crawl_website()
                test_scrape_url()
                print("\nAll Web tests completed successfully!")
            elif choice == '27':
                test_rag_initialization()
                test_document_loading(persistent=True)
                test_document_querying()
                test_rag_error_handling()
                print("\nAll RAG tests completed successfully!")
            elif choice == '28':
                test_persistent_conversation()
                test_disposable_conversation()
                print("\nAll Conversation tests completed successfully!")
            elif choice == '29':
                test_get_available_models()
                test_basic_chat_completion()
                test_streaming_chat_completion()
                test_override_default_model()
                test_chat_completion_with_stop_sequence()
                await test_async_generation()
                await test_streaming_async_chat_completion()
                test_json_mode()
                test_tool_usage()
                test_vision()
                test_cot_problem_solving()
                test_cot_step_generation()
                test_cot_synthesis()
                test_rag_initialization()
                test_document_loading(persistent=True)
                test_document_querying()
                test_rag_error_handling()
                test_persistent_conversation()
                test_disposable_conversation()
                test_response_evaluation()
                test_web_search()
                test_get_web_content()
                test_crawl_website()
                test_scrape_url()
                test_autonomous_agent()
                print("\nAll tests completed successfully!")
            else:
                print("Invalid choice. Please try again.")
        except GroqAPIKeyMissingError as e:
            print(f"Error: {e}")
        except GroqAPIError as e:
            print(f"API Error: {e}")
        except AssertionError as e:
            print(f"Assertion Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    asyncio.run(main())

```

# pocketgroq\autonomous_agent.py

```python
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
```

# pocketgroq\config.py

```python
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
```

# pocketgroq\enhanced_web_tool.py

```python
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, urljoin
import html2text
import json

class EnhancedWebTool:
    def __init__(self, max_depth: int = 3, max_pages: int = 100):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        self.html2text_converter = html2text.HTML2Text()
        self.html2text_converter.ignore_links = False
        self.html2text_converter.ignore_images = False
        self.html2text_converter.ignore_emphasis = False
        self.html2text_converter.body_width = 0  # Disable line wrapping

    def crawl(self, start_url: str, formats: List[str] = ["markdown"]) -> List[Dict[str, Any]]:
        visited = set()
        to_visit = [(start_url, 0)]
        results = []

        while to_visit and len(results) < self.max_pages:
            url, depth = to_visit.pop(0)
            if url in visited or depth > self.max_depth:
                continue

            visited.add(url)
            page_content = self.scrape_page(url, formats)
            if page_content:
                results.append(page_content)

            if depth < self.max_depth:
                links = self.extract_links(url, page_content.get('html', ''))
                to_visit.extend((link, depth + 1) for link in links if link not in visited)

        return results

    def scrape_page(self, url: str, formats: List[str]) -> Dict[str, Any]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            result = {
                'url': url,
                'metadata': self.extract_metadata(soup, url),
            }

            if 'markdown' in formats:
                result['markdown'] = self.html_to_markdown(str(soup))
            if 'html' in formats:
                result['html'] = str(soup)
            if 'structured_data' in formats:
                result['structured_data'] = self.extract_structured_data(soup)

            return result
        except requests.RequestException as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def extract_links(self, base_url: str, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = urlparse(base_url).netloc
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == base_domain:
                links.append(full_url)

        return links

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        metadata = {
            'title': soup.title.string if soup.title else '',
            'description': '',
            'language': soup.html.get('lang', ''),
            'sourceURL': url,
        }

        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') == 'description':
                metadata['description'] = tag.get('content', '')
            elif tag.get('property') == 'og:description':
                metadata['og_description'] = tag.get('content', '')

        return metadata

    def html_to_markdown(self, html_content: str) -> str:
        # Convert HTML to Markdown using html2text
        markdown = self.html2text_converter.handle(html_content)
        
        # Clean up the markdown
        markdown = markdown.strip()
        
        return markdown
        
        return markdown

    def extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        structured_data = {}

        # Extract all text content
        all_text = soup.get_text(separator=' ', strip=True)
        structured_data['full_text'] = all_text

        # Extract headings
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            if h_tags:
                headings[f'h{i}'] = [tag.get_text(strip=True) for tag in h_tags]
        structured_data['headings'] = headings

        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'text': a.get_text(strip=True),
                'href': a['href']
            })
        structured_data['links'] = links

        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({
                'src': img['src'],
                'alt': img.get('alt', '')
            })
        structured_data['images'] = images

        # Extract JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data['json_ld'] = data
            except json.JSONDecodeError:
                pass

        return structured_data
```

# pocketgroq\exceptions.py

```python
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
```

# pocketgroq\groq_provider.py

```python
import asyncio
import json
import logging
import os
import subprocess
import requests

from collections import defaultdict
from groq import Groq, AsyncGroq
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from typing import Callable, Dict, Any, List, Union, AsyncIterator, Optional

from .enhanced_web_tool import EnhancedWebTool
from .exceptions import GroqAPIKeyMissingError, GroqAPIError, OllamaServerNotRunningError
from .web_tool import WebTool
from .chain_of_thought.cot_manager import ChainOfThoughtManager
from .chain_of_thought.llm_interface import LLMInterface
from .rag_manager import RAGManager

logger = logging.getLogger(__name__)

class GroqProvider(LLMInterface):
    def __init__(self, api_key: str = None, rag_persistent: bool = True, rag_index_path: str = "faiss_index.pkl"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise GroqAPIKeyMissingError("Groq API key is not provided")
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        self.tool_use_models = [
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview"
        ]
        self.web_tool = WebTool()
        self.cot_manager = ChainOfThoughtManager(llm=self)
        self.rag_manager = None
        self.tools = {}
        self.rag_persistent = rag_persistent
        self.rag_index_path = rag_index_path
        self.enhanced_web_tool = EnhancedWebTool()

        # Initialize conversation sessions
        self.conversation_sessions = defaultdict(list)  # session_id -> list of messages

        # Check if Ollama server is running and initialize RAG if it is
        if self.is_ollama_server_running():
            if self.rag_persistent:
                logger.info("Initializing RAG with persistence enabled.")
                self.initialize_rag(index_path=self.rag_index_path)
        else:
            logger.warning("Ollama server is not running. RAG functionality will be limited.")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Fetch the list of available models from the Groq provider.

        Returns:
            List[Dict[str, Any]]: A list of models with their details.
        """
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return models
        except requests.RequestException as e:
            logger.error(f"Failed to fetch models: {e}")
            raise GroqAPIError(f"Failed to fetch models: {e}")

    def crawl_website(self, url: str, formats: List[str] = ["markdown"], max_depth: int = 3, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Crawl a website and return its content in specified formats.
        
        Args:
            url (str): The starting URL to crawl.
            formats (List[str]): List of desired output formats (e.g., ["markdown", "html", "structured_data"]).
            max_depth (int): Maximum depth to crawl.
            max_pages (int): Maximum number of pages to crawl.
        
        Returns:
            List[Dict[str, Any]]: List of crawled pages with their content in specified formats.
        """
        self.enhanced_web_tool.max_depth = max_depth
        self.enhanced_web_tool.max_pages = max_pages
        return self.enhanced_web_tool.crawl(url, formats)

    def is_ollama_server_running(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def ensure_ollama_server_running(func):
        """Decorator to ensure Ollama server is running for functions that require it."""
        def wrapper(self, *args, **kwargs):
            if not self.is_ollama_server_running():
                raise OllamaServerNotRunningError("Ollama server is not running. Please start it and try again.")
            return func(self, *args, **kwargs)
        return wrapper
    
    def evaluate_response(self, request: str, response: str) -> bool:
        """
        Evaluate if a response satisfies a given request using an AI LLM.
        
        Args:
            request (str): The original request or question.
            response (str): The response to be evaluated.
        
        Returns:
            bool: True if the response is deemed satisfactory, False otherwise.
        """
        evaluation_prompt = f"""
        You will be given a request and a response. Your task is to evaluate the response based on the following criteria:
        1. **Informative and Correct**: The response must be accurate and provide clear, useful, and sufficient information to fully answer the request.
        2. **No Uncertainty**: The response should not express any uncertainty, such as language indicating doubt (e.g., "maybe," "possibly," "it seems") or statements that are inconclusive.

        Request: {request}
        Response: {response}

        Based on these criteria, is the response satisfactory? Answer with only 'Yes' or 'No'.
        """

        evaluation = self.generate(evaluation_prompt, temperature=0.0, max_tokens=1)
        
        # Clean up the response and convert to boolean
        evaluation = evaluation.strip().lower()
        return evaluation == 'yes'

    def register_tool(self, name: str, func: callable):
        self.tools[name] = func

    def scrape_url(self, url: str, formats: List[str] = ["markdown"]) -> Dict[str, Any]:
        """
        Scrape a single URL and return its content in specified formats.
        
        Args:
            url (str): The URL to scrape.
            formats (List[str]): List of desired output formats (e.g., ["markdown", "html", "structured_data"]).
        
        Returns:
            Dict[str, Any]: The scraped content in specified formats.
        """
        return self.enhanced_web_tool.scrape_page(url, formats)

    def end_conversation(self, conversation_id: str):
        """
        Ends a conversation and clears its history.

        Args:
            conversation_id (str): The ID of the conversation to end.
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Ended conversation with ID: {conversation_id}")
        else:
            logger.warning(f"Attempted to end non-existent conversation ID: {conversation_id}")

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history for a given session.

        Args:
            session_id (str): Unique identifier for the conversation session.

        Returns:
            List[Dict[str, str]]: List of messages in the conversation.
        """
        return self.conversation_sessions.get(session_id, [])            

    def start_conversation(self, session_id: str):
        """
        Initialize a new conversation session.

        Args:
            session_id (str): Unique identifier for the conversation session.
        """
        if session_id in self.conversation_sessions:
            logger.warning(f"Session '{session_id}' already exists. Overwriting.")
        self.conversation_sessions[session_id] = []
        logger.info(f"Started new conversation session '{session_id}'.")
    
    def reset_conversation(self, session_id: str):
        if session_id in self.conversation_sessions:
            del self.conversation_sessions[session_id]
            logger.info(f"Conversation session '{session_id}' has been reset.")
        else:
            logger.warning(f"Attempted to reset non-existent session '{session_id}'.")   

    def generate(self, prompt: str, session_id: Optional[str] = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        if session_id:
            messages = self.conversation_sessions[session_id]
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self._create_completion(messages, **kwargs)

        if session_id:
            if isinstance(response, str):
                self.conversation_sessions[session_id].append({"role": "assistant", "content": response})
            elif asyncio.iscoroutine(response):
                # Handle asynchronous streaming responses if needed
                pass

        return response

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)

    def _create_completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[str, AsyncIterator[str]]:
        completion_kwargs = {
            "model": self._select_model(kwargs.get("model"), kwargs.get("tools")),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.5),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1),
            "stop": kwargs.get("stop", None),
            "stream": kwargs.get("stream", False),
        }

        if kwargs.get("json_mode", False):
            completion_kwargs["response_format"] = {"type": "json_object"}

        if kwargs.get("tools"):
            completion_kwargs["tools"] = self._prepare_tools(kwargs["tools"])
            completion_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        if kwargs.get("async_mode", False):
            return self._async_create_completion(**completion_kwargs)
        else:
            return self._sync_create_completion(**completion_kwargs)

    def _select_model(self, requested_model: str, tools: List[Dict[str, Any]]) -> str:
        if tools and not requested_model:
            return self.tool_use_models[0]
        elif tools and requested_model not in self.tool_use_models:
            print(f"Warning: {requested_model} is not optimized for tool use. Switching to {self.tool_use_models[0]}.")
            return self.tool_use_models[0]
        return requested_model or os.environ.get('GROQ_MODEL', 'llama3-8b-8192')
    
    def _prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_tools = []
        for tool in tools:
            prepared_tool = tool.copy()
            if 'function' in prepared_tool:
                prepared_tool['function'] = {k: v for k, v in prepared_tool['function'].items() if k != 'implementation'}
            prepared_tools.append(prepared_tool)
        return prepared_tools

    def _sync_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = self.client.chat.completions.create(**kwargs)
            if kwargs.get("stream", False):
                return (chunk.choices[0].delta.content for chunk in response)
            else:
                return self._process_tool_calls(response)
        except Exception as e:
            raise GroqAPIError(f"Error in Groq API call: {str(e)}")

    async def _async_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = await self.async_client.chat.completions.create(**kwargs)
            if kwargs.get("stream", False):
                async def async_generator():
                    async for chunk in response:
                        yield chunk.choices[0].delta.content
                return async_generator()
            else:
                return await self._async_process_tool_calls(response)
        except Exception as e:
            raise GroqAPIError(f"Error in async Groq API call: {str(e)}")

    def _process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = self._execute_tool_calls(message.tool_calls)
            new_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            return self._create_completion([new_message] + tool_results)
        return message.content

    async def _async_process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = await self._async_execute_tool_calls(message.tool_calls)
            new_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            for result in tool_results:
                new_message["tool_results"] = result
            return await self._async_create_completion([new_message])
        return message.content

    def _execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            if tool_call.function.name in self.tools:
                args = json.loads(tool_call.function.arguments)
                result = self.tools[tool_call.function.name](**args)
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    async def _async_execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = await asyncio.to_thread(self.web_tool.search, args.get("query", ""))
            elif tool_call.function.name == "get_web_content":
                args = json.loads(tool_call.function.arguments)
                result = await asyncio.to_thread(self.web_tool.get_web_content, args.get("url", ""))
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps(result),
            })
        return results

    def web_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Perform a web search using the integrated WebTool."""
        return self.web_tool.search(query)

    def get_web_content(self, url: str) -> str:
        """Retrieve the content of a web page using the integrated WebTool."""
        return self.web_tool.get_web_content(url)

    def is_url(self, text: str) -> bool:
        """Check if the given text is a valid URL using the integrated WebTool."""
        return self.web_tool.is_url(text)
    
    def solve_problem_with_cot(self, problem: str, **kwargs) -> str:
        """
        Solve a problem using Chain of Thought reasoning.
        """
        return self.cot_manager.solve_problem(problem)

    def generate_cot(self, problem: str, **kwargs) -> List[str]:
        """
        Generate Chain of Thought steps for a given problem.
        """
        return self.cot_manager.generate_cot(problem)

    def synthesize_cot(self, cot_steps: List[str], **kwargs) -> str:
        """
        Synthesize a final answer from Chain of Thought steps.
        """
        return self.cot_manager.synthesize_response(cot_steps)
    
    @ensure_ollama_server_running
    def initialize_rag(self, ollama_base_url: str = "http://localhost:11434", model_name: str = "nomic-embed-text", index_path: str = "faiss_index.pkl"):
        try:
            # Attempt to pull the model if it's not already available
            subprocess.run(["ollama", "pull", model_name], check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Failed to pull model {model_name}. Ensure Ollama is installed and running.")
            raise

        embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=model_name)
        self.rag_manager = RAGManager(embeddings, index_path=index_path)
        logger.info("RAG initialized successfully.")

    @ensure_ollama_server_running
    def load_documents(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                       progress_callback: Callable[[int, int], None] = None, timeout: int = 300, 
                       persistent: bool = None):
        if persistent is None:
            persistent = self.rag_persistent
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")

        # Use a separate index path if non-persistent
        index_path = self.rag_index_path if persistent else f"temp_{self.rag_index_path}"
        self.rag_manager.index_path = index_path

        self.rag_manager.load_and_process_documents(source, chunk_size, chunk_overlap, progress_callback, timeout)

    @ensure_ollama_server_running
    def query_documents(self, query: str, session_id: Optional[str] = None, **kwargs) -> str:
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")
        
        llm = ChatGroq(groq_api_key=self.api_key, model_name=kwargs.get("model", "llama3-8b-8192"))
        response = self.rag_manager.query_documents(llm, query)
        return response['answer']

    def query_documents(self, query: str, session_id: Optional[str] = None, **kwargs) -> str:
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")
        
        llm = ChatGroq(groq_api_key=self.api_key, model_name=kwargs.get("model", "llama3-8b-8192"))
        response = self.rag_manager.query_documents(llm, query)
        return response['answer']

```

# pocketgroq\rag_manager.py

```python
# pocketgroq/rag_manager.py

import os
import pickle
from typing import List, Dict, Any, Callable
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self, embeddings, index_path: str = "faiss_index.pkl"):
        self.embeddings = embeddings
        self.vector_store = None
        self.index_path = index_path

    def load_and_process_documents(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                                   progress_callback: Callable[[int, int], None] = None, 
                                   timeout: int = 300):  # 5 minutes timeout
        start_time = time.time()

        if os.path.exists(self.index_path):
            logger.info("Loading persisted FAISS index.")
            with open(self.index_path, 'rb') as f:
                self.vector_store = pickle.load(f)
            logger.info("FAISS index loaded successfully.")
            return

        if source.startswith(('http://', 'https://')):
            loader = WebBaseLoader(source)
        elif os.path.isfile(source):
            loader = TextLoader(source)
        else:
            raise ValueError(f"Unsupported source: {source}. Must be a valid URL or file path.")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)

        total_chunks = len(split_documents)
        for i, doc in enumerate(split_documents):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Document processing exceeded the {timeout} seconds timeout.")

            if not self.vector_store:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            else:
                self.vector_store.add_documents([doc])

            if progress_callback:
                progress_callback(i + 1, total_chunks)

        # Persist the FAISS index
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.vector_store, f)
        logger.info("FAISS index persisted to disk.")

    def query_documents(self, llm, query: str) -> Dict[str, Any]:
        if not self.vector_store:
            raise ValueError("Documents have not been loaded. Call load_and_process_documents first.")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context only.
            Provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain.invoke({"input": query})

```

# pocketgroq\utils.py

```python
import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file if it exists."""
    load_dotenv()

def get_env_variable(var_name: str, default: str = None) -> str:
    """Retrieve an environment variable or return a default value."""
    return os.getenv(var_name, default)
```

# pocketgroq\web_tool.py

```python
# pocketgroq/web_tool.py

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, quote_plus

DEBUG = False  # Set to True for debugging

def log_debug(message):
    if DEBUG:
        print(f"DEBUG: {message}")

class WebTool:
    def __init__(self, num_results: int = 10, max_tokens: int = 4096):
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        log_debug(f"Performing web search for: {query}")
        search_results = self._perform_web_search(query)
        filtered_results = self._filter_search_results(search_results)
        deduplicated_results = self._remove_duplicates(filtered_results)
        log_debug(f"Found {len(deduplicated_results)} unique results")
        return deduplicated_results[:self.num_results]

    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        encoded_query = quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded_query}&num={self.num_results * 2}"
        log_debug(f"Search URL: {search_url}")
        
        try:
            log_debug("Sending GET request to Google")
            response = requests.get(search_url, headers=self.headers, timeout=10)
            log_debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            log_debug("Parsing HTML with BeautifulSoup")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            log_debug("Searching for result divs")
            search_results = []
            for g in soup.find_all('div', class_='g'):
                log_debug("Processing a search result div")
                anchor = g.find('a')
                title = g.find('h3').text if g.find('h3') else 'No title'
                url = anchor.get('href', 'No URL') if anchor else 'No URL'
                
                description = ''
                description_div = g.find('div', class_=['VwiC3b', 'yXK7lf'])
                if description_div:
                    description = description_div.get_text(strip=True)
                else:
                    description = g.get_text(strip=True)
                
                log_debug(f"Found result: Title: {title[:30]}..., URL: {url[:30]}...")
                search_results.append({
                    'title': title,
                    'description': description,
                    'url': url
                })
            
            log_debug(f"Successfully retrieved {len(search_results)} search results for query: {query}")
            return search_results
        except requests.RequestException as e:
            log_debug(f"Error performing search: {str(e)}")
            return []

    def _filter_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [result for result in results if result['description'] and result['title'] != 'No title' and result['url'].startswith('https://')]
        log_debug(f"Filtered to {len(filtered)} results")
        return filtered

    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        unique_results = []
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        log_debug(f"Removed duplicates, left with {len(unique_results)} results")
        return unique_results

    def get_web_content(self, url: str) -> str:
        log_debug(f"Fetching content from: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            content = text[:self.max_tokens]
            log_debug(f"Retrieved {len(content)} characters of content")
            return content
        except requests.RequestException as e:
            log_debug(f"Error retrieving content from {url}: {str(e)}")
            return ""

    def is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _clean_url(self, url: str) -> str:
        url = url.rstrip(')')  # Remove trailing parenthesis if present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url  # Add https:// if missing
        return url
```

# pocketgroq\__init__.py

```python
# pocketgroq/__init__.py

from .groq_provider import GroqProvider
from .exceptions import GroqAPIKeyMissingError, GroqAPIError, OllamaServerNotRunningError
from .config import get_api_key
from .chain_of_thought import ChainOfThoughtManager, LLMInterface, sanitize_input, validate_cot_steps
from .rag_manager import RAGManager
from .web_tool import WebTool
from .enhanced_web_tool import EnhancedWebTool

__all__ = [
    'GroqProvider',
    'GroqAPIKeyMissingError',
    'GroqAPIError',
    'OllamaServerNotRunningError',
    'get_api_key',
    'ChainOfThoughtManager',
    'LLMInterface',
    'sanitize_input',
    'validate_cot_steps',
    'RAGManager',
    'WebTool',
    'EnhancedWebTool'
]
```

# tests\test_groq_provider.py

```python
import pytest
from unittest.mock import patch, MagicMock
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIKeyMissingError, GroqAPIError

@pytest.fixture
def mock_groq_client():
    with patch('pocketgroq.groq_provider.Groq') as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_async_groq_client():
    with patch('pocketgroq.groq_provider.AsyncGroq') as mock_async_groq:
        mock_client = MagicMock()
        mock_async_groq.return_value = mock_client
        yield mock_client

def test_groq_provider_initialization(mock_groq_client):
    with patch('pocketgroq.groq_provider.get_api_key', return_value='test_api_key'):
        provider = GroqProvider()
        assert provider.api_key == 'test_api_key'
        mock_groq_client.assert_called_once_with(api_key='test_api_key')

def test_groq_provider_initialization_no_api_key(mock_groq_client):
    with patch('pocketgroq.groq_provider.get_api_key', return_value=None):
        with pytest.raises(GroqAPIKeyMissingError):
            GroqProvider()

def test_generate_text(mock_groq_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Generated text"
    mock_groq_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key='test_api_key')
    result = provider.generate("Test prompt")

    assert result == "Generated text"
    mock_groq_client.chat.completions.create.assert_called_once()

def test_generate_text_with_stream(mock_groq_client):
    mock_stream = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2"))]),
    ]
    mock_groq_client.chat.completions.create.return_value = mock_stream

    provider = GroqProvider(api_key='test_api_key')
    result = list(provider.generate("Test prompt", stream=True))

    assert result == ["chunk1", "chunk2"]
    mock_groq_client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_async(mock_async_groq_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Generated text"
    mock_async_groq_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key='test_api_key')
    result = await provider.generate("Test prompt", async_mode=True)

    assert result == "Generated text"
    mock_async_groq_client.chat.completions.create.assert_called_once()

def test_generate_text_with_tool(mock_groq_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Tool result"
    mock_groq_client.chat.completions.create.return_value = mock_completion

    def mock_tool(arg):
        return f"Processed {arg}"

    provider = GroqProvider(api_key='test_api_key')
    result = provider.generate("Use tool", tools=[
        {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "A mock tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {
                            "type": "string",
                            "description": "An argument",
                        }
                    },
                    "required": ["arg"],
                },
                "implementation": mock_tool
            }
        }
    ])

    assert result == "Tool result"
    mock_groq_client.chat.completions.create.assert_called_once()

def test_api_error(mock_groq_client):
    mock_groq_client.chat.completions.create.side_effect = Exception("API Error")

    provider = GroqProvider(api_key='test_api_key')
    with pytest.raises(GroqAPIError):
        provider.generate("Test prompt")

def test_get_available_models():
    mock_response = {
        "object": "list",
        "data": [
            {
                "id": "gemma-7b-it",
                "object": "model",
                "created": 1693721698,
                "owned_by": "Google",
                "active": True,
                "context_window": 8192
            },
            {
                "id": "llama2-70b-4096",
                "object": "model",
                "created": 1693721698,
                "owned_by": "Meta",
                "active": True,
                "context_window": 4096
            }
        ]
    }

    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        provider = GroqProvider(api_key='test_api_key')
        models = provider.get_available_models()

        assert len(models) == 2
        assert models[0]['id'] == "gemma-7b-it"
        assert models[1]['id'] == "llama2-70b-4096"
        mock_get.assert_called_once_with("https://api.groq.com/openai/v1/models")

```

# pocketgroq\chain_of_thought\cot_manager.py

```python
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
```

# pocketgroq\chain_of_thought\llm_interface.py

```python
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
```

# pocketgroq\chain_of_thought\utils.py

```python
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
```

# pocketgroq\chain_of_thought\__init__.py

```python
# pocketgroq/chain_of_thought/__init__.py

from .cot_manager import ChainOfThoughtManager
from .llm_interface import LLMInterface
from .utils import sanitize_input, validate_cot_steps

__all__ = ['ChainOfThoughtManager', 'LLMInterface', 'sanitize_input', 'validate_cot_steps']
```

# build\lib\chain_of_thought\cot_manager.py

```python
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
```

# build\lib\chain_of_thought\llm_interface.py

```python
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
```

# build\lib\chain_of_thought\utils.py

```python
# pocketgroq/chain_of_thought/utils.py

import re

def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks or unwanted content.
    """
    # Remove potentially harmful characters or patterns
    sanitized = re.sub(r'[<>]', '', text)
    return sanitized.strip()
```

# build\lib\chain_of_thought\__init__.py

```python
# pocketgroq/chain_of_thought/__init__.py

from .cot_manager import ChainOfThoughtManager
from .llm_interface import LLMInterface
from .utils import sanitize_input

__all__ = ['ChainOfThoughtManager', 'LLMInterface', 'sanitize_input']
```

# build\lib\pocketgroq\autonomous_agent.py

```python
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
```

# build\lib\pocketgroq\config.py

```python
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
```

# build\lib\pocketgroq\enhanced_web_tool.py

```python
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, urljoin
import html2text
import json

class EnhancedWebTool:
    def __init__(self, max_depth: int = 3, max_pages: int = 100):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        self.html2text_converter = html2text.HTML2Text()
        self.html2text_converter.ignore_links = False
        self.html2text_converter.ignore_images = False
        self.html2text_converter.ignore_emphasis = False
        self.html2text_converter.body_width = 0  # Disable line wrapping

    def crawl(self, start_url: str, formats: List[str] = ["markdown"]) -> List[Dict[str, Any]]:
        visited = set()
        to_visit = [(start_url, 0)]
        results = []

        while to_visit and len(results) < self.max_pages:
            url, depth = to_visit.pop(0)
            if url in visited or depth > self.max_depth:
                continue

            visited.add(url)
            page_content = self.scrape_page(url, formats)
            if page_content:
                results.append(page_content)

            if depth < self.max_depth:
                links = self.extract_links(url, page_content.get('html', ''))
                to_visit.extend((link, depth + 1) for link in links if link not in visited)

        return results

    def scrape_page(self, url: str, formats: List[str]) -> Dict[str, Any]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            result = {
                'url': url,
                'metadata': self.extract_metadata(soup, url),
            }

            if 'markdown' in formats:
                result['markdown'] = self.html_to_markdown(str(soup))
            if 'html' in formats:
                result['html'] = str(soup)
            if 'structured_data' in formats:
                result['structured_data'] = self.extract_structured_data(soup)

            return result
        except requests.RequestException as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def extract_links(self, base_url: str, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, 'html.parser')
        base_domain = urlparse(base_url).netloc
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == base_domain:
                links.append(full_url)

        return links

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        metadata = {
            'title': soup.title.string if soup.title else '',
            'description': '',
            'language': soup.html.get('lang', ''),
            'sourceURL': url,
        }

        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') == 'description':
                metadata['description'] = tag.get('content', '')
            elif tag.get('property') == 'og:description':
                metadata['og_description'] = tag.get('content', '')

        return metadata

    def html_to_markdown(self, html_content: str) -> str:
        # Convert HTML to Markdown using html2text
        markdown = self.html2text_converter.handle(html_content)
        
        # Clean up the markdown
        markdown = markdown.strip()
        
        return markdown
        
        return markdown

    def extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        structured_data = {}

        # Extract all text content
        all_text = soup.get_text(separator=' ', strip=True)
        structured_data['full_text'] = all_text

        # Extract headings
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            if h_tags:
                headings[f'h{i}'] = [tag.get_text(strip=True) for tag in h_tags]
        structured_data['headings'] = headings

        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'text': a.get_text(strip=True),
                'href': a['href']
            })
        structured_data['links'] = links

        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({
                'src': img['src'],
                'alt': img.get('alt', '')
            })
        structured_data['images'] = images

        # Extract JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data['json_ld'] = data
            except json.JSONDecodeError:
                pass

        return structured_data
```

# build\lib\pocketgroq\exceptions.py

```python
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
```

# build\lib\pocketgroq\groq_provider.py

```python
import asyncio
import json
import logging
import os
import subprocess
import requests

from collections import defaultdict
from groq import Groq, AsyncGroq
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from typing import Callable, Dict, Any, List, Union, AsyncIterator, Optional

from .enhanced_web_tool import EnhancedWebTool
from .exceptions import GroqAPIKeyMissingError, GroqAPIError, OllamaServerNotRunningError
from .web_tool import WebTool
from .chain_of_thought.cot_manager import ChainOfThoughtManager
from .chain_of_thought.llm_interface import LLMInterface
from .rag_manager import RAGManager

logger = logging.getLogger(__name__)

class GroqProvider(LLMInterface):
    def __init__(self, api_key: str = None, rag_persistent: bool = True, rag_index_path: str = "faiss_index.pkl"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise GroqAPIKeyMissingError("Groq API key is not provided")
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        self.tool_use_models = [
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview"
        ]
        self.web_tool = WebTool()
        self.cot_manager = ChainOfThoughtManager(llm=self)
        self.rag_manager = None
        self.tools = {}
        self.rag_persistent = rag_persistent
        self.rag_index_path = rag_index_path
        self.enhanced_web_tool = EnhancedWebTool()

        # Initialize conversation sessions
        self.conversation_sessions = defaultdict(list)  # session_id -> list of messages

        # Check if Ollama server is running and initialize RAG if it is
        if self.is_ollama_server_running():
            if self.rag_persistent:
                logger.info("Initializing RAG with persistence enabled.")
                self.initialize_rag(index_path=self.rag_index_path)
        else:
            logger.warning("Ollama server is not running. RAG functionality will be limited.")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Fetch the list of available models from the Groq provider.

        Returns:
            List[Dict[str, Any]]: A list of models with their details.
        """
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return models
        except requests.RequestException as e:
            logger.error(f"Failed to fetch models: {e}")
            raise GroqAPIError(f"Failed to fetch models: {e}")

    def crawl_website(self, url: str, formats: List[str] = ["markdown"], max_depth: int = 3, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Crawl a website and return its content in specified formats.
        
        Args:
            url (str): The starting URL to crawl.
            formats (List[str]): List of desired output formats (e.g., ["markdown", "html", "structured_data"]).
            max_depth (int): Maximum depth to crawl.
            max_pages (int): Maximum number of pages to crawl.
        
        Returns:
            List[Dict[str, Any]]: List of crawled pages with their content in specified formats.
        """
        self.enhanced_web_tool.max_depth = max_depth
        self.enhanced_web_tool.max_pages = max_pages
        return self.enhanced_web_tool.crawl(url, formats)

    def is_ollama_server_running(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def ensure_ollama_server_running(func):
        """Decorator to ensure Ollama server is running for functions that require it."""
        def wrapper(self, *args, **kwargs):
            if not self.is_ollama_server_running():
                raise OllamaServerNotRunningError("Ollama server is not running. Please start it and try again.")
            return func(self, *args, **kwargs)
        return wrapper
    
    def evaluate_response(self, request: str, response: str) -> bool:
        """
        Evaluate if a response satisfies a given request using an AI LLM.
        
        Args:
            request (str): The original request or question.
            response (str): The response to be evaluated.
        
        Returns:
            bool: True if the response is deemed satisfactory, False otherwise.
        """
        evaluation_prompt = f"""
        You will be given a request and a response. Your task is to evaluate the response based on the following criteria:
        1. **Informative and Correct**: The response must be accurate and provide clear, useful, and sufficient information to fully answer the request.
        2. **No Uncertainty**: The response should not express any uncertainty, such as language indicating doubt (e.g., "maybe," "possibly," "it seems") or statements that are inconclusive.

        Request: {request}
        Response: {response}

        Based on these criteria, is the response satisfactory? Answer with only 'Yes' or 'No'.
        """

        evaluation = self.generate(evaluation_prompt, temperature=0.0, max_tokens=1)
        
        # Clean up the response and convert to boolean
        evaluation = evaluation.strip().lower()
        return evaluation == 'yes'

    def register_tool(self, name: str, func: callable):
        self.tools[name] = func

    def scrape_url(self, url: str, formats: List[str] = ["markdown"]) -> Dict[str, Any]:
        """
        Scrape a single URL and return its content in specified formats.
        
        Args:
            url (str): The URL to scrape.
            formats (List[str]): List of desired output formats (e.g., ["markdown", "html", "structured_data"]).
        
        Returns:
            Dict[str, Any]: The scraped content in specified formats.
        """
        return self.enhanced_web_tool.scrape_page(url, formats)

    def end_conversation(self, conversation_id: str):
        """
        Ends a conversation and clears its history.

        Args:
            conversation_id (str): The ID of the conversation to end.
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Ended conversation with ID: {conversation_id}")
        else:
            logger.warning(f"Attempted to end non-existent conversation ID: {conversation_id}")

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history for a given session.

        Args:
            session_id (str): Unique identifier for the conversation session.

        Returns:
            List[Dict[str, str]]: List of messages in the conversation.
        """
        return self.conversation_sessions.get(session_id, [])            

    def start_conversation(self, session_id: str):
        """
        Initialize a new conversation session.

        Args:
            session_id (str): Unique identifier for the conversation session.
        """
        if session_id in self.conversation_sessions:
            logger.warning(f"Session '{session_id}' already exists. Overwriting.")
        self.conversation_sessions[session_id] = []
        logger.info(f"Started new conversation session '{session_id}'.")
    
    def reset_conversation(self, session_id: str):
        if session_id in self.conversation_sessions:
            del self.conversation_sessions[session_id]
            logger.info(f"Conversation session '{session_id}' has been reset.")
        else:
            logger.warning(f"Attempted to reset non-existent session '{session_id}'.")   

    def generate(self, prompt: str, session_id: Optional[str] = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        if session_id:
            messages = self.conversation_sessions[session_id]
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self._create_completion(messages, **kwargs)

        if session_id:
            if isinstance(response, str):
                self.conversation_sessions[session_id].append({"role": "assistant", "content": response})
            elif asyncio.iscoroutine(response):
                # Handle asynchronous streaming responses if needed
                pass

        return response

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)

    def _create_completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[str, AsyncIterator[str]]:
        completion_kwargs = {
            "model": self._select_model(kwargs.get("model"), kwargs.get("tools")),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.5),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1),
            "stop": kwargs.get("stop", None),
            "stream": kwargs.get("stream", False),
        }

        if kwargs.get("json_mode", False):
            completion_kwargs["response_format"] = {"type": "json_object"}

        if kwargs.get("tools"):
            completion_kwargs["tools"] = self._prepare_tools(kwargs["tools"])
            completion_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        if kwargs.get("async_mode", False):
            return self._async_create_completion(**completion_kwargs)
        else:
            return self._sync_create_completion(**completion_kwargs)

    def _select_model(self, requested_model: str, tools: List[Dict[str, Any]]) -> str:
        if tools and not requested_model:
            return self.tool_use_models[0]
        elif tools and requested_model not in self.tool_use_models:
            print(f"Warning: {requested_model} is not optimized for tool use. Switching to {self.tool_use_models[0]}.")
            return self.tool_use_models[0]
        return requested_model or os.environ.get('GROQ_MODEL', 'llama3-8b-8192')
    
    def _prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_tools = []
        for tool in tools:
            prepared_tool = tool.copy()
            if 'function' in prepared_tool:
                prepared_tool['function'] = {k: v for k, v in prepared_tool['function'].items() if k != 'implementation'}
            prepared_tools.append(prepared_tool)
        return prepared_tools

    def _sync_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = self.client.chat.completions.create(**kwargs)
            if kwargs.get("stream", False):
                return (chunk.choices[0].delta.content for chunk in response)
            else:
                return self._process_tool_calls(response)
        except Exception as e:
            raise GroqAPIError(f"Error in Groq API call: {str(e)}")

    async def _async_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = await self.async_client.chat.completions.create(**kwargs)
            if kwargs.get("stream", False):
                async def async_generator():
                    async for chunk in response:
                        yield chunk.choices[0].delta.content
                return async_generator()
            else:
                return await self._async_process_tool_calls(response)
        except Exception as e:
            raise GroqAPIError(f"Error in async Groq API call: {str(e)}")

    def _process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = self._execute_tool_calls(message.tool_calls)
            new_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            return self._create_completion([new_message] + tool_results)
        return message.content

    async def _async_process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = await self._async_execute_tool_calls(message.tool_calls)
            new_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            for result in tool_results:
                new_message["tool_results"] = result
            return await self._async_create_completion([new_message])
        return message.content

    def _execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            if tool_call.function.name in self.tools:
                args = json.loads(tool_call.function.arguments)
                result = self.tools[tool_call.function.name](**args)
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    async def _async_execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = await asyncio.to_thread(self.web_tool.search, args.get("query", ""))
            elif tool_call.function.name == "get_web_content":
                args = json.loads(tool_call.function.arguments)
                result = await asyncio.to_thread(self.web_tool.get_web_content, args.get("url", ""))
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps(result),
            })
        return results

    def web_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Perform a web search using the integrated WebTool."""
        return self.web_tool.search(query)

    def get_web_content(self, url: str) -> str:
        """Retrieve the content of a web page using the integrated WebTool."""
        return self.web_tool.get_web_content(url)

    def is_url(self, text: str) -> bool:
        """Check if the given text is a valid URL using the integrated WebTool."""
        return self.web_tool.is_url(text)
    
    def solve_problem_with_cot(self, problem: str, **kwargs) -> str:
        """
        Solve a problem using Chain of Thought reasoning.
        """
        return self.cot_manager.solve_problem(problem)

    def generate_cot(self, problem: str, **kwargs) -> List[str]:
        """
        Generate Chain of Thought steps for a given problem.
        """
        return self.cot_manager.generate_cot(problem)

    def synthesize_cot(self, cot_steps: List[str], **kwargs) -> str:
        """
        Synthesize a final answer from Chain of Thought steps.
        """
        return self.cot_manager.synthesize_response(cot_steps)
    
    @ensure_ollama_server_running
    def initialize_rag(self, ollama_base_url: str = "http://localhost:11434", model_name: str = "nomic-embed-text", index_path: str = "faiss_index.pkl"):
        try:
            # Attempt to pull the model if it's not already available
            subprocess.run(["ollama", "pull", model_name], check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Failed to pull model {model_name}. Ensure Ollama is installed and running.")
            raise

        embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=model_name)
        self.rag_manager = RAGManager(embeddings, index_path=index_path)
        logger.info("RAG initialized successfully.")

    @ensure_ollama_server_running
    def load_documents(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                       progress_callback: Callable[[int, int], None] = None, timeout: int = 300, 
                       persistent: bool = None):
        if persistent is None:
            persistent = self.rag_persistent
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")

        # Use a separate index path if non-persistent
        index_path = self.rag_index_path if persistent else f"temp_{self.rag_index_path}"
        self.rag_manager.index_path = index_path

        self.rag_manager.load_and_process_documents(source, chunk_size, chunk_overlap, progress_callback, timeout)

    @ensure_ollama_server_running
    def query_documents(self, query: str, session_id: Optional[str] = None, **kwargs) -> str:
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")
        
        llm = ChatGroq(groq_api_key=self.api_key, model_name=kwargs.get("model", "llama3-8b-8192"))
        response = self.rag_manager.query_documents(llm, query)
        return response['answer']

    def query_documents(self, query: str, session_id: Optional[str] = None, **kwargs) -> str:
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")
        
        llm = ChatGroq(groq_api_key=self.api_key, model_name=kwargs.get("model", "llama3-8b-8192"))
        response = self.rag_manager.query_documents(llm, query)
        return response['answer']

```

# build\lib\pocketgroq\rag_manager.py

```python
# pocketgroq/rag_manager.py

import os
import pickle
from typing import List, Dict, Any, Callable
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self, embeddings, index_path: str = "faiss_index.pkl"):
        self.embeddings = embeddings
        self.vector_store = None
        self.index_path = index_path

    def load_and_process_documents(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                                   progress_callback: Callable[[int, int], None] = None, 
                                   timeout: int = 300):  # 5 minutes timeout
        start_time = time.time()

        if os.path.exists(self.index_path):
            logger.info("Loading persisted FAISS index.")
            with open(self.index_path, 'rb') as f:
                self.vector_store = pickle.load(f)
            logger.info("FAISS index loaded successfully.")
            return

        if source.startswith(('http://', 'https://')):
            loader = WebBaseLoader(source)
        elif os.path.isfile(source):
            loader = TextLoader(source)
        else:
            raise ValueError(f"Unsupported source: {source}. Must be a valid URL or file path.")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)

        total_chunks = len(split_documents)
        for i, doc in enumerate(split_documents):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Document processing exceeded the {timeout} seconds timeout.")

            if not self.vector_store:
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
            else:
                self.vector_store.add_documents([doc])

            if progress_callback:
                progress_callback(i + 1, total_chunks)

        # Persist the FAISS index
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.vector_store, f)
        logger.info("FAISS index persisted to disk.")

    def query_documents(self, llm, query: str) -> Dict[str, Any]:
        if not self.vector_store:
            raise ValueError("Documents have not been loaded. Call load_and_process_documents first.")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context only.
            Provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain.invoke({"input": query})

```

# build\lib\pocketgroq\utils.py

```python
import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file if it exists."""
    load_dotenv()

def get_env_variable(var_name: str, default: str = None) -> str:
    """Retrieve an environment variable or return a default value."""
    return os.getenv(var_name, default)
```

# build\lib\pocketgroq\web_tool.py

```python
# pocketgroq/web_tool.py

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse, quote_plus

DEBUG = False  # Set to True for debugging

def log_debug(message):
    if DEBUG:
        print(f"DEBUG: {message}")

class WebTool:
    def __init__(self, num_results: int = 10, max_tokens: int = 4096):
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        log_debug(f"Performing web search for: {query}")
        search_results = self._perform_web_search(query)
        filtered_results = self._filter_search_results(search_results)
        deduplicated_results = self._remove_duplicates(filtered_results)
        log_debug(f"Found {len(deduplicated_results)} unique results")
        return deduplicated_results[:self.num_results]

    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        encoded_query = quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded_query}&num={self.num_results * 2}"
        log_debug(f"Search URL: {search_url}")
        
        try:
            log_debug("Sending GET request to Google")
            response = requests.get(search_url, headers=self.headers, timeout=10)
            log_debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            log_debug("Parsing HTML with BeautifulSoup")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            log_debug("Searching for result divs")
            search_results = []
            for g in soup.find_all('div', class_='g'):
                log_debug("Processing a search result div")
                anchor = g.find('a')
                title = g.find('h3').text if g.find('h3') else 'No title'
                url = anchor.get('href', 'No URL') if anchor else 'No URL'
                
                description = ''
                description_div = g.find('div', class_=['VwiC3b', 'yXK7lf'])
                if description_div:
                    description = description_div.get_text(strip=True)
                else:
                    description = g.get_text(strip=True)
                
                log_debug(f"Found result: Title: {title[:30]}..., URL: {url[:30]}...")
                search_results.append({
                    'title': title,
                    'description': description,
                    'url': url
                })
            
            log_debug(f"Successfully retrieved {len(search_results)} search results for query: {query}")
            return search_results
        except requests.RequestException as e:
            log_debug(f"Error performing search: {str(e)}")
            return []

    def _filter_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [result for result in results if result['description'] and result['title'] != 'No title' and result['url'].startswith('https://')]
        log_debug(f"Filtered to {len(filtered)} results")
        return filtered

    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        unique_results = []
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        log_debug(f"Removed duplicates, left with {len(unique_results)} results")
        return unique_results

    def get_web_content(self, url: str) -> str:
        log_debug(f"Fetching content from: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            content = text[:self.max_tokens]
            log_debug(f"Retrieved {len(content)} characters of content")
            return content
        except requests.RequestException as e:
            log_debug(f"Error retrieving content from {url}: {str(e)}")
            return ""

    def is_url(self, text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _clean_url(self, url: str) -> str:
        url = url.rstrip(')')  # Remove trailing parenthesis if present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url  # Add https:// if missing
        return url
```

# build\lib\pocketgroq\__init__.py

```python
# pocketgroq/__init__.py

from .groq_provider import GroqProvider
from .exceptions import GroqAPIKeyMissingError, GroqAPIError, OllamaServerNotRunningError
from .config import get_api_key
from .chain_of_thought import ChainOfThoughtManager, LLMInterface, sanitize_input, validate_cot_steps
from .rag_manager import RAGManager
from .web_tool import WebTool
from .enhanced_web_tool import EnhancedWebTool

__all__ = [
    'GroqProvider',
    'GroqAPIKeyMissingError',
    'GroqAPIError',
    'OllamaServerNotRunningError',
    'get_api_key',
    'ChainOfThoughtManager',
    'LLMInterface',
    'sanitize_input',
    'validate_cot_steps',
    'RAGManager',
    'WebTool',
    'EnhancedWebTool'
]
```

# build\lib\pocketgroq\chain_of_thought\cot_manager.py

```python
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
```

# build\lib\pocketgroq\chain_of_thought\llm_interface.py

```python
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
```

# build\lib\pocketgroq\chain_of_thought\utils.py

```python
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
```

# build\lib\pocketgroq\chain_of_thought\__init__.py

```python
# pocketgroq/chain_of_thought/__init__.py

from .cot_manager import ChainOfThoughtManager
from .llm_interface import LLMInterface
from .utils import sanitize_input, validate_cot_steps

__all__ = ['ChainOfThoughtManager', 'LLMInterface', 'sanitize_input', 'validate_cot_steps']
```

