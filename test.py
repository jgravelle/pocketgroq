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
    response_url = groq.generate(
        prompt="Describe this image in one sentence.",
        model="llava-v1.5-7b-4096-preview",
        image_url=image_url
    )
    print("Response:", response_url)
    assert isinstance(response_url, str) and len(response_url) > 0

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

def display_menu():
    print("\nPocketGroq Test Menu:")
    print("1. Basic Chat Completion")
    print("2. Streaming Chat Completion")
    print("3. Override Default Model")
    print("4. Chat Completion with Stop Sequence")
    print("5. Asynchronous Generation")
    print("6. Streaming Async Chat Completion")
    print("7. JSON Mode")
    print("8. Tool Usage")
    print("9. Vision")
    print("10. Chain of Thought Problem Solving")
    print("11. Chain of Thought Step Generation")
    print("12. Chain of Thought Synthesis")
    print("13. Test RAG Initialization")
    print("14. Test Document Loading")
    print("15. Test Document Querying")
    print("16. Test RAG Error Handling")
    print("17. Test Persistent Conversation")
    print("18. Test Disposable Conversation")
    print("19. Web Search")
    print("20. Get Web Content")
    print("21. Crawl Website")
    print("22. Scrape URL")

    print("23. Run All Web Tests")
    print("24. Run All RAG Tests")
    print("25. Run All Conversation Tests")
    print("26. Run All Tests")
    print("0. Exit")

async def main():
    start_conversations()
    
    while True:
        display_menu()
        choice = input("Enter your choice (0-27): ")
        
        try:
            if choice == '0':
                break
            elif choice == '1':
                test_basic_chat_completion()
            elif choice == '2':
                test_streaming_chat_completion()
            elif choice == '3':
                test_override_default_model()
            elif choice == '4':
                test_chat_completion_with_stop_sequence()
            elif choice == '5':
                await test_async_generation()
            elif choice == '6':
                await test_streaming_async_chat_completion()
            elif choice == '7':
                test_json_mode()
            elif choice == '8':
                test_tool_usage()
            elif choice == '9':
                test_vision()
            elif choice == '10':
                test_cot_problem_solving()
            elif choice == '11':
                test_cot_step_generation()
            elif choice == '12':
                test_cot_synthesis()
            elif choice == '13':
                test_rag_initialization()
            elif choice == '14':
                test_document_loading(persistent=True)
            elif choice == '15':
                test_document_querying()
            elif choice == '16':
                test_rag_error_handling()
            elif choice == '17':
                test_persistent_conversation()
            elif choice == '18':
                test_disposable_conversation()
            elif choice == '19':
                test_web_search()
            elif choice == '20':
                test_get_web_content()
            elif choice == '21':
                test_crawl_website()
            elif choice == '22':
                test_scrape_url()
            elif choice == '23':
                test_web_search()
                test_get_web_content()
                test_crawl_website()
                test_scrape_url()
                print("\nAll Web tests completed successfully!")
            elif choice == '24':
                test_rag_initialization()
                test_document_loading(persistent=True)
                test_document_querying()
                test_rag_error_handling()
                print("\nAll RAG tests completed successfully!")
            elif choice == '25':
                test_persistent_conversation()
                test_disposable_conversation()
                print("\nAll Conversation tests completed successfully!")
            elif choice == '26':
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
                test_web_search()
                test_get_web_content()
                test_crawl_website()
                test_scrape_url()
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