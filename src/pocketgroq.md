# setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pocketgroq",
    version="0.2.6",  # Increment the version number
    author="PocketGroq Team",
    author_email="pocketgroq@example.com",
    description="A library for easy integration with Groq API, including image handling",
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
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.7",
    install_requires=[
        "groq==0.8.0",
        "python-dotenv==0.19.1",
        "requests>=2.32.3",
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-asyncio==0.21.0",
        ],
    },
)
```

# test.py

```python
import asyncio
import json
import logging
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the GroqProvider
groq = GroqProvider()

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
    print(response)
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
    print(response)
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
    print(response)
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

        @validator('quantity', pre=True)
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
            # Parse the JSON response
            json_data = json.loads(response)
            
            # Check if the recipe is nested under a 'recipe' key
            if 'recipe' in json_data:
                json_data = json_data['recipe']
            
            # Attempt to create a Recipe object
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
                },
                "implementation": reverse_string
            }
        }
    ]

    response = groq.generate("Please reverse the string 'hello world'", tools=tools)
    print("Response:", response)
    assert "dlrow olleh" in response.lower()

def test_vision():
    print("\nTesting Vision...")
    # Note: This test requires a valid image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    response_url = groq.generate(
        prompt="Describe this image in one sentence.",
        model="llava-v1.5-7b-4096-preview",
        image_url=image_url
    )
    print(response_url)
    assert isinstance(response_url, str) and len(response_url) > 0

def main():
    try:
        test_basic_chat_completion()
        test_streaming_chat_completion()
        test_override_default_model()
        test_chat_completion_with_stop_sequence()
        asyncio.run(test_async_generation())
        asyncio.run(test_streaming_async_chat_completion())
        test_json_mode()
        test_tool_usage()
        test_vision()
        print("\nAll tests completed successfully!")
    except GroqAPIKeyMissingError as e:
        print(f"Error: {e}")
    except GroqAPIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
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

# pocketgroq\exceptions.py

```python
class GroqAPIKeyMissingError(Exception):
    """Raised when the Groq API key is missing."""
    pass

class GroqAPIError(Exception):
    """Raised when there's an error with the Groq API."""
    pass
```

# pocketgroq\groq_provider.py

```python
# pocketgroq/groq_provider.py

import os
import json
from typing import Dict, Any, List, Union, AsyncIterator
import asyncio

from groq import Groq, AsyncGroq
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .web_tool import WebTool

class GroqProvider:
    def __init__(self, api_key: str = None):
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

    def generate(self, prompt: str, **kwargs) -> Union[str, AsyncIterator[str]]:
        messages = [{"role": "user", "content": prompt}]
        return self._create_completion(messages, **kwargs)

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
            completion_kwargs["tools"] = kwargs["tools"]
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
            for result in tool_results:
                new_message["tool_results"] = result
            return self._create_completion([new_message])
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
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = self.web_tool.search(args.get("query", ""))
            elif tool_call.function.name == "get_web_content":
                args = json.loads(tool_call.function.arguments)
                result = self.web_tool.get_web_content(args.get("url", ""))
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps(result),
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
import os

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
from .groq_provider import GroqProvider
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .config import get_api_key
    

__all__ = ['GroqProvider', 'GroqAPIKeyMissingError', 'GroqAPIError', 'get_api_key']
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

# build\lib\pocketgroq\exceptions.py

```python
class GroqAPIKeyMissingError(Exception):
    """Raised when the Groq API key is missing."""
    pass

class GroqAPIError(Exception):
    """Raised when there's an error with the Groq API."""
    pass
```

# build\lib\pocketgroq\groq_provider.py

```python
# pocketgroq/groq_provider.py

import os
import json
from typing import Dict, Any, List, Union, AsyncIterator
import asyncio

from groq import Groq, AsyncGroq
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .web_tool import WebTool

class GroqProvider:
    def __init__(self, api_key: str = None):
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

    def generate(self, prompt: str, **kwargs) -> Union[str, AsyncIterator[str]]:
        messages = [{"role": "user", "content": prompt}]
        return self._create_completion(messages, **kwargs)

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
            completion_kwargs["tools"] = kwargs["tools"]
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
            for result in tool_results:
                new_message["tool_results"] = result
            return self._create_completion([new_message])
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
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = self.web_tool.search(args.get("query", ""))
            elif tool_call.function.name == "get_web_content":
                args = json.loads(tool_call.function.arguments)
                result = self.web_tool.get_web_content(args.get("url", ""))
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}
            
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": json.dumps(result),
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
import os

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
from .groq_provider import GroqProvider
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .config import get_api_key
    

__all__ = ['GroqProvider', 'GroqAPIKeyMissingError', 'GroqAPIError', 'get_api_key']
```

