# config.py

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

# exceptions.py

```python
class GroqAPIKeyMissingError(Exception):
    """Raised when the Groq API key is missing."""
    pass

class GroqAPIError(Exception):
    """Raised when there's an error with the Groq API."""
    pass
```

# groq_provider.py

```python
import os
import json
from typing import Dict, Any, List, Union, AsyncIterator
import asyncio
from groq import Groq, AsyncGroq
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .config import get_api_key

class GroqProvider:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise GroqAPIKeyMissingError("Groq API key is not provided")
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        self.tool_use_models = [
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview"
        ]

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
            return self.tool_use_models[0]  # Default to the 70B model for tool use
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
                return self._process_tool_calls(response, kwargs.get("tools"))
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
                return await self._async_process_tool_calls(response, kwargs.get("tools"))
        except Exception as e:
            raise GroqAPIError(f"Error in async Groq API call: {str(e)}")

    def _process_tool_calls(self, response, tools: List[Dict[str, Any]]) -> str:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = self._execute_tool_calls(message.tool_calls, tools)
            new_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            for result in tool_results:
                new_message["tool_results"] = result
            return self._create_completion([new_message])
        return message.content

    async def _async_process_tool_calls(self, response, tools: List[Dict[str, Any]]) -> str:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = await self._async_execute_tool_calls(message.tool_calls, tools)
            new_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
            for result in tool_results:
                new_message["tool_results"] = result
            return await self._async_create_completion([new_message])
        return message.content

    def _execute_tool_calls(self, tool_calls, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            tool = next((t for t in tools if t["function"]["name"] == tool_call.function.name), None)
            if tool:
                function = tool["function"]["implementation"]
                args = json.loads(tool_call.function.arguments)
                result = function(**args)
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
        return results

    async def _async_execute_tool_calls(self, tool_calls, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            tool = next((t for t in tools if t["function"]["name"] == tool_call.function.name), None)
            if tool:
                function = tool["function"]["implementation"]
                args = json.loads(tool_call.function.arguments)
                if asyncio.iscoroutinefunction(function):
                    result = await function(**args)
                else:
                    result = function(**args)
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
        return results

    def get_available_models(self) -> Dict[str, int]:
        try:
            response = self.client.models.list()
            return {model.id: model.max_tokens for model in response.data}
        except Exception as e:
            raise GroqAPIError(f"Error fetching available models: {str(e)}")

    def process_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, str):
            return {"content": response}
        return response

    def send_request(self, data: Dict[str, Any]) -> Any:
        return self._create_completion(data["messages"], **data)
```

# setup.py

```python
from setuptools import setup, find_packages

setup(
    name="pocketgroq",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for easy integration with Groq API",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pocketgroq",
    packages=find_packages(),
    install_requires=[
        "groq==0.3.0",
        "python-dotenv==0.19.1",
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-asyncio==0.21.0",
        ],
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
    ],
    python_requires=">=3.7",
)
```

# utils.py

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

# __init__.py

```python
 

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

