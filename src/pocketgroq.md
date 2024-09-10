# setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pocketgroq",
    version="0.2.3",  # Increment the version number
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
import os
import json
import base64
from typing import Dict, Any, List, Union, AsyncIterator
import asyncio
import requests

try:
    from groq import Groq, AsyncGroq
except ImportError:
    print("Warning: Unable to import Groq and AsyncGroq from groq package. Make sure you have the correct version installed.")
    Groq = None
    AsyncGroq = None

class GroqAPIKeyMissingError(Exception):
    pass

class GroqAPIError(Exception):
    pass

def get_api_key():
    return os.environ.get("GROQ_API_KEY")

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
        self.vision_model = "llava-v1.5-7b-4096-preview"
        self.available_models = self.get_available_models()
        self.validate_and_update_tool_use_models()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        try:
            response = self.client.models.list()
            return {model.id: {
                "context_window": model.context_window,
                "owned_by": model.owned_by,
                "created": model.created,
                "active": model.active
            } for model in response.data}
        except Exception as e:
            raise GroqAPIError(f"Error fetching available models: {str(e)}")

    def generate(self, prompt: str, model: str = None, image_path: str = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        messages = self._prepare_messages(prompt, image_path)
        if "tools" in kwargs:
            tools = [
                {
                    "type": tool["type"],
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"]
                    }
                }
                for tool in kwargs["tools"]
            ]
            self.tool_implementations = {tool["function"]["name"]: tool["function"]["implementation"] for tool in kwargs["tools"]}
            kwargs["tools"] = tools
            print("Serialized Tools:", kwargs["tools"])
        return self._create_completion(messages, model=model, **kwargs)
    
    def _prepare_messages(self, prompt: str, image_path: str = None) -> List[Dict[str, Any]]:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            if image_path.startswith(('http://', 'https://')):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_path}
                })
            else:
                base64_image = self._encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        return [{"role": "user", "content": content}]

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def validate_and_update_tool_use_models(self):
        valid_tool_use_models = [model for model in self.tool_use_models if model in self.available_models]
        if not valid_tool_use_models:
            raise GroqAPIError("No valid tool use models found in the available models list")
        self.tool_use_models = valid_tool_use_models

    def _create_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        completion_kwargs = {
            "model": self._select_model(model, kwargs.get("tools"), messages),
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
            print("Completion kwargs with tools:", completion_kwargs)

        if kwargs.get("async_mode", False):
            return self._async_create_completion(**completion_kwargs)
        else:
            return self._sync_create_completion(**completion_kwargs)

    def _select_model(self, requested_model: str, tools: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> str:
        if any(isinstance(content, dict) and content.get("type") == "image_url" for message in messages for content in message["content"]):
            return self.vision_model
        if tools:
            if not requested_model or requested_model not in self.tool_use_models:
                selected_model = self.tool_use_models[0]
                if requested_model:
                    print(f"Warning: {requested_model} is not optimized for tool use. Switching to {selected_model}.")
                return selected_model
        return requested_model or os.environ.get('GROQ_MODEL', 'llama3-8b-8192')

    def _sync_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = self.client.chat.completions.create(**kwargs)
            # print("Initial Response:", response)
            if kwargs.get("stream", False):
                return (chunk.choices[0].delta.content for chunk in response)
            else:
                return self._process_tool_calls(response)
        except Exception as e:
            print("Error in initial API call:", e)
            raise GroqAPIError(f"Error in Groq API call: {str(e)}")

    async def _async_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = await self.async_client.chat.completions.create(**kwargs)
            print("Initial Async Response:", response)
            if kwargs.get("stream", False):
                async def async_generator():
                    async for chunk in response:
                        yield chunk.choices[0].delta.content
                return async_generator()
            else:
                return await self._async_process_tool_calls(response)
        except Exception as e:
            print("Error in initial async API call:", e)
            raise GroqAPIError(f"Error in async Groq API call: {str(e)}")

    def _process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        # print("Processing Tool Calls. Message:", message)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = self._execute_tool_calls(message.tool_calls)
            response_content = f"Tool results: {tool_results[0]['content']}" if tool_results else message.content
            print("Processed Tool Calls. Response Content:", response_content)
            return response_content
        return message.content

    async def _async_process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        # print("Processing Async Tool Calls. Message:", message)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = await self._async_execute_tool_calls(message.tool_calls)
            response_content = f"Tool results: {tool_results[0]['content']}" if tool_results else message.content
            print("Processed Async Tool Calls. Response Content:", response_content)
            return response_content
        return message.content

    def _execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        print("Executing Tool Calls:", tool_calls)
        for tool_call in tool_calls:
            function = self.tool_implementations.get(tool_call.function.name)
            if function:
                args = json.loads(tool_call.function.arguments)
                result = function(**args)
                results.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
                print("Executed Tool Call Result:", result)
        return results

    async def _async_execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        print("Executing Async Tool Calls:", tool_calls)
        for tool_call in tool_calls:
            function = self.tool_implementations.get(tool_call.function.name)
            if function:
                args = json.loads(tool_call.function.arguments)
                if asyncio.iscoroutinefunction(function):
                    result = await function(**args)
                else:
                    result = function(**args)
                results.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
                print("Executed Async Tool Call Result:", result)
        return results
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
import os
import json
import base64
from typing import Dict, Any, List, Union, AsyncIterator
import asyncio
import requests

try:
    from groq import Groq, AsyncGroq
except ImportError:
    print("Warning: Unable to import Groq and AsyncGroq from groq package. Make sure you have the correct version installed.")
    Groq = None
    AsyncGroq = None

class GroqAPIKeyMissingError(Exception):
    pass

class GroqAPIError(Exception):
    pass

def get_api_key():
    return os.environ.get("GROQ_API_KEY")

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
        self.vision_model = "llava-v1.5-7b-4096-preview"
        self.available_models = self.get_available_models()
        self.validate_and_update_tool_use_models()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        try:
            response = self.client.models.list()
            return {model.id: {
                "context_window": model.context_window,
                "owned_by": model.owned_by,
                "created": model.created,
                "active": model.active
            } for model in response.data}
        except Exception as e:
            raise GroqAPIError(f"Error fetching available models: {str(e)}")

    def generate(self, prompt: str, model: str = None, image_path: str = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        messages = self._prepare_messages(prompt, image_path)
        if "tools" in kwargs:
            tools = [
                {
                    "type": tool["type"],
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"]
                    }
                }
                for tool in kwargs["tools"]
            ]
            self.tool_implementations = {tool["function"]["name"]: tool["function"]["implementation"] for tool in kwargs["tools"]}
            kwargs["tools"] = tools
            print("Serialized Tools:", kwargs["tools"])
        return self._create_completion(messages, model=model, **kwargs)
    
    def _prepare_messages(self, prompt: str, image_path: str = None) -> List[Dict[str, Any]]:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            if image_path.startswith(('http://', 'https://')):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_path}
                })
            else:
                base64_image = self._encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        return [{"role": "user", "content": content}]

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def validate_and_update_tool_use_models(self):
        valid_tool_use_models = [model for model in self.tool_use_models if model in self.available_models]
        if not valid_tool_use_models:
            raise GroqAPIError("No valid tool use models found in the available models list")
        self.tool_use_models = valid_tool_use_models

    def _create_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        completion_kwargs = {
            "model": self._select_model(model, kwargs.get("tools"), messages),
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
            print("Completion kwargs with tools:", completion_kwargs)

        if kwargs.get("async_mode", False):
            return self._async_create_completion(**completion_kwargs)
        else:
            return self._sync_create_completion(**completion_kwargs)

    def _select_model(self, requested_model: str, tools: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> str:
        if any(isinstance(content, dict) and content.get("type") == "image_url" for message in messages for content in message["content"]):
            return self.vision_model
        if tools:
            if not requested_model or requested_model not in self.tool_use_models:
                selected_model = self.tool_use_models[0]
                if requested_model:
                    print(f"Warning: {requested_model} is not optimized for tool use. Switching to {selected_model}.")
                return selected_model
        return requested_model or os.environ.get('GROQ_MODEL', 'llama3-8b-8192')

    def _sync_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = self.client.chat.completions.create(**kwargs)
            # print("Initial Response:", response)
            if kwargs.get("stream", False):
                return (chunk.choices[0].delta.content for chunk in response)
            else:
                return self._process_tool_calls(response)
        except Exception as e:
            print("Error in initial API call:", e)
            raise GroqAPIError(f"Error in Groq API call: {str(e)}")

    async def _async_create_completion(self, **kwargs) -> Union[str, AsyncIterator[str]]:
        try:
            response = await self.async_client.chat.completions.create(**kwargs)
            print("Initial Async Response:", response)
            if kwargs.get("stream", False):
                async def async_generator():
                    async for chunk in response:
                        yield chunk.choices[0].delta.content
                return async_generator()
            else:
                return await self._async_process_tool_calls(response)
        except Exception as e:
            print("Error in initial async API call:", e)
            raise GroqAPIError(f"Error in async Groq API call: {str(e)}")

    def _process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        print("Processing Tool Calls. Message:", message)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = self._execute_tool_calls(message.tool_calls)
            response_content = f"Tool results: {tool_results[0]['content']}" if tool_results else message.content
            print("Processed Tool Calls. Response Content:", response_content)
            return response_content
        return message.content

    async def _async_process_tool_calls(self, response) -> str:
        message = response.choices[0].message
        print("Processing Async Tool Calls. Message:", message)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = await self._async_execute_tool_calls(message.tool_calls)
            response_content = f"Tool results: {tool_results[0]['content']}" if tool_results else message.content
            print("Processed Async Tool Calls. Response Content:", response_content)
            return response_content
        return message.content

    def _execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        print("Executing Tool Calls:", tool_calls)
        for tool_call in tool_calls:
            function = self.tool_implementations.get(tool_call.function.name)
            if function:
                args = json.loads(tool_call.function.arguments)
                result = function(**args)
                results.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
                print("Executed Tool Call Result:", result)
        return results

    async def _async_execute_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        print("Executing Async Tool Calls:", tool_calls)
        for tool_call in tool_calls:
            function = self.tool_implementations.get(tool_call.function.name)
            if function:
                args = json.loads(tool_call.function.arguments)
                if asyncio.iscoroutinefunction(function):
                    result = await function(**args)
                else:
                    result = function(**args)
                results.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result),
                })
                print("Executed Async Tool Call Result:", result)
        return results
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

# build\lib\pocketgroq\__init__.py

```python
from .groq_provider import GroqProvider
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .config import get_api_key
    

__all__ = ['GroqProvider', 'GroqAPIKeyMissingError', 'GroqAPIError', 'get_api_key']
```

