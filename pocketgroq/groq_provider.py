import os
import json
import requests
from typing import Dict, Any, List, Union, AsyncIterator
import asyncio

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
        if Groq is None or AsyncGroq is None:
            raise ImportError("Groq and AsyncGroq classes could not be imported. Please check your groq package installation.")
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        self.tool_use_models = [
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview"
        ]
        self.tool_implementations = {}

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models = response.json().get("data", [])
            return {model["id"]: model for model in models}
        else:
            raise GroqAPIError(f"Error fetching available models: {response.text}")

    def generate(self, prompt: str, model: str = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        messages = [{"role": "user", "content": prompt}]
        if "tools" in kwargs:
            # Extract only serializable parts of tools and store implementations separately
            tools = [
                {
                    "type": tool["type"],
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"]
                    }
                }
            for tool in kwargs["tools"]]
            self.tool_implementations = {tool["function"]["name"]: tool["function"]["implementation"] for tool in kwargs["tools"]}
            kwargs["tools"] = tools
            print("Serialized Tools:", kwargs["tools"])
        return self._create_completion(messages, model=model, **kwargs)

    def _create_completion(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> Union[str, AsyncIterator[str]]:
        completion_kwargs = {
            "model": self._select_model(model, kwargs.get("tools")),
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
            print("Initial Response:", response)
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