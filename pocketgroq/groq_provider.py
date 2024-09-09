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