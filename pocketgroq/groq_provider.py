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