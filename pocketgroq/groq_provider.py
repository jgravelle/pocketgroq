# pocketgroq/groq_provider.py

import asyncio
import os
import json
import subprocess

from groq import Groq, AsyncGroq
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from typing import Callable, Dict, Any, List, Union, AsyncIterator
from .exceptions import GroqAPIKeyMissingError, GroqAPIError
from .web_tool import WebTool
from .chain_of_thought.cot_manager import ChainOfThoughtManager
from .chain_of_thought.llm_interface import LLMInterface
from .rag_manager import RAGManager

class GroqProvider(LLMInterface):
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
        self.cot_manager = ChainOfThoughtManager(llm=self)
        self.rag_manager = None
        self.tools = {} 

    def register_tool(self, name: str, func: callable):
        self.tools[name] = func

    def generate(self, prompt: str, **kwargs) -> Union[str, AsyncIterator[str]]:
        messages = [{"role": "user", "content": prompt}]
        return self._create_completion(messages, **kwargs)

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
    
    def initialize_rag(self, ollama_base_url: str = "http://localhost:11434", model_name: str = "nomic-embed-text"):
        try:
            # Attempt to pull the model if it's not already available
            subprocess.run(["ollama", "pull", model_name], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to pull model {model_name}. Ensure Ollama is installed and running.")
            raise

        embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=model_name)
        self.rag_manager = RAGManager(embeddings)

    def load_documents(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                       progress_callback: Callable[[int, int], None] = None, timeout: int = 300):
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")
        self.rag_manager.load_and_process_documents(source, chunk_size, chunk_overlap, progress_callback, timeout)


    def query_documents(self, query: str, **kwargs) -> str:
        if not self.rag_manager:
            raise ValueError("RAG has not been initialized. Call initialize_rag first.")
        
        llm = ChatGroq(groq_api_key=self.api_key, model_name=kwargs.get("model", "llama3-8b-8192"))
        response = self.rag_manager.query_documents(llm, query)
        return response['answer']