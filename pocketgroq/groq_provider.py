# pocketgroq/groq_provider.py

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