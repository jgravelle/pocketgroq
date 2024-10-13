# PocketGroq v0.4.8: Enhanced Web Capabilities and Flexible Ollama Integration
![PocketGroq Logo](https://github.com/user-attachments/assets/d06b6aaf-400e-40db-bdaf-626aaa1040ef)

## What's New in v0.4.8

PocketGroq v0.4.8 brings significant enhancements to web-related functionalities and improves the flexibility of Ollama integration:

- **Advanced Web Scraping**: Improved capabilities for crawling websites and extracting content.
- **Flexible Ollama Integration**: PocketGroq now operates more flexibly with or without an active Ollama server.
- **Enhanced Web Search**: Upgraded web search functionality with more robust result parsing.
- **Improved Error Handling**: Better management of web-related errors and Ollama server status.
- **Updated Test Suite**: Comprehensive tests for new web capabilities and Ollama integration.

## Web Capabilities

### Web Crawling

PocketGroq now offers advanced web crawling capabilities:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

# Crawl a website
results = groq.crawl_website(
    "https://example.com",
    formats=["markdown", "html"],
    max_depth=2,
    max_pages=5
)

for page in results:
    print(f"URL: {page['url']}")
    print(f"Title: {page['metadata']['title']}")
    print(f"Markdown content: {page['markdown'][:100]}...")  # First 100 characters
    print("---")
```

### URL Scraping

Extract content from a single URL in various formats:

```python
url = "https://example.com"
result = groq.scrape_url(url, formats=["markdown", "html", "structured_data"])

print(f"Markdown content length: {len(result['markdown'])}")
print(f"HTML content length: {len(result['html'])}")
if 'structured_data' in result:
    print("Structured data:", json.dumps(result['structured_data'], indent=2))
```

### Enhanced Web Search

Perform web searches with improved result parsing:

```python
query = "Latest developments in AI"
search_results = groq.web_search(query)

for result in search_results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Description: {result['description']}")
    print("---")
```

## Flexible Ollama Integration

PocketGroq v0.4.8 introduces more flexible integration with Ollama:

- **Optional Ollama**: Core features of PocketGroq now work without requiring an active Ollama server.
- **Graceful Degradation**: When Ollama is not available, PocketGroq provides clear error messages for Ollama-dependent features.
- **Persistent Features**: Ollama is still required for certain persistence features, including RAG functionality.

### Initializing RAG with Flexible Ollama Integration

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

try:
    groq.initialize_rag()
    print("RAG initialized successfully with Ollama.")
except OllamaServerNotRunningError:
    print("Ollama server is not running. RAG features will be limited.")
    # Proceed with non-RAG features
```

## Error Handling

PocketGroq v0.4.8 introduces a new exception for Ollama-related errors:

```python
from pocketgroq import GroqProvider, OllamaServerNotRunningError

groq = GroqProvider()

try:
    groq.initialize_rag()
    # Use RAG features
except OllamaServerNotRunningError:
    print("Ollama server is not running. Proceeding with limited functionality.")
    # Use non-RAG features
```

## Updated Test Suite

The test suite has been expanded to cover the new web capabilities and Ollama integration. To run the tests:

1. Navigate to the PocketGroq directory.
2. Run the test script:

```bash
python test.py
```

3. You will see an updated menu with options to run individual tests or groups of tests:

```
PocketGroq Test Menu:
1. Basic Chat Completion
2. Streaming Chat Completion
3. Override Default Model
4. Chat Completion with Stop Sequence
5. Asynchronous Generation
6. Streaming Async Chat Completion
7. JSON Mode
8. Tool Usage
9. Vision
10. Chain of Thought Problem Solving
11. Chain of Thought Step Generation
12. Chain of Thought Synthesis
13. Test RAG Initialization
14. Test Document Loading
15. Test Document Querying
16. Test RAG Error Handling
17. Test Persistent Conversation
18. Test Disposable Conversation
19. Web Search
20. Get Web Content
21. Crawl Website
22. Scrape URL
23. Run All Web Tests
24. Run All RAG Tests
25. Run All Conversation Tests
26. Run All Tests
0. Exit
```

4. Select the desired option by entering the corresponding number.

## Configuration

PocketGroq uses environment variables for configuration. Set `GROQ_API_KEY` in your environment or in a `.env` file in your project root. This API key is essential for authenticating with the Groq API.

Additionally, you may need to set a `USER_AGENT` environment variable for certain web-related functionalities. Here are a couple of ways to set these variables:

1. Using a `.env` file:

```
GROQ_API_KEY=your_api_key_here
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36
```

2. Setting environment variables in your script:

```python
import os

os.environ['GROQ_API_KEY'] = 'your_api_key_here'
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
```

Make sure to keep your API key confidential and never commit it to version control.

## Comprehensive List of PocketGroq Methods

Here's a comprehensive list of all the methods/functions available in PocketGroq, grouped logically by function:

### GroqProvider Class (Main Interface)

#### Initialization and Configuration
- `__init__(api_key: str = None, rag_persistent: bool = True, rag_index_path: str = "faiss_index.pkl")`: Initializes the GroqProvider with API key and RAG settings.
- `set_api_key(api_key: str)`: Updates the API key and reinitializes the Groq clients.

#### Text Generation
- `generate(prompt: str, session_id: Optional[str] = None, **kwargs) -> Union[str, AsyncIterator[str]]`: Generates text based on the given prompt.
- `_create_completion(messages: List[Dict[str, str]], **kwargs) -> Union[str, AsyncIterator[str]]`: Internal method for API call to Groq for text generation.
- `_sync_create_completion(**kwargs) -> Union[str, AsyncIterator[str]]`: Synchronous version of completion creation.
- `_async_create_completion(**kwargs) -> Union[str, AsyncIterator[str]]`: Asynchronous version of completion creation.

#### Conversation Management
- `start_conversation(session_id: str)`: Initializes a new conversation session.
- `reset_conversation(session_id: str)`: Resets an existing conversation session.
- `end_conversation(conversation_id: str)`: Ends and removes a conversation session.
- `get_conversation_history(session_id: str) -> List[Dict[str, str]]`: Retrieves conversation history.

#### Web Tools
- `web_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]`: Performs a web search.
- `get_web_content(url: str) -> str`: Retrieves content of a web page.
- `is_url(text: str) -> bool`: Checks if given text is a valid URL.
- `crawl_website(url: str, formats: List[str] = ["markdown"], max_depth: int = 3, max_pages: int = 100) -> List[Dict[str, Any]]`: Crawls a website.
- `scrape_url(url: str, formats: List[str] = ["markdown"]) -> Dict[str, Any]`: Scrapes a single URL.

#### Chain of Thought Reasoning
- `solve_problem_with_cot(problem: str, **kwargs) -> str`: Solves a problem using Chain of Thought reasoning.
- `generate_cot(problem: str, **kwargs) -> List[str]`: Generates Chain of Thought steps.
- `synthesize_cot(cot_steps: List[str], **kwargs) -> str`: Synthesizes a final answer from CoT steps.

#### RAG (Retrieval-Augmented Generation)
- `initialize_rag(ollama_base_url: str = "http://localhost:11434", model_name: str = "nomic-embed-text", index_path: str = "faiss_index.pkl")`: Initializes the RAG system.
- `load_documents(source: str, chunk_size: int = 1000, chunk_overlap: int = 200, progress_callback: Callable[[int, int], None] = None, timeout: int = 300, persistent: bool = None)`: Loads and processes documents for RAG.
- `query_documents(query: str, session_id: Optional[str] = None, **kwargs) -> str`: Queries loaded documents using RAG.

#### Tool Management
- `register_tool(name: str, func: callable)`: Registers a custom tool for use in text generation.

#### Utility Methods
- `is_ollama_server_running() -> bool`: Checks if the Ollama server is running.
- `ensure_ollama_server_running`: Decorator to ensure Ollama server is running for functions that require it.

### WebTool Class
- `search(query: str) -> List[Dict[str, Any]]`: Performs a web search and returns filtered, deduplicated results.
- `get_web_content(url: str) -> str`: Retrieves and processes the content of a web page.
- `is_url(text: str) -> bool`: Checks if the given text is a valid URL.

### EnhancedWebTool Class
- `crawl(start_url: str, formats: List[str] = ["markdown"]) -> List[Dict[str, Any]]`: Crawls a website and returns its content in specified formats.
- `scrape_page(url: str, formats: List[str]) -> Dict[str, Any]`: Scrapes a single page and returns its content in specified formats.

### RAGManager Class
- `load_and_process_documents(source: str, chunk_size: int = 1000, chunk_overlap: int = 200, progress_callback: Callable[[int, int], None] = None, timeout: int = 300)`: Loads, processes, and indexes documents for RAG.
- `query_documents(llm, query: str) -> Dict[str, Any]`: Queries the indexed documents using the provided language model.

### ChainOfThoughtManager Class
- `generate_cot(problem: str) -> List[str]`: Generates Chain of Thought steps for a given problem.
- `synthesize_response(cot_steps: List[str]) -> str`: Synthesizes a final answer from Chain of Thought steps.
- `solve_problem(problem: str) -> str`: Completes the entire Chain of Thought process to solve a problem.

This comprehensive list covers all the main methods and functions available in PocketGroq, grouped logically by their functionality.

## License

This project is licensed under the MIT License. When using PocketGroq in your projects, please include a mention of J. Gravelle in your code and/or documentation.

![J. Gravelle](https://github.com/user-attachments/assets/73c812cd-685e-4969-9497-639ae9312d6c)

---

Thank you for using PocketGroq! We hope this tool enhances your development process and enables you to create amazing AI-powered applications with ease. If you have any questions or need further assistance, don't hesitate to reach out to the community or check the documentation. Happy coding!
