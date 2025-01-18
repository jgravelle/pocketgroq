# PocketGroq v0.5.6: Vision and Speech Processing Meets Autonomous Agents!
![PocketGroq Logo](https://github.com/user-attachments/assets/d06b6aaf-400e-40db-bdaf-626aaa1040ef)

## What's New in v0.5.6

## Vision Capabilities

PocketGroq now includes powerful vision analysis capabilities, allowing you to process both images and screen content:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

# Analyze an image from URL
image_url = "https://example.com/image.jpg"
response = groq.process_image(
    prompt="What do you see in this image?",
    image_source=image_url
)

print(f"Analysis: {response}")

# Analyze your screen
screen_analysis = groq.process_image_desktop(
    prompt="What applications are open on my screen?"
)

print(f"Screen analysis: {screen_analysis}")

# Analyze specific screen region
region_analysis = groq.process_image_desktop_region(
    prompt="What's in this part of the screen?",
    x1=0,    # Top-left corner
    y1=0,    # Top-left corner
    x2=400,  # Width
    y2=300   # Height
)

print(f"Region analysis: {region_analysis}")
```

You can also have multi-turn conversations about images:

```python
# Start a conversation about an image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see in this image?"
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        ]
    }
]

response1 = groq.process_image_conversation(messages=messages)
print(f"First response: {response1}")

# Add follow-up question
messages.append({
    "role": "assistant",
    "content": response1
})
messages.append({
    "role": "user",
    "content": "What colors are most prominent?"
})

response2 = groq.process_image_conversation(messages=messages)
print(f"Second response: {response2}")
```

## Speech Processing

PocketGroq now supports advanced speech processing with transcription and translation capabilities:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

# Transcribe audio
response = groq.transcribe_audio(
    audio_file="recording.wav",
    language="en",
    model="distil-whisper-large-v3-en"  # Fastest for English
)

print(f"Transcription: {response}")

# Translate audio to English
translation = groq.translate_audio(
    audio_file="french_speech.wav",
    model="whisper-large-v3",  # Required for translation
    prompt="This is a French conversation about cooking."
)

print(f"Translation: {translation}")
```

### Speech Model Selection

PocketGroq offers three Whisper models with different capabilities:

* `whisper-large-v3`: Best for multilingual tasks and translation ($0.111/hour)
* `whisper-large-v3-turbo`: Fast multilingual transcription without translation ($0.04/hour)
* `distil-whisper-large-v3-en`: Fastest English-only transcription ($0.02/hour)

Choose your model based on your needs:
- For translation: Use `whisper-large-v3`
- For fast multilingual transcription: Use `whisper-large-v3-turbo`
- For English-only transcription: Use `distil-whisper-large-v3-en`

### Additional Speech Settings

Fine-tune your speech processing:

```python
# Transcription with advanced options
response = groq.transcribe_audio(
    audio_file="recording.wav",
    language="en",              # Specify language
    prompt="Technical terms",   # Context for better accuracy
    response_format="json",     # 'json' or 'text'
    temperature=0.3            # Control variation
)

# Translation with custom settings
translation = groq.translate_audio(
    audio_file="speech.wav",
    prompt="Medical terminology",  # Context for accuracy
    response_format="json",        # Structured output
    temperature=0              # Maximum accuracy
)
```

## What's NEW in v0.5.4!

## Autonomous Agent

PocketGroq now includes an AutonomousAgent class that can autonomously research and answer questions:

```python
from pocketgroq import GroqProvider
from pocketgroq.autonomous_agent import AutonomousAgent

groq = GroqProvider()
agent = AutonomousAgent(groq)

request = "What is the current temperature in Sheboygan, Wisconsin?"
response = agent.process_request(request)

print(f"Final response: {response}")
```

The AutonomousAgent:
- Attempts to answer the question using its initial knowledge.
- If unsuccessful, it uses web search tools to find relevant information.
- Evaluates each potential response for accuracy and completeness.
- Keeps the user informed of its progress throughout the process.
- Handles rate limiting and errors gracefully.

You can customize the agent's behavior:

```python
# Set a custom maximum number of sources to check
agent = AutonomousAgent(groq, max_sources=10)

# Or specify it for a single request
response = agent.process_request(request, max_sources=8)
```

The agent will search up to the specified number of sources, waiting at least 2 seconds between requests to avoid overwhelming the search services.

### ALSO: get_available_models()

(It does what you think it does.)

## What's New in v0.4.9

## Response Evaluation

PocketGroq now includes a method to evaluate whether a response satisfies a given request using AI:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

request = "What is the current temperature in Sheboygan?"
response1 = "58 degrees"
response2 = "As a large language model, I do not have access to current temperature data"

is_satisfactory1 = groq.evaluate_response(request, response1)
is_satisfactory2 = groq.evaluate_response(request, response2)

print(f"Response 1 is satisfactory: {is_satisfactory1}")  # Expected: True
print(f"Response 2 is satisfactory: {is_satisfactory2}")  # Expected: False
```

This method uses an AI LLM to analyze the request-response pair and determine if the response is satisfactory based on informativeness, correctness, and lack of uncertainty.

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

#### Vision Processing
- `process_image(prompt: str, image_source: str) -> str`: Analyzes an image with given prompt.
- `process_image_desktop(prompt: str, region=None) -> str`: Analyzes screen content.
- `process_image_desktop_region(prompt: str, x1: int, y1: int, x2: int, y2: int) -> str`: Analyzes specific screen region.
- `process_image_conversation(messages: List[Dict[str, Any]], model: str = None, **kwargs) -> str`: Handles multi-turn conversations about images.

#### Speech Processing
- `transcribe_audio(audio_file: str, language: Optional[str] = None, model: str = "distil-whisper-large-v3-en", **kwargs) -> str`: Transcribes audio to text.
- `translate_audio(audio_file: str, model: str = "whisper-large-v3", **kwargs) -> str`: Translates non-English audio to English text.

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
- `get_available_models() -> List[Dict[str, Any]]`: Retrieves list of available models.
- `evaluate_response(request: str, response: str) -> bool`: Evaluates response quality.

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

### AutonomousAgent Class
- `process_request(request: str, max_sources: int = None, verify: bool = False) -> str`: Processes a request autonomously.
- `_select_best_response(verified_sources: List[tuple], verify: bool) -> str`: Selects the best response from verified sources.
- `_generate_search_query(request: str) -> str`: Generates an optimized search query.
- `_evaluate_response(request: str, response: str) -> bool`: Evaluates response quality.

## License

This project is licensed under the MIT License. When using PocketGroq in your projects, please include a mention of J. Gravelle in your code and/or documentation.

![J. Gravelle](https://github.com/user-attachments/assets/73c812cd-685e-4969-9497-639ae9312d6c)

---

Thank you for using PocketGroq! We hope this tool enhances your development process and enables you to create amazing AI-powered applications with ease. If you have any questions or need further assistance, don't hesitate to reach out to the community or check the documentation. Happy coding!