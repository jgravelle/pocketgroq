# PocketGroq v0.4.5: Enhanced Web Capabilities and Flexible Ollama Integration
![PocketGroq Logo](https://github.com/user-attachments/assets/d06b6aaf-400e-40db-bdaf-626aaa1040ef)

## What's New in v0.4.5

PocketGroq v0.4.5 brings significant enhancements to web-related functionalities and improves the flexibility of Ollama integration:

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

PocketGroq v0.4.5 introduces more flexible integration with Ollama:

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

PocketGroq v0.4.5 introduces a new exception for Ollama-related errors:

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

The test suite has been expanded to cover the new web capabilities and Ollama integration:

```python
# In test.py

def test_web_search():
    print("\nTesting Web Search...")
    query = "What is PocketGroq?"
    results = groq.web_search(query)
    print(f"Search query: {query}")
    print(f"Number of results: {len(results)}")
    assert isinstance(results, list) and len(results) > 0
    print("First result:", results[0])

def test_crawl_website():
    print("\nTesting Website Crawling...")
    url = "https://example.com"
    results = groq.crawl_website(url, formats=["markdown", "html"], max_depth=2, max_pages=5)
    print(f"Crawl results for {url}:")
    print(f"Number of pages crawled: {len(results)}")
    assert isinstance(results, list) and len(results) > 0
    print("First page title:", results[0]['metadata']['title'])

def test_scrape_url():
    print("\nTesting URL Scraping...")
    url = "https://example.com"
    result = groq.scrape_url(url, formats=["markdown", "html", "structured_data"])
    print(f"Scrape result for {url}:")
    assert isinstance(result, dict) and 'markdown' in result and 'html' in result
    print("Markdown content length:", len(result['markdown']))
    print("HTML content length:", len(result['html']))
    if 'structured_data' in result:
        print("Structured data:", json.dumps(result['structured_data'], indent=



## Version 0.4.3 Update

In this latest version (0.4.3), we've made significant improvements to how PocketGroq interacts with the Ollama server:

- **Flexible Ollama Dependency**: PocketGroq no longer requires Ollama to be running all the time. This allows for more versatile use of the library's core features.

- **Persistent Features**: Users should be aware that Ollama will need to be running to take advantage of various persistence features, including RAG (Retrieval-Augmented Generation) functionality.

- **Graceful Handling**: When Ollama-dependent features are accessed without the server running, PocketGroq will now provide clear error messages, allowing users to start the server and retry their request.


### Conversational Persistence

PocketGroq supports maintaining conversation history across multiple interactions, allowing for more coherent and context-aware dialogues.

#### Simple Usage:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

# Start a new conversation
session_id = "user_123"
groq.start_conversation(session_id)

# First interaction
response1 = groq.generate("What is the capital of France?", session_id=session_id)
print("Response 1:", response1)

# Second interaction (context-aware)
response2 = groq.generate("What is its population?", session_id=session_id)
print("Response 2:", response2)

# End the conversation when done
groq.end_conversation(session_id)
```

#### Advanced Usage:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

def chat_session(topic):
    session_id = f"chat_{topic}"
    groq.start_conversation(session_id)
    
    print(f"Starting chat session about {topic}")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = groq.generate(user_input, session_id=session_id)
        print("AI:", response)
    
    groq.end_conversation(session_id)
    print("Chat session ended")

# Example usage
chat_session("space exploration")
```

### Enhanced RAG with Persistence

PocketGroq's RAG capabilities now support persistent storage of document embeddings, allowing for faster querying and reduced processing time for repeated operations.

#### Simple Usage:

```python
from pocketgroq import GroqProvider

groq = GroqProvider(rag_persistent=True, rag_index_path="my_persistent_index.pkl")

# Initialize RAG
groq.initialize_rag()

# Load documents (will be stored persistently)
groq.load_documents("https://en.wikipedia.org/wiki/Artificial_intelligence")
groq.load_documents("path/to/local/document.txt")

# Query documents
query = "What are the main applications of AI?"
response = groq.query_documents(query)
print(response)
```

#### Advanced Usage: Research Assistant

```python
from pocketgroq import GroqProvider

class ResearchAssistant:
    def __init__(self):
        self.groq = GroqProvider(rag_persistent=True, rag_index_path="research_index.pkl")
        self.groq.initialize_rag()

    def load_source(self, source):
        print(f"Loading source: {source}")
        self.groq.load_documents(source)

    def research(self, topic):
        print(f"Researching: {topic}")
        query = f"Provide a comprehensive summary of {topic} based on the loaded documents."
        response = self.groq.query_documents(query)
        return response

    def generate_report(self, topic, research_summary):
        prompt = f"Based on this research summary about {topic}, generate a detailed report:\n\n{research_summary}"
        report = self.groq.generate(prompt, max_tokens=2000)
        return report

# Example usage
assistant = ResearchAssistant()

# Load multiple sources
assistant.load_source("https://en.wikipedia.org/wiki/Climate_change")
assistant.load_source("https://www.ipcc.ch/reports/")
assistant.load_source("path/to/local/climate_study.pdf")

# Perform research
research_summary = assistant.research("Impact of climate change on biodiversity")
print("\nResearch Summary:")
print(research_summary)

# Generate a report
report = assistant.generate_report("Impact of climate change on biodiversity", research_summary)
print("\nGenerated Report:")
print(report)
```

## Benefits of New Persistence Features

1. **Improved Context Awareness**: Conversation persistence allows for more natural, context-aware dialogues across multiple interactions.
2. **Efficient Information Retrieval**: Persistent RAG reduces processing time for repeated queries on the same document set.
3. **Enhanced User Experience**: Maintain coherent conversations and provide more accurate, contextually relevant responses.
4. **Scalability**: Handle complex, multi-turn conversations and large document sets more effectively.
5. **Flexibility**: Choose between persistent and non-persistent modes based on your application's needs.

Upgrade to PocketGroq v0.4.3 today to leverage these powerful new persistence capabilities in your projects!

## What's New in v0.4.3

- **Conversation Persistence**: Maintain context across multiple interactions for more coherent dialogues.
- **Enhanced Retrieval-Augmented Generation (RAG)**: Improve response accuracy with persistent document context.
- **Improved Document Handling**: Load and query both local and web-based documents with persistent storage options.
- **Context-Aware Generation**: Generate responses that leverage both conversation history and loaded document context.
- **Ollama Integration**: Utilizes Ollama for efficient, locally-generated embeddings in RAG functionality.

For full details on the new persistence features and RAG enhancements, see the [Persistence Features](#persistence-features) and [RAG Features](#rag-features) sections below.


# PocketGroq v0.4.3: Now with Retrieval-Augmented Generation (RAG)!

PocketGroq has been upgraded to version 0.4.3, introducing powerful Retrieval-Augmented Generation (RAG) capabilities. This major update enhances PocketGroq's ability to provide context-aware responses by leveraging external knowledge sources.

## New RAG Features

### Initializing RAG

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

groq.initialize_rag()  # Initialize RAG with default settings
```

### Loading Documents

```python
# Load a local document

groq.load_documents("path/to/local/document.txt")

# Load content from a web page

groq.load_documents("https://example.com/article")
```

### Querying Documents

```python
query = "What are the main benefits of renewable energy?"
response = groq.query_documents(query)
print(response)
```

### RAG-Enhanced Generation

```python
prompt = "Explain the impact of renewable energy on climate change, using the loaded documents as context."
response = groq.generate(prompt, use_rag=True)
print(response)
```

## RAG Use Case: Research Assistant

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

groq.initialize_rag()

# Load multiple sources

groq.load_documents("https://en.wikipedia.org/wiki/Renewable_energy")

groq.load_documents("https://www.energy.gov/eere/renewables")

groq.load_documents("path/to/local/energy_report.pdf")

# Perform a complex query
query = "Summarize the current state of renewable energy adoption globally, highlighting key challenges and opportunities."
response = groq.query_documents(query)

print("Research Summary:")
print(response)

# Generate a report based on the research
report_prompt = f"Using the following research summary as a basis, write a detailed report on the global state of renewable energy:\n\n{response}"
report = groq.generate(report_prompt, use_rag=True, max_tokens=2000)

print("\nGenerated Report:")
print(report)
```

This RAG functionality allows PocketGroq to:
- Integrate external knowledge into its responses
- Provide more accurate and context-aware information
- Support complex research and analysis tasks

Upgrade to PocketGroq v0.4.3 today to leverage these powerful new RAG capabilities in your projects!

## Other New Features

### Chain of Thought (CoT) Reasoning

PocketGroq now supports Chain of Thought reasoning, allowing for more complex problem-solving and step-by-step analysis. The new CoT features include:

- `solve_problem_with_cot(problem: str)`: Solves a complex problem using Chain of Thought reasoning.
- `generate_cot(problem: str)`: Generates intermediate reasoning steps for a given problem.
- `synthesize_cot(cot_steps: List[str])`: Synthesizes a final answer from Chain of Thought steps.

### Enhanced Test Suite

The test suite has been expanded and now includes a menu-driven interface for selective test execution. New tests have been added for the Chain of Thought functionality.

## Installation and Upgrading

### Installing PocketGroq

#### Option 1: Install from PyPI (Recommended)

The easiest way to install PocketGroq is directly from PyPI using pip:

```bash
pip install pocketgroq
```

This will install the latest stable version of PocketGroq and its dependencies.

#### Option 2: Install from Source

If you want to use the latest development version or contribute to PocketGroq, you can install it from the source:

1. Clone the repository:

```bash
git clone https://github.com/jgravelle/pocketgroq.git
cd pocketgroq
```

2. Install the package and its dependencies:

```bash
pip install -e .
```

This will install PocketGroq in editable mode, allowing you to make changes to the source code and immediately see the effects.

### Upgrading PocketGroq

To upgrade an existing installation of PocketGroq to the latest version, use the following command:

```bash
pip install --upgrade pocketgroq
```

This will fetch and install the most recent version of PocketGroq from PyPI, along with any updated dependencies.

To upgrade to a specific version, you can specify the version number:

```bash
pip install --upgrade pocketgroq==0.4.5
```

After upgrading, it's a good idea to verify the installed version:

```bash
pip show pocketgroq
```

This will display information about the installed PocketGroq package, including its version number.

## Basic Usage

### Initializing GroqProvider and WebTool

```python
from pocketgroq import GroqProvider
from pocketgroq.web_tool import WebTool

# Initialize the GroqProvider

groq = GroqProvider()

# Initialize the WebTool
web_tool = WebTool(num_results=5, max_tokens=4096)
```

### Performing Web Searches

```python
query = "Latest developments in AI"
search_results = web_tool.search(query)

for result in search_results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Description: {result['description']}")
    print("---")
```

### Retrieving Web Content

```python
url = "https://example.com/article"
content = web_tool.get_web_content(url)
print(content[:500])  # Print first 500 characters
```

### Combining Web Search with Language Model

```python
query = "Explain the latest breakthroughs in quantum computing"
search_results = web_tool.search(query)

# Prepare context from search results
context = "\n".join([f"{r['title']}: {r['description']}" for r in search_results])

# Generate response using the context
prompt = f"Based on the following information:\n\n{context}\n\nProvide a concise summary of the latest breakthroughs in quantum computing."
response = groq.generate(prompt, max_tokens=4096, model="llama3-70b-8192", temperature=0.0)
print(response)
```

### Performing Basic Chat Completion

```python
response = groq.generate(
    prompt="Explain the importance of fast language models",
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    stop=None,
    stream=False
)
print(response)
```

### Streaming a Chat Completion

```python
stream = groq.generate(
    prompt="Explain the importance of fast language models",
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    stop=None,
    stream=True
)

for chunk in stream:
    print(chunk, end="")
```

### Overriding the Default Model

```python
selected_model = 'llama3-groq-8b-8192-tool-use-preview'
response = groq.generate("Explain quantum computing", model=selected_model)
print("Response with Selected Model:", response)
```

### Performing a Chat Completion with a Stop Sequence

```python
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
```

### Asynchronous Generation

```python
import asyncio

async def main():
    response = await groq.generate(
        prompt="Explain the theory of relativity",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        async_mode=True
    )
    print(response)

asyncio.run(main())
```

### Streaming an Async Chat Completion

```python
import asyncio

async def main():
    stream = await groq.generate(
        prompt="Explain the importance of fast language models",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
        async_mode=True
    )

    async for chunk in stream:
        print(chunk, end="")

asyncio.run(main())
```

### JSON Mode

```python
from typing import List, Optional
from pydantic import BaseModel
from pocketgroq import GroqProvider

class Ingredient(BaseModel):
    name: str
    quantity: str
    quantity_unit: Optional[str]

class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[Ingredient]
    directions: List[str]


def get_recipe(recipe_name: str) -> Recipe:
    response = groq.generate(
        prompt=f"Fetch a recipe for {recipe_name}",
        model="llama3-8b-8192",
        temperature=0,
        stream=False,
        json_mode=True
    )
    return Recipe.parse_raw(response)


def print_recipe(recipe: Recipe):
    print("Recipe:", recipe.recipe_name)
    print("\nIngredients:")
    for ingredient in recipe.ingredients:
        print(f"- {ingredient.name}: {ingredient.quantity} {ingredient.quantity_unit or ''}")
    print("\nDirections:")
    for step, direction in enumerate(recipe.directions, start=1):
        print(f"{step}. {direction}")

recipe = get_recipe("apple pie")
print_recipe(recipe)
```

### Tool Usage

PocketGroq allows you to define tools (functions) that the model can use during the conversation:

```python

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
```

### Vision (llava-v1.5-7b-4096-preview model only)

```python
from pocketgroq import GroqProvider
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


groq = GroqProvider()

# Via URL
response_url = groq.generate(
    prompt="What's in this image?",
    model="llava-v1.5-7b-4096-preview",
    image_url="https://example.com/image.png"
)
print(response_url)

# Via passed-in image
image_path = "path_to_your_image.jpg"
base64_image = encode_image(image_path)

response_base64 = groq.generate(
    prompt="What's in this image?",
    model="llava-v1.5-7b-4096-preview",
    image_url=f"data:image/jpeg;base64,{base64_image}"
)
print(response_base64)
```

## Chain of Thought Usage

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

# Solve a complex problem using Chain of Thought
problem = """
A farmer has a rectangular field that is 100 meters long and 50 meters wide. 
He wants to increase the area of the field by 20% by increasing both the length and the width by the same percentage. 
What should be the new length and width of the field? 
Round your answer to the nearest centimeter.
"""
answer = groq.solve_problem_with_cot(problem)
print("Answer:", answer)

# Generate Chain of Thought steps
problem = "What is the sum of the first 10 prime numbers?"
cot_steps = groq.generate_cot(problem)
print("Chain of Thought Steps:")
for i, step in enumerate(cot_steps, 1):
    print(f"{i}. {step}")

# Synthesize an answer from Chain of Thought steps
cot_steps = [
    "The first 10 prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
    "To find the sum, we add these numbers: 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29",
    "Calculating the sum: 129"
]
final_answer = groq.synthesize_cot(cot_steps)
print("Synthesized Answer:", final_answer)
```

## WebTool Functionality

The WebTool provides two main functions:

1. `search(query: str, num_results: int = 10) -> List[Dict[str, Any]]`: Performs a web search and returns a list of search results.
2. `get_web_content(url: str) -> str`: Retrieves the content of a web page.

### Example: Advanced Web Search and Content Analysis

```python
from pocketgroq import GroqProvider
from pocketgroq.web_tool import WebTool

groq = GroqProvider()
web_tool = WebTool()

# Perform a web search
query = "Recent advancements in renewable energy"
search_results = web_tool.search(query, num_results=3)

# Analyze each search result
for result in search_results:
    print(f"Analyzing: {result['title']}")
    content = web_tool.get_web_content(result['url'])
    
    analysis_prompt = f"Analyze the following content about renewable energy and provide key insights:\n\n{content[:4000]}"
    analysis = groq.generate(analysis_prompt, max_tokens=1000)
    
    print(f"Analysis: {analysis}")
    print("---")
```

This example demonstrates how to use the WebTool to perform a search, retrieve content from each search result, and then use the GroqProvider to analyze the content.

## Use Case Scenarios

1. **Content Generation**: Use PocketGroq for automated blog post writing, social media content creation, or product descriptions.

```python
blog_topic = "The Future of Artificial Intelligence"
blog_post = groq.generate(f"Write a 500-word blog post about {blog_topic}")
print(blog_post)
```

2. **Code Assistant**: Leverage PocketGroq for code explanation, debugging, or generation.

```python
code_snippet = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
explanation = groq.generate(f"Explain this Python code and suggest any improvements:\n\n{code_snippet}")
print(explanation)
```

3. **Data Analysis**: Use PocketGroq to interpret data or generate data analysis reports.

```python
data = {
    "sales": [100, 150, 200, 180, 220],
    "expenses": [80, 90, 110, 100, 130]
}
analysis = groq.generate(f"Analyze this sales and expenses data and provide insights:\n\n{data}", json_mode=True)
print(analysis)
```

4. **Image Analysis**: Utilize PocketGroq's image handling capabilities for various visual tasks.

```python
image_url = "https://example.com/chart.jpg"
chart_analysis = groq.generate("Analyze this chart image and provide key insights", image_path=image_url)
print(chart_analysis)
```

5. **Automated Customer Support**: Implement PocketGroq in a chatbot for handling customer inquiries.

```python
user_query = "How do I reset my password?"
response = groq.generate(f"Provide a step-by-step guide to answer this customer query: {user_query}")
print(response)
```

6. **Web Research Assistant**: Utilize PocketGroq's WebTool for automated web research and summarization.

```python
from pocketgroq import GroqProvider
from pocketgroq.web_tool import WebTool

groq = GroqProvider()
web_tool = WebTool()

research_topic = "Impact of artificial intelligence on job markets"
search_results = web_tool.search(research_topic, num_results=5)

research_summary = groq.generate(
    f"Based on the following search results about '{research_topic}', provide a comprehensive summary:\n\n" +
    "\n".join([f"- {r['title']}: {r['description']}" for r in search_results])
)

print(research_summary)
```

7. **Complex Problem Solving**: Utilize PocketGroq's Chain of Thought capabilities for solving intricate problems.

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

complex_problem = """In a small town, 60% of the adults work in manufacturing, 25% work in services, and the rest are unemployed. 
If the town has 10,000 adults and the local government wants to reduce unemployment by half by creating new 
manufacturing jobs, how many new jobs need to be created?
"""

solution = groq.solve_problem_with_cot(complex_problem)
print(solution)
```

## Testing

PocketGroq now includes a comprehensive test suite with a menu-driven interface for selective test execution. To run the tests:

1. Navigate to the PocketGroq directory.
2. Run the test script:

```bash
python test.py
```

3. You will see a menu with options to run individual tests or all tests at once:

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
13. Run All Tests
0. Exit
```

4. Select the desired option by entering the corresponding number.

The test suite covers:

- Basic chat completion
- Streaming chat completion
- Model override
- Chat completion with stop sequence
- Asynchronous generation
- Streaming asynchronous chat completion
- JSON mode
- Tool usage
- Vision capabilities
- Chain of Thought problem solving
- Chain of Thought step generation
- Chain of Thought synthesis

Each test demonstrates a specific feature of PocketGroq and checks if the output meets the expected criteria. Running these tests helps ensure that all functionalities are working correctly after updates or modifications to the codebase.

## Ollama Server Requirement for RAG

The new Retrieval-Augmented Generation (RAG) feature in PocketGroq v0.4.3 requires an Ollama server for generating embeddings. Ollama is an open-source, locally-run language model server that provides fast and efficient embeddings for RAG functionality.

### Setting up Ollama

1. **Install Ollama**:
   Visit the official Ollama website at [https://ollama.ai/](https://ollama.ai/) and follow the installation instructions for your operating system.

2. **Start the Ollama server**:
   After installation, start the Ollama server by running the following command in your terminal:
   ```
   ollama serve
   ```

3. **Pull the required model**:
   PocketGroq uses the 'nomic-embed-text' model for embeddings. Pull this model by running:
   ```
   ollama pull nomic-embed-text
   ```

### Configuring PocketGroq for Ollama

By default, PocketGroq will attempt to connect to the Ollama server at `http://localhost:11434`. If your Ollama server is running on a different address or port, you can specify this when initializing RAG:

```python

groq = GroqProvider()

groq.initialize_rag(ollama_base_url="http://your-ollama-server:port")
```

### Troubleshooting

- Ensure the Ollama server is running before using RAG features in PocketGroq.
- If you encounter connection errors, check that the Ollama server is accessible at the expected address and port.
- For Windows users using WSL (Windows Subsystem for Linux), you may need to adjust the base URL to point to your WSL IP address instead of localhost.

For more detailed information about Ollama, including advanced configuration and available models, please refer to the [Ollama documentation](https://github.com/ollama/ollama).

## Configuration

PocketGroq uses environment variables for configuration. Set `GROQ_API_KEY` in your environment or in a `.env` file in your project root. This API key is essential for authenticating with the Groq API.

Example of setting the API key in a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

Make sure to keep your API key confidential and never commit it to version control.

## Error Handling

PocketGroq raises custom exceptions to help you handle errors more effectively:

- `GroqAPIKeyMissingError`: Raised when the Groq API key is missing.
- `GroqAPIError`: Raised when there's an error with the Groq API.

Handle these exceptions in your code for robust error management. For example:

```python
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError

try:
    groq = GroqProvider()
    response = groq.generate("Hello, world!")
except GroqAPIKeyMissingError:
    print("Please set your GROQ_API_KEY environment variable.")
except GroqAPIError as e:
    print(f"An error occurred while calling the Groq API: {e}")
```

## Contributing

Contributions to PocketGroq are welcome! If you encounter any problems, have feature suggestions, or want to improve the codebase, feel free to:

1. Open issues on the [GitHub repository](https://github.com/jgravelle/pocketgroq).
2. Submit pull requests with bug fixes or new features.
3. Improve documentation or add examples.

When contributing, please:

- Follow the existing code style and conventions.
- Write clear commit messages.
- Add or update tests for new features or bug fixes.
- Update the README if you're adding new functionality.

## License

This project is licensed under the MIT License. When using PocketGroq in your projects, please include a mention of J. Gravelle in your code and/or documentation. He's kinda full of himself, but he'd appreciate the acknowledgment.

![J. Gravelle](https://github.com/user-attachments/assets/73c812cd-685e-4969-9497-639ae9312d6c)

---

Thank you for using PocketGroq! We hope this tool enhances your development process and enables you to create amazing AI-powered applications with ease. If you have any questions or need further assistance, don't hesitate to reach out to the community or check the documentation. Happy coding!
