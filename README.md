# PocketGroq

PocketGroq is a powerful and user-friendly Python library that provides seamless integration with the Groq API. It offers a simple interface to leverage Groq's advanced language models for various natural language processing tasks, including text generation, tool use, and more.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
   - [JSON Mode](#json-mode)
   - [Streaming](#streaming)
   - [Tool Use](#tool-use)
   - [Asynchronous Operations](#asynchronous-operations)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

Install PocketGroq using pip:

```bash
pip install pocketgroq
```

## Configuration

Before using PocketGroq, you need to set up your Groq API key. There are two ways to do this:

1. Environment Variable:
   Set the `GROQ_API_KEY` environment variable:

   ```bash
   export GROQ_API_KEY=your_api_key_here
   ```

2. .env File:
   Create a `.env` file in your project root and add your API key:

   ```
   GROQ_API_KEY=your_api_key_here
   ```

   PocketGroq will automatically load the API key from the .env file.

## Basic Usage

Here's a simple example of how to use PocketGroq for text generation:

```python
from pocketgroq import GroqProvider

# Initialize the GroqProvider
groq = GroqProvider()

# Generate text
response = groq.generate("Tell me a joke about programming")
print(response)
```

## Advanced Features

### JSON Mode

You can use JSON mode to ensure the model's output is in a valid JSON format:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

prompt = "Generate a JSON object with name, age, and occupation fields"
response = groq.generate(prompt, json_mode=True)
print(response)
```

### Streaming

For long responses, you can use streaming to get partial results as they're generated:

```python
from pocketgroq import GroqProvider

groq = GroqProvider()

prompt = "Write a short story about a time traveler"
for chunk in groq.generate(prompt, stream=True):
    print(chunk, end="", flush=True)
```

### Tool Use

PocketGroq supports tool use, allowing the model to call predefined functions:

```python
from pocketgroq import GroqProvider

def get_weather(location: str):
    # Simulated weather function
    return {"temperature": 20, "condition": "sunny"}

groq = GroqProvider()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
            "implementation": get_weather
        }
    }
]

response = groq.generate("What's the weather in San Francisco?", tools=tools)
print(response)
```

### Asynchronous Operations

PocketGroq supports asynchronous operations for both regular and streaming responses:

```python
import asyncio
from pocketgroq import GroqProvider

groq = GroqProvider()

async def async_generate():
    response = await groq.generate("Explain quantum computing", async_mode=True)
    print(response)

async def async_stream():
    async for chunk in groq.generate("Tell me a story about AI", async_mode=True, stream=True):
        print(chunk, end="", flush=True)

async def main():
    await async_generate()
    print("\n---\n")
    await async_stream()

asyncio.run(main())
```

## Error Handling

PocketGroq provides custom exceptions for better error handling:

```python
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIKeyMissingError, GroqAPIError

try:
    groq = GroqProvider()
    response = groq.generate("Tell me a joke")
    print(response)
except GroqAPIKeyMissingError:
    print("Please set your GROQ_API_KEY environment variable or in the .env file")
except GroqAPIError as e:
    print(f"An error occurred while calling the Groq API: {str(e)}")
```

## Best Practices

1. Always handle exceptions to gracefully manage API errors and missing keys.
2. Use streaming for long-form content generation to improve user experience.
3. Leverage tool use for complex tasks that require external data or computations.
4. Use JSON mode when you need structured output from the model.
5. For high-throughput applications, consider using asynchronous operations.

## Contributing

We welcome contributions to PocketGroq! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get started.

## License

PocketGroq is released under the MIT License. See the [LICENSE](LICENSE) file for more details.