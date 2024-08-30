# PocketGroq

PocketGroq provides a simpler interface to interact with the Groq API, aiding in rapid development by abstracting complex API calls into simple functions.

## Installation

1. Clone the repository or download the source code:

```bash
git clone https://github.com/jgravelle/pocketgroq.git
cd pocketgroq
```

2. Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install the following dependencies:
- groq>=0.8.0
- python-dotenv>=0.19.1
- pytest>=7.3.1 (for development)
- pytest-asyncio>=0.21.0 (for development)
- requests>=2.32.3

Note: If you're not planning to contribute to the development of PocketGroq, you can omit the pytest packages by creating a new requirements file without those lines.

## Basic Usage

### Initializing GroqProvider

```python
from pocketgroq import GroqProvider

# Initialize the GroqProvider
groq = GroqProvider()
```

### Simple Text Generation

```python
response = groq.generate("Tell me a joke about programming.")
print(response)
```

### Tool Usage Example: String Reverser

PocketGroq allows you to define tools (functions) that the model can use during the conversation:

```python
from typing import Dict

def reverse_string(input_string: str) -> Dict[str, str]:
    """ Reverse the given string """
    return {"reversed_string": input_string[::-1]}

# Define the tool
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
                        "description": "The string to be reversed, e.g., 'hello'",
                    }
                },
                "required": ["input_string"],
            },
            "implementation": reverse_string
        }
    }
]

# Generate a response using the tool
response = groq.generate("Please reverse the string 'hello world'", tools=tools)
print("Response:", response)
```

### Retrieving Available Models

You can retrieve all available models:

```python
models = groq.get_available_models()
print("Available Models:", models)
```

### Overriding the Default Model

Override the default model by passing the `model` parameter to the `generate` method:

```python
# Use a specific model (ensure it's available in your Groq account)
selected_model = 'llama3-groq-8b-8192-tool-use-preview'
response = groq.generate("Please reverse the string 'hello world'", model=selected_model, tools=tools)
print("Response with Selected Model:", response)
```

## Advanced Usage

### Streaming Responses

For long responses, you can use streaming:

```python
for chunk in groq.generate("Write a short story about AI", stream=True):
    print(chunk, end='', flush=True)
```

### Asynchronous Generation

For asynchronous operations:

```python
import asyncio

async def main():
    response = await groq.generate("Explain quantum computing", async_mode=True)
    print(response)

asyncio.run(main())
```

### JSON Mode

To get responses in JSON format:

```python
response = groq.generate("List 3 programming languages and their main uses", json_mode=True)
print(response)
```

## Configuration

PocketGroq uses environment variables for configuration. Set `GROQ_API_KEY` in your environment or in a `.env` file in your project root.

## Error Handling

PocketGroq raises custom exceptions:

- `GroqAPIKeyMissingError`: Raised when the Groq API key is missing.
- `GroqAPIError`: Raised when there's an error with the Groq API.

Handle these exceptions in your code for robust error management.

## Contributing

Feel free to open issues or submit pull requests on the [GitHub repository](https://github.com/jgravelle/pocketgroq) if you encounter any problems or have feature suggestions.

## License

This project is licensed under the MIT License.