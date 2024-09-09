# PocketGroq

PocketGroq provides a simpler interface to interact with the Groq API, aiding in rapid development by abstracting complex API calls into simple functions.

## Installation

### Option 1: Install from PyPI (Recommended)

The easiest way to install PocketGroq is directly from PyPI using pip:

```bash
pip install pocketgroq
```

This will install the latest stable version of PocketGroq and its dependencies.

### Option 2: Install from Source

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

### Retrieving Available Models

```python
models = groq.get_available_models()
print("Available Models:", models)
```

### Overriding the Default Model

```python
selected_model = 'llama3-groq-8b-8192-tool-use-preview'
response = groq.generate("Explain quantum computing", model=selected_model)
print("Response with Selected Model:", response)
```

## Advanced Features

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
    response = await groq.generate("Explain the theory of relativity", async_mode=True)
    print(response)

asyncio.run(main())
```

### JSON Mode

To get responses in JSON format:

```python
response = groq.generate("List 3 programming languages and their main uses", json_mode=True)
print(response)
```

### Image Handling

PocketGroq supports image analysis using compatible models:

```python
image_url = "https://example.com/image.jpg"
response = groq.generate("Describe this image in detail", image_path=image_url)
print(response)

# For local images
local_image_path = "path/to/local/image.jpg"
response = groq.generate("What objects do you see in this image?", image_path=local_image_path)
print(response)
```

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

This project is licensed under the MIT License.  Mention J. Gravelle in your code and/or docs.  He's kinda full of himself...