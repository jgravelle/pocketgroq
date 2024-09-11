# PocketGroq

PocketGroq provides a simpler interface to interact with the Groq API, aiding in rapid development by abstracting complex API calls into simple functions.

## Installation and Upgrading

### Installing PocketGroq

#### Option 1: Install from PyPI (Recommended)

The easiest way to install PocketGroq is directly from PyPI using pip:

```bash
pip install pocketgroq
```

This will install the latest stable version of PocketGroq and its dependencies.

#### Option 2: Install from Source

If you want to use the latest development version or contribute to PocketGro, you can install it from the source:

1. Clone the repository:

```bash
git clone https://github.com/jgravelle/pocketgroq.git
cd pocketgroq
```

2. Install the package and its dependencies:

```bash
pip install -e .
```

This will install PocketGroq in editable mode, allowing you make changes to the source code and immediately see the effects.

### Upgrading PocketGroq

To upgrade an existing installation ofGroq to the latest version, use the following command:

```bash
pip install --upgrade pocketgroq
```

This will fetch and install the recent version of PocketGroq from PyPI, along with any updated dependencies.

To upgrade to a specific version, you can specify the version number:

```bash
pip install --upgrade pocketgroq==0.2.0
```

After upgrading, it's a good idea to verify the installed version:

```bash
pip show pocketgroq
```

This will display information about the installed PocketGroq package, including its version number.

## Basic Usage

### Initializing GroProvider

```python
from pocketgroq import GroqProvider

# Initialize the GroqProvider
groq = GroqProvider()
```

### Performing Basic Chat Completion

```python
response =q.generate(
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
    prompt=" the importance of fast language models",
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
    prompt="Count to . Your response must begin with \"1, \". Example: 1, 2, 3, ...",
    model="llama3-8b-8192",
    temperature=0.,
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
       ="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024        top_p=1,
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
        temperature=0.5        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
        async_mode=True
    )

    async for chunk stream:
        print(chunk, end="")

asyncio.run(main())
```

### JSON Mode

```
from typing import List, Optional
from pydantic import BaseModel
from pocketgroq import GroqProvider

class Ingredient(BaseModel):
    name: str
    quantity: str
    quantity_unit: Optional[str]

class Recipe(BaseModel):
    recipe_name: str
    ingredients: ListIngredient]
    directions: List[str]

def get_recipe(recipe_name: str) -> Recipe:
    response = groq.generate(
        prompt=f"Fetch a recipe for {recipe_name}",
        model="ll3-8b-8192",
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

PocketGroq allows to define tools (functions) that the model can use during the conversation:

```python
def reverse_string(input_string: str) ->:
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
                "required":input_string"],
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
    image_urlhttps://example.com/image.png"
)
print(response_url)

# Via passed-in image
image_path = "path_to_your_image.jpg"
base64_image = encode_image(image_path)

response_base64 = groq.generate(
    prompt=" in this image?",
    model="llava-v1.5-7b-4096-preview",
    image_url=f"data:image/jpeg;base64,{base64_image}"
)
print(response_base64```

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

4. **Image Analysis: Utilize PocketGroq's image handling capabilities for various visual tasks.

```python
image_url = "https://example.com/chart.jpg"
chart_analysis = groq.generateAnalyze this chart image and provide key insights", image_path=image_url)
print_analysis)
```

5. **Automated Customer Support**: Implement PocketGroq in a chatbot for handling customer inquiries.

```python
user_query = "How do I reset my password?"
response = groq.generate(f"Provide a step-by-step guide to answer this customer query: {user_query}")
print(response)
```

## Configuration

PocketGroq uses environment variables for configuration. `GROQ_API_KEY` in your environment or in a `.env` file in your project root.

## Error Handling

PocketGroq raises custom:

- `GroqAPIKeyMissingError`: Raised when the Groq API key is missing.
- `GroqAPIError`: Raised when there's an error with the Groq API.

Handle exceptions in your code for robust error management.

## Contributing

Feel free to open issues or submit pull requests on the [GitHub repository](https://github.com/jgravelle/pocketgroq) if you encounter any problems or have feature suggestions.

## License

This project is licensed under the MIT License. Mention J. Gravelle in your code and/or docs. He's kinda full of himself...