import asyncio
import json
import logging
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the GroqProvider
groq = GroqProvider()

def test_basic_chat_completion():
    print("Testing Basic Chat Completion...")
    response = groq.generate(
        prompt="Explain the importance of fast language models in one sentence.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    print(response)
    assert isinstance(response, str) and len(response) > 0

def test_streaming_chat_completion():
    print("\nTesting Streaming Chat Completion...")
    stream = groq.generate(
        prompt="Count from 1 to 5.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True
    )
    full_response = ""
    for chunk in stream:
        if chunk is not None:
            print(chunk, end="")
            full_response += chunk
        else:
            print("<None>", end="")
    print()
    assert isinstance(full_response, str) and len(full_response) > 0
    print(f"Full response: '{full_response}'")

def test_override_default_model():
    print("\nTesting Override Default Model...")
    selected_model = 'llama3-groq-8b-8192-tool-use-preview'
    response = groq.generate("Explain quantum computing in one sentence.", model=selected_model)
    print("Response with Selected Model:", response)
    assert isinstance(response, str) and len(response) > 0

def test_chat_completion_with_stop_sequence():
    print("\nTesting Chat Completion with Stop Sequence...")
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
    assert isinstance(response, str) and "5" in response and "6" not in response

async def test_async_generation():
    print("\nTesting Asynchronous Generation...")
    response = await groq.generate(
        prompt="Explain the theory of relativity in one sentence.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        async_mode=True
    )
    print(response)
    assert isinstance(response, str) and len(response) > 0

async def test_streaming_async_chat_completion():
    print("\nTesting Streaming Async Chat Completion...")
    stream = await groq.generate(
        prompt="Count from 1 to 5.",
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
        async_mode=True
    )
    full_response = ""
    async for chunk in stream:
        if chunk is not None:
            print(chunk, end="")
            full_response += chunk
        else:
            print("<None>", end="")
    print()
    assert isinstance(full_response, str) and len(full_response) > 0
    print(f"Full response: '{full_response}'")

def test_json_mode():
    print("\nTesting JSON Mode...")
    class Ingredient(BaseModel):
        name: str
        quantity: Union[int, str]
        quantity_unit: Optional[str]

        @validator('quantity', pre=True)
        def quantity_to_string(cls, v):
            return str(v)

    class Recipe(BaseModel):
        recipe_name: str = Field(..., alias="name")
        ingredients: List[Ingredient]
        directions: List[str] = Field(..., alias="instructions")

    def get_recipe(recipe_name: str) -> Recipe:
        response = groq.generate(
            prompt=f"Create a recipe for {recipe_name} and return it in JSON format. Include 'name', 'ingredients' (each with 'name', 'quantity', and 'quantity_unit'), and 'instructions' fields.",
            model="llama3-8b-8192",
            temperature=0,
            stream=False,
            json_mode=True
        )
        logger.debug(f"Raw JSON response: {response}")
        
        try:
            # Parse the JSON response
            json_data = json.loads(response)
            
            # Check if the recipe is nested under a 'recipe' key
            if 'recipe' in json_data:
                json_data = json_data['recipe']
            
            # Attempt to create a Recipe object
            recipe = Recipe.model_validate(json_data)
            return recipe
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise
        except ValueError as e:
            logger.error(f"Failed to validate Recipe model: {e}")
            raise

    def print_recipe(recipe: Recipe):
        print("Recipe:", recipe.recipe_name)
        print("\nIngredients:")
        for ingredient in recipe.ingredients:
            print(f"- {ingredient.name}: {ingredient.quantity} {ingredient.quantity_unit or ''}")
        print("\nDirections:")
        for step, direction in enumerate(recipe.directions, start=1):
            print(f"{step}. {direction}")

    try:
        recipe = get_recipe("simple pancakes")
        print_recipe(recipe)
        assert isinstance(recipe, Recipe)
        assert len(recipe.ingredients) > 0
        assert len(recipe.directions) > 0
    except Exception as e:
        logger.error(f"Error in JSON mode test: {e}")
        raise

def test_tool_usage():
    print("\nTesting Tool Usage...")
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
    assert "dlrow olleh" in response.lower()

def test_vision():
    print("\nTesting Vision...")
    # Note: This test requires a valid image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    response_url = groq.generate(
        prompt="Describe this image in one sentence.",
        model="llava-v1.5-7b-4096-preview",
        image_url=image_url
    )
    print(response_url)
    assert isinstance(response_url, str) and len(response_url) > 0

def main():
    try:
        test_basic_chat_completion()
        test_streaming_chat_completion()
        test_override_default_model()
        test_chat_completion_with_stop_sequence()
        asyncio.run(test_async_generation())
        asyncio.run(test_streaming_async_chat_completion())
        test_json_mode()
        test_tool_usage()
        test_vision()
        print("\nAll tests completed successfully!")
    except GroqAPIKeyMissingError as e:
        print(f"Error: {e}")
    except GroqAPIError as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()