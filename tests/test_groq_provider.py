import pytest
from unittest.mock import patch, MagicMock
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIKeyMissingError, GroqAPIError

@pytest.fixture
def mock_groq_client():
    with patch('pocketgroq.groq_provider.Groq') as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_async_groq_client():
    with patch('pocketgroq.groq_provider.AsyncGroq') as mock_async_groq:
        mock_client = MagicMock()
        mock_async_groq.return_value = mock_client
        yield mock_client

def test_groq_provider_initialization(mock_groq_client):
    with patch('pocketgroq.groq_provider.get_api_key', return_value='test_api_key'):
        provider = GroqProvider()
        assert provider.api_key == 'test_api_key'
        mock_groq_client.assert_called_once_with(api_key='test_api_key')

def test_groq_provider_initialization_no_api_key(mock_groq_client):
    with patch('pocketgroq.groq_provider.get_api_key', return_value=None):
        with pytest.raises(GroqAPIKeyMissingError):
            GroqProvider()

def test_generate_text(mock_groq_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Generated text"
    mock_groq_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key='test_api_key')
    result = provider.generate("Test prompt")

    assert result == "Generated text"
    mock_groq_client.chat.completions.create.assert_called_once()

def test_generate_text_with_stream(mock_groq_client):
    mock_stream = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk1"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="chunk2"))]),
    ]
    mock_groq_client.chat.completions.create.return_value = mock_stream

    provider = GroqProvider(api_key='test_api_key')
    result = list(provider.generate("Test prompt", stream=True))

    assert result == ["chunk1", "chunk2"]
    mock_groq_client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_async(mock_async_groq_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Generated text"
    mock_async_groq_client.chat.completions.create.return_value = mock_completion

    provider = GroqProvider(api_key='test_api_key')
    result = await provider.generate("Test prompt", async_mode=True)

    assert result == "Generated text"
    mock_async_groq_client.chat.completions.create.assert_called_once()

def test_generate_text_with_tool(mock_groq_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Tool result"
    mock_groq_client.chat.completions.create.return_value = mock_completion

    def mock_tool(arg):
        return f"Processed {arg}"

    provider = GroqProvider(api_key='test_api_key')
    result = provider.generate("Use tool", tools=[
        {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "A mock tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {
                            "type": "string",
                            "description": "An argument",
                        }
                    },
                    "required": ["arg"],
                },
                "implementation": mock_tool
            }
        }
    ])

    assert result == "Tool result"
    mock_groq_client.chat.completions.create.assert_called_once()

def test_api_error(mock_groq_client):
    mock_groq_client.chat.completions.create.side_effect = Exception("API Error")

    provider = GroqProvider(api_key='test_api_key')
    with pytest.raises(GroqAPIError):
        provider.generate("Test prompt")