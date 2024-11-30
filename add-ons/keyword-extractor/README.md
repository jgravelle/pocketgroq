# PocketGroq LongKey Add-on

A powerful keyphrase extraction add-on for PocketGroq based on the LongKey methodology, designed specifically for analyzing long documents with advanced contextual understanding.  Inspired by this whitepaper:  https://arxiv.org/pdf/2411.17863

## Features

- **Long Document Support**: Process documents up to 96K tokens using efficient chunking mechanisms
- **Advanced Embedding Strategy**: Utilizes convolutional neural networks and max pooling for robust keyphrase representation
- **Context-Aware**: Captures document-wide context through the Longformer architecture
- **Flexible Integration**: Seamlessly extends PocketGroq's capabilities

## Installation

```bash
pip install pocketgroq-longkey
```

Make sure you have PocketGroq already installed:
```bash
pip install pocketgroq
```

## Requirements

- Python >= 3.7
- PocketGroq >= 0.5.5
- PyTorch >= 1.7.0
- Transformers >= 4.0.0

## Quick Start

```python
from pocketgroq import GroqProvider
from pocketgroq_longkey import add_longkey_to_groq

# Initialize PocketGroq
groq = GroqProvider(api_key="your-groq-api-key")

# Add LongKey functionality
add_longkey_to_groq(groq)

# Extract keyphrases from text
text = """
Your long document text here. The LongKey add-on is specifically designed
to handle long documents effectively, capturing context across the entire text.
"""

keyphrases = groq.extract_keyphrases(text, top_k=5)

# Print extracted keyphrases and their scores
for kp in keyphrases:
    print(f"Keyphrase: {kp['keyphrase']}, Score: {kp['score']:.3f}")
```

## Advanced Usage

### Customizing Extraction Parameters

You can customize various parameters when initializing the LongKey extractor:

```python
from pocketgroq_longkey import LongKeyExtractor

extractor = LongKeyExtractor(
    groq_provider=groq,
    max_phrase_length=5,  # Maximum words in a keyphrase
    chunk_size=8192      # Tokens per chunk for processing
)

# Add custom extractor to GroqProvider
def extract_keyphrases(self, text: str, top_k: int = 5):
    return extractor.extract_keyphrases(text, top_k)

setattr(GroqProvider, 'extract_keyphrases', extract_keyphrases)
```

### Processing Multiple Documents

```python
documents = [
    "First long document text...",
    "Second long document text...",
    "Third long document text..."
]

all_keyphrases = []
for doc in documents:
    keyphrases = groq.extract_keyphrases(doc, top_k=5)
    all_keyphrases.append(keyphrases)
```

## Architecture Overview

The LongKey add-on implements the architecture described in the LongKey paper with several key components:

1. **Document Encoding**:
   - Uses Longformer for processing long documents
   - Implements efficient chunking for documents exceeding the model's context window
   - Preserves global context through special attention mechanisms

2. **Keyphrase Embedding**:
   - Employs convolutional layers for n-gram representation
   - Uses max pooling to aggregate keyphrase occurrences
   - Captures contextual information across the document

3. **Scoring Mechanism**:
   - Combines ranking and chunking scores
   - Implements margin ranking loss for optimization
   - Uses binary cross-entropy for chunking classification

## Performance Considerations

- Processing speed depends on document length and available GPU resources
- Documents are automatically chunked if they exceed 8192 tokens
- GPU acceleration is automatically used if available

## Error Handling

The add-on includes robust error handling:

```python
from pocketgroq.exceptions import GroqAPIError

try:
    keyphrases = groq.extract_keyphrases(text)
except GroqAPIError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Document Preparation**:
   - Clean your text input to remove unnecessary formatting
   - Consider breaking very long documents into logical sections
   - Remove redundant or boilerplate text

2. **Parameter Tuning**:
   - Adjust `max_phrase_length` based on your domain's typical keyphrase length
   - Modify `top_k` based on document length and application needs
   - Consider document length when setting `chunk_size`

3. **Resource Management**:
   - Process documents in batches for better efficiency
   - Monitor memory usage when processing very large documents
   - Use GPU acceleration when available

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this add-on in your research, please cite:

```bibtex
@article{alves2024longkey,
  title={LongKey: Keyphrase Extraction for Long Documents},
  author={Alves, Jeovane Honorio and State, Radu and Freitas, Cinthia Obladen de Almendra and Barddal, Jean Paul},
  journal={arXiv preprint arXiv:2411.17863},
  year={2024}
}
```

## Acknowledgments

- Based on the LongKey paper by Alves et al.
- Built on top of the PocketGroq framework
- Uses the Longformer architecture for long document processing