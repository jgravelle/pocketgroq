from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel, LongformerTokenizer
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIError

class KeyphraseEmbeddingPooler(nn.Module):
    """
    Implements the keyphrase embedding pooler from the LongKey paper.
    Uses convolutional layers for n-gram representation and max pooling
    for aggregating keyphrase occurrences.
    """
    def __init__(self, embedding_dim: int = 768, max_phrase_length: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_phrase_length = max_phrase_length
        
        # Create n-gram convolutional layers (1 through max_phrase_length)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=n,
                padding=0
            ) for n in range(1, max_phrase_length + 1)
        ])

    def forward(self, word_embeddings: torch.Tensor, phrase_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for keyphrase embedding generation.
        
        Args:
            word_embeddings: Tensor of shape (batch_size, seq_length, embedding_dim)
            phrase_masks: List of tensors identifying keyphrase occurrences
            
        Returns:
            Tensor of shape (num_keyphrases, embedding_dim)
        """
        # Transpose for conv1d operation
        word_embeddings = word_embeddings.transpose(1, 2)  # (batch_size, embedding_dim, seq_length)
        
        keyphrase_embeddings = []
        
        # Process each n-gram length
        for n, conv in enumerate(self.conv_layers, 1):
            # Get phrases of current length
            current_masks = [mask for mask in phrase_masks if mask.size(1) == n]
            if not current_masks:
                continue
                
            # Combine masks
            combined_mask = torch.cat(current_masks, dim=0)
            
            # Apply convolution
            conv_output = conv(word_embeddings)  # (batch_size, embedding_dim, seq_length - n + 1)
            
            # Extract phrase embeddings using masks
            phrase_embeds = []
            for mask in current_masks:
                # Apply mask and max pool
                masked_output = conv_output * mask.unsqueeze(1)
                phrase_embed = torch.max(masked_output, dim=2)[0]
                phrase_embeds.append(phrase_embed)
            
            keyphrase_embeddings.extend(phrase_embeds)
        
        # Combine all keyphrase embeddings
        return torch.stack(keyphrase_embeddings) if keyphrase_embeddings else torch.empty(0, self.embedding_dim)

class LongKeyExtractor:
    """
    Implements keyphrase extraction using the LongKey approach.
    Integrates with PocketGroq while implementing the core LongKey functionality.
    """
    def __init__(self, groq_provider: GroqProvider, max_phrase_length: int = 5, chunk_size: int = 8192):
        self.groq = groq_provider
        self.max_phrase_length = max_phrase_length
        self.chunk_size = chunk_size
        
        # Initialize Longformer model and tokenizer
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # Initialize keyphrase embedding pooler
        self.pooler = KeyphraseEmbeddingPooler(
            embedding_dim=self.model.config.hidden_size,
            max_phrase_length=max_phrase_length
        )
        
        # Scoring layers
        self.ranking_layer = nn.Linear(self.model.config.hidden_size, 1)
        self.chunking_layer = nn.Linear(self.model.config.hidden_size, 1)
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.pooler.to(self.device)
        self.ranking_layer.to(self.device)
        self.chunking_layer.to(self.device)

    def extract_keyphrases(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Extract keyphrases from the given text.
        
        Args:
            text: Input document text
            top_k: Number of keyphrases to return
            
        Returns:
            List of dictionaries containing keyphrases and their scores
        """
        # Split text into chunks if needed
        chunks = self._split_into_chunks(text)
        
        all_keyphrases = []
        for chunk in chunks:
            # Tokenize and encode
            inputs = self.tokenizer(
                chunk,
                return_tensors='pt',
                max_length=self.chunk_size,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get word embeddings from Longformer
            with torch.no_grad():
                outputs = self.model(**inputs)
                word_embeddings = outputs.last_hidden_state
            
            # Extract candidate keyphrases and create masks
            candidates, phrase_masks = self._extract_candidates(chunk, word_embeddings)
            
            if not candidates:
                continue
                
            # Generate keyphrase embeddings
            keyphrase_embeddings = self.pooler(word_embeddings, phrase_masks)
            
            # Score keyphrases
            ranking_scores = self.ranking_layer(keyphrase_embeddings).squeeze(-1)
            chunking_scores = torch.sigmoid(self.chunking_layer(keyphrase_embeddings)).squeeze(-1)
            
            # Combine scores
            final_scores = ranking_scores * chunking_scores
            
            # Convert to list of dictionaries
            chunk_keyphrases = [
                {
                    'keyphrase': candidates[i],
                    'score': score.item()
                }
                for i, score in enumerate(final_scores)
            ]
            
            all_keyphrases.extend(chunk_keyphrases)
        
        # Sort by score and return top_k
        all_keyphrases.sort(key=lambda x: x['score'], reverse=True)
        return all_keyphrases[:top_k]

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split long text into chunks that fit within model context window."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_tokens = len(self.tokenizer.tokenize(word))
            if current_length + word_tokens > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _extract_candidates(self, text: str, word_embeddings: torch.Tensor) -> tuple:
        """Extract candidate keyphrases and create corresponding masks."""
        # This is a simplified version - you'd want more sophisticated candidate extraction
        words = text.split()
        candidates = []
        masks = []
        
        # Generate n-gram candidates up to max_phrase_length
        for n in range(1, self.max_phrase_length + 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                candidates.append(phrase)
                
                # Create mask for this phrase occurrence
                mask = torch.zeros((1, word_embeddings.size(1)))
                mask[0, i:i+n] = 1
                masks.append(mask.to(self.device))
        
        return candidates, masks

def add_longkey_to_groq(groq_provider: GroqProvider) -> None:
    """
    Add LongKey keyphrase extraction capabilities to a GroqProvider instance.
    
    Args:
        groq_provider: The GroqProvider instance to enhance
    """
    extractor = LongKeyExtractor(groq_provider)
    
    def extract_keyphrases(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract keyphrases using the LongKey approach."""
        return extractor.extract_keyphrases(text, top_k)
    
    # Add method to GroqProvider
    setattr(GroqProvider, 'extract_keyphrases', extract_keyphrases)