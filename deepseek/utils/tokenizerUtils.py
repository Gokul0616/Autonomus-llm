# tokenizer_utils.py
import json
from typing import Optional
from BPETokenizer import CustomBPETokenizer  # Replace with actual import path

def load_tokenizer(
    path: str = "tokenizer.json",
    vocab_size: int = 10000,
    special_tokens: Optional[list] = None
) -> CustomBPETokenizer:
    """
    Production-grade tokenizer loader with error handling and validation
    
    Args:
        path: Path to saved tokenizer JSON
        vocab_size: Expected vocabulary size
        special_tokens: List of special tokens used during training
    
    Returns:
        Initialized CustomBPETokenizer instance
    """
    # Validate inputs
    if not path.endswith('.json'):
        raise ValueError("Tokenizer path must be a JSON file")
    
    # Initialize tokenizer with original parameters
    tokenizer = CustomBPETokenizer(
        vocab_size=vocab_size,
        special_tokens=special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )
    
    try:
        # Load trained tokenizer
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        # Validate tokenizer structure
        required_keys = {'token_to_id', 'bpe_ranks'}
        if not required_keys.issubset(tokenizer_data.keys()):
            missing = required_keys - tokenizer_data.keys()
            raise ValueError(f"Invalid tokenizer file, missing keys: {missing}")
            
        # Load into tokenizer instance
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.bpe_ranks = {tuple(k.split('|')): v  # Adjust based on your serialization
                               for k, v in tokenizer_data['bpe_ranks'].items()}
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        
        # Post-load validation
        if len(tokenizer.token_to_id) != len(tokenizer.id_to_token):
            raise ValueError("Tokenizer mapping corruption detected")
            
        if tokenizer.vocab_size != len(tokenizer.token_to_id):
            raise ValueError(f"Vocab size mismatch. Expected {vocab_size}, got {len(tokenizer.token_to_id)}")
            
        return tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {path}: {str(e)}") from e