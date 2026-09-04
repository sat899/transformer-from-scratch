"""
tokenization.py

Text tokenization pipelines spanning low-level algorithms to high-level libraries. 
Features raw NumPy/Python implementations of Byte-Pair Encoding (BPE) and WordPiece 
built from scratch, alongside thin wrapper classes around Hugging Face's Tokenizers 
for production-grade vocabulary management.
""" 

from collections import Counter
import numpy as np

def _split_whitespace(text):
    return text.split() # Splits string by whitespace into a list of word tokens

def build_vocabulary(text, top_k=100):
    """
    creates a vocabulary (dict) mapping tokens to indices
    """
    
    tokens = _split_whitespace(text) # Converts the input string into a token list
    token_counter = Counter(tokens) # Initializes Counter and counts frequency of each token
    most_common = token_counter.most_common(top_k) # Returns the top_k most frequent (token, count) tuples
    
    # Map each token to a unique integer index
    vocab = {}
    for i, (token, _) in enumerate(most_common): # Loops through top tokens, generating a unique index i
        vocab[token] = i # Assigns index i as the integer ID for the token
    
    return vocab # Returns the final {token: integer_id} vocabulary dictionary

def create_inputs(text, vocab, seq_len):
    """
    encodes a string into integer token ids and packs them into (batch, seq_len)
    """
    
    tokens = _split_whitespace(text) # Converts the input string into a token list
    
    # Map each token to its integer id using the provided vocabulary
    ids = []
    for token in tokens: # Loops through tokens in order, including repeats
        ids.append(vocab[token]) # Looks up the integer ID for the token
    
    n = (len(ids) // seq_len) * seq_len # Drops leftover tokens that do not fill a full sequence
    input_ids = np.array(ids[:n]).reshape(-1, seq_len) # Packs ids into shape (batch, seq_len)
    
    return input_ids # Returns the batched integer token ids    