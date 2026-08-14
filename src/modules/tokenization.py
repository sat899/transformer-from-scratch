"""
tokenization.py

Text tokenization pipelines spanning low-level algorithms to high-level libraries. 
Features raw NumPy/Python implementations of Byte-Pair Encoding (BPE) and WordPiece 
built from scratch, alongside thin wrapper classes around Hugging Face's Tokenizers 
for production-grade vocabulary management.
""" 

from collections import Counter

def split_whitespace(text):
    return text.split() # Splits string by whitespace into a list of word tokens

def build_vocabulary(text, top_k=100):
    tokens = split_whitespace(text) # Converts the input string into a token list
    token_counter = Counter(tokens) # Initializes Counter and counts frequency of each word
    most_common = token_counter.most_common(top_k) # Returns the top_k most frequent (word, count) tuples
    
    # Map each word to a unique integer index
    vocab = {}
    for i, (word, _) in enumerate(most_common): # Loops through top words, generating a unique index i
        vocab[word] = i # Assigns index i as the integer ID for the word
    
    return vocab # Returns the final {word: integer_id} vocabulary dictionary