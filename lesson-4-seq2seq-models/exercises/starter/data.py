"""
Data generation and preprocessing for Q&A system.
"""
import random
from collections import Counter

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SEP_TOKEN = 4

# Building blocks for synthetic Q&A data
ENTITIES = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry',
            'Iris', 'Jack', 'Kate', 'Leo', 'Mary', 'Nathan', 'Olivia', 'Peter']

LOCATIONS = ['Paris', 'London', 'Tokyo', 'Sydney', 'Rome', 'Berlin', 'Madrid', 'Beijing',
             'Moscow', 'Cairo', 'Delhi', 'Bangkok', 'Seoul', 'Vienna', 'Oslo', 'Lisbon']

OBJECTS = ['book', 'phone', 'laptop', 'pen', 'key', 'wallet', 'watch', 'camera',
           'tablet', 'notebook', 'badge', 'card', 'bottle', 'bag', 'document', 'file']

PLACES = ['office', 'home', 'library', 'cafe', 'park', 'store', 'hotel', 'airport',
          'station', 'museum', 'theater', 'restaurant', 'school', 'hospital', 'bank', 'mall']

COLORS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange',
          'pink', 'brown', 'gray', 'silver', 'gold', 'navy', 'teal', 'maroon']

ADJECTIVES = ['big', 'small', 'tall', 'short', 'old', 'new', 'fast', 'slow',
              'bright', 'dark', 'clean', 'dirty', 'hot', 'cold', 'soft', 'hard']

FILLER_WORDS = ['the', 'a', 'an', 'is', 'was', 'has', 'had', 'with', 'from', 'very',
                'quite', 'really', 'also', 'just', 'still', 'even', 'many', 'some']


class SyntheticQAGenerator:
    """Generate synthetic Q&A pairs with controlled context lengths."""
    
    def __init__(self, seed=42):
        """Initialize with random seed."""
        random.seed(seed)
    
    def generate_qa_pair(self, context_length='short'):
        """
        Generate a single Q&A pair with controlled context length.
        
        Args:
            context_length: 'short' (~12 words), 'medium' (~30 words), 'long' (~60 words)
        
        Returns:
            (context, question, answer, context_word_count)
        """
        # TODO: Implement Q&A generation logic
        # For short context: "Alice lives in Paris" → "where does Alice live" → "Paris"
        # For medium context: Add adjectives and filler words
        # For long context: Add more filler words and descriptive phrases
        
        # Hints:
        # - Use random.choice() to select from word lists
        # - Generate 3 question types: 'where', 'what', 'who'
        # - Return context, question, answer, len(context.split())
        pass
    
    def generate_dataset(self, n_short=200, n_medium=200, n_long=200, shuffle=True):
        """
        Generate a full dataset with specified number of examples per length.
        
        Args:
            n_short: Number of short context examples
            n_medium: Number of medium context examples
            n_long: Number of long context examples
            shuffle: Whether to shuffle the data
        
        Returns:
            (qa_data, context_lengths) - Lists of Q&A pairs and their word counts
        """
        # TODO: Generate qa_data and context_lengths lists
        # Use generate_qa_pair() method for each length category
        # Combine all examples and optionally shuffle
        pass


def build_vocabulary(qa_pairs):
    """
    Build vocabulary from Q&A pairs.
    
    Args:
        qa_pairs: List of (context, question, answer) tuples
    
    Returns:
        vocab: Dictionary mapping words to indices
    """
    # TODO: Build vocabulary with special tokens
    # Initialize with PAD, SOS, EOS, UNK, SEP tokens
    # Count word frequencies from all contexts, questions, and answers
    # Return vocab dict mapping words to indices
    pass


def encode_text(text, vocab):
    """
    Encode text to token indices using vocabulary.
    
    Args:
        text: Input text string
        vocab: Vocabulary dictionary
    
    Returns:
        List of token indices
    """
    # TODO: Convert text to lowercase and split
    # Use vocab.get() to lookup indices, use UNK_TOKEN for unknown words
    pass


def decode_text(indices, idx2word):
    """
    Decode token indices back to text.
    
    Args:
        indices: List of token indices
        idx2word: Reverse vocabulary mapping indices to words
    
    Returns:
        Decoded text string
    """
    # TODO: Convert indices to words
    # Skip special tokens (PAD, SOS, EOS)
    # Join words with spaces
    pass
