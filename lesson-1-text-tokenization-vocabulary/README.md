# Lesson 1: Text Tokenization & Vocabulary Building

## Overview
Fundamental preprocessing techniques for converting raw text into numerical sequences for neural networks.

## Directory Structure
- **`demo/`** - Demonstration notebook showing tokenization concepts and strategies
- **`exercises/`** - Hands-on tokenization exercises
  - **`starter/`** - Exercise templates with TODOs
  - **`solution/`** - Complete solutions

## Learning Objectives
- Understand why text needs to be converted to numbers for neural networks
- Implement different tokenization strategies (word-level, character-level, subword)
- Build and manage vocabularies with frequency filtering
- Handle unknown words and vocabulary trade-offs
- Compare custom tokenizers with modern approaches (BERT tokenization)
1. **Demo first** → Understand why tokenization matters
2. **Starter exercise** → Build core implementation skills  
3. **Solution reference** → See production-quality code
4. **Compare approaches** → Discover modern solutions

## Key Concepts

### The Tokenization Pipeline
```
Raw Text → Tokens → Vocabulary → Indices → PyTorch Tensors → Embeddings
```
## 🔄 Tokenization Pipeline Example

```python
# Step 1: Raw Text → Tokens
text = "Hello world!"
tokens = ["Hello", "world", "!"]

# Step 2: Tokens → Vocabulary (mapping)
vocab = {"Hello": 0, "world": 1, "!": 2, "<UNK>": 3}

# Step 3: Vocabulary → Indices (apply mapping)
indices = [0, 1, 2]  # Python list of integers

# Step 4: Indices → PyTorch Tensors
import torch
tensor_ids = torch.tensor([0, 1, 2])  # tensor([0, 1, 2])

# Step 5: Tensors → Embeddings (lookup dense vectors)
embedding_layer = torch.nn.Embedding(4, 2)  # vocab_size=4, embed_dim=2
embeddings = embedding_layer(tensor_ids)
# Result: tensor([[0.1, 0.3], [0.8, 0.2], [0.4, 0.9]])
```

### Real Dataset
- **Bitext Customer Support Dataset** - Real customer messages for tokenization practice
- Accessed via HuggingFace `datasets` library
- Provides diverse examples of customer inquiries and terminology

## Extensions & Next Steps

After mastering this lesson:
1. **Multi-lingual tokenization** - Explore `bert-base-multilingual-cased`
2. **Domain adaptation** - Train custom tokenizers for specialized domains
3. **Efficiency optimization** - Fast tokenizers with Rust backend
4. **Sequence modeling** - Ready for RNNs, LSTMs, and Transformers!

## Assessment

Complete the exercises and ensure you can:
- [ ] Explain why order matters in sequence data
- [ ] Implement word-level and character-level tokenizers
- [ ] Identify trade-offs between tokenization strategies
- [ ] Create properly padded training batches
- [ ] Choose appropriate tokenization for different scenarios

🎯 **Remember**: Tokenization is the foundation of all NLP models. Understanding these concepts deeply will help you debug issues and make better architectural decisions in advanced sequence models.
