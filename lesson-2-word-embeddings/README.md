# Lesson 2: Word Embeddings

## Overview
Converting discrete tokens into dense vector representations that capture semantic meaning and relationships.

## Directory Structure
- **`demo/`** - Demonstration notebook showing embedding concepts and pretrained models
- **`exercises/`** - Hands-on embedding exercises
  - **`starter/`** - Exercise templates with TODOs
  - **`solution/`** - Complete solutions

## Learning Objectives
- Understand limitations of one-hot encoding for semantic tasks
- Explore different embedding approaches (Word2Vec, GloVe)
- Measure semantic similarity using cosine similarity
- Perform word arithmetic and analogy tasks
- Integrate embedding layers into PyTorch neural networks
- **Solution Overview**: Dense vectors capture semantic relationships
- **Hands-on Exploration**: GloVe embeddings and word arithmetic
- **PyTorch Integration**: Ready-to-use `nn.Embedding` layers

**Approach**: Interactive exploration - students discover embedding properties

### 🏋️ Exercises (`exercises/`)
**Time Estimate:** 15-20 minutes | **Purpose:** Practical Implementation

#### Starter Version (`exercises/starter/exercise_starter.ipynb`) 
**Status:** 🚧 Template with guided implementation

**Core Tasks:**
1. **Embedding Comparison** - Compare Word2Vec, GloVe, random vectors
2. **Similarity Analysis** - Build news article recommendation engine
3. **Visualization** - Reduce dimensions and plot semantic clusters
4. **Custom Training** - Train domain-specific embeddings on news data
5. **Evaluation** - Measure embedding quality on similarity tasks

#### Solution Version (`exercises/solution/exercise_solution.ipynb`)
**Status:** ✅ Complete implementation with analysis

**Advanced Features:**
- Multiple embedding model comparisons
- Interactive similarity search and recommendation
- Beautiful visualizations of embedding spaces  
- Quantitative evaluation metrics
- Integration with AG News dataset (5,000 articles)

### 🎯 Learning Progression
1. **Demo first** → Understand why embeddings matter and how they work
2. **Starter exercise** → Build recommendation systems with embeddings  
3. **Solution reference** → See production-quality embedding applications
4. **Real dataset** → Apply to 5,000 news articles across 4 categories

## Key Concepts

### The Embedding Revolution
```
One-Hot: [0,0,1,0,0...] (50,000 dims, sparse, no similarity)
                ↓
Embedding: [0.2, -0.8, 0.1...] (300 dims, dense, semantic similarity)
```


### The Geometry of Meaning

- **Semantic similarity** → Vector cosine similarity
- **Word relationships** → Consistent vector directions  
- **Analogies** → Vector arithmetic (king - man + woman ≈ queen)
- **Clustering** → Related concepts group together

### Real Dataset
- **AG News Dataset** - 5,000 news article titles across 4 categories
- Accessed via HuggingFace `datasets` library  
- Categories: World, Sports, Business, Technology
- Perfect for testing embedding-based recommendations

## Extensions & Next Steps

After mastering embeddings:
1. **Contextual embeddings** - Explore BERT, GPT for context-aware representations
2. **Multilingual embeddings** - Cross-language similarity and translation
3. **Domain adaptation** - Fine-tune embeddings for specific industries  
4. **Embedding evaluation** - Systematic testing of embedding quality

## Assessment

Complete the exercises and ensure you can:
- [ ] Explain why embeddings outperform one-hot encoding
- [ ] Load and use pretrained word vectors effectively  
- [ ] Measure semantic similarity between words and documents
- [ ] Build a functional content recommendation system
- [ ] Visualize and interpret high-dimensional embedding spaces

**Success Metric**: You can build a news recommendation system that suggests semantically related articles even when they don't share exact keywords.

---

🎯 **Remember**: Word embeddings bridge the gap between symbolic and continuous representations, enabling neural networks to understand meaning rather than just manipulating symbols.
