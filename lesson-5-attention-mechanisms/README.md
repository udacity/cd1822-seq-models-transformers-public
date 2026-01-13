# Lesson 5: Attention Mechanisms

## Overview
Understanding and implementing attention mechanisms to improve sequence models by allowing dynamic focus on relevant input parts.

## Directory Structure
- **`demo/`** - Demonstration notebook showing attention mechanism implementations and visualizations
- **`exercises/`** - Hands-on attention implementation exercises
  - **`starter/`** - Exercise templates with TODOs
  - **`solution/`** - Complete solutions

## Learning Objectives
- Understand the attention mechanism concept and motivation (solving information bottleneck)
- Implement different types of attention (additive/Bahdanau, multiplicative/Luong, scaled dot-product)
- Apply attention to seq2seq models for improved translation performance
- Visualize attention weights for model interpretability and debugging
- Master the mathematical foundations leading to self-attention and Transformers

### 🎯 Learning Progression
1. **Demo first** → Understand why fixed context vectors limit seq2seq performance
2. **Starter exercise** → Implement attention mechanisms from scratch
3. **Solution reference** → See optimized attention implementations with visualization
4. **Real applications** → Apply to neural machine translation with attention analysis

## Key Concepts

### The Information Bottleneck Problem
```
Without Attention:
Long Input → [Encoder] → Single Context Vector → [Decoder] → Output
              (Information loss for long sequences)

With Attention:
Long Input → [Encoder] → All Hidden States → [Attention] → [Decoder] → Output
              (Dynamic access to all input information)
```

### Attention Mechanisms Types

#### Additive (Bahdanau) Attention
```python
# Attention energy computation
e_{t,s} = v^T tanh(W_1 h_t + W_2 h_s)
α_{t,s} = softmax(e_{t,s})  # Attention weights
c_t = Σ α_{t,s} h_s         # Context vector
```

#### Multiplicative (Luong) Attention
```python
# Simpler energy computation
e_{t,s} = h_t^T W h_s       # or h_t^T h_s for dot-product
α_{t,s} = softmax(e_{t,s})
c_t = Σ α_{t,s} h_s
```

### Attention Visualization
- **Attention Heat Maps** - Show which input words the model focuses on
- **Alignment Quality** - Verify sensible word-to-word correspondences
- **Debugging Tool** - Identify model failures and biases

### Real Dataset
- **WMT14 English-German** - Standard translation benchmark with long sentences
- **Multi30K Extended** - Image captions in multiple languages
- **OpenSubtitles** - Conversational data requiring contextual understanding
- Demonstrates attention effectiveness on challenging translation pairs

## Extensions & Next Steps

After mastering attention:
1. **Self-attention** - Apply attention within single sequence (key to Transformers)
2. **Multi-head attention** - Multiple attention patterns in parallel
3. **Positional encoding** - Add sequence position information
4. **Transformer architecture** - Pure attention-based models without RNNs

## Assessment

Complete the exercises and ensure you can:
- [ ] Explain why attention solves the information bottleneck in seq2seq models
- [ ] Implement additive and multiplicative attention mechanisms correctly
- [ ] Integrate attention into encoder-decoder architectures
- [ ] Interpret attention visualizations for model debugging and validation
- [ ] Compare translation quality with and without attention mechanisms

**Success Metric**: You can build an attention-enhanced neural translator that significantly outperforms standard seq2seq on long sentences and produces interpretable attention alignments.

---

🎯 **Remember**: Attention mechanisms revolutionized sequence modeling by enabling dynamic information access, directly leading to the Transformer architecture that dominates modern NLP.
