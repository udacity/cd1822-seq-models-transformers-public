# Lesson 3: RNNs, LSTMs, and GRUs

## Overview
Introduction to recurrent neural networks and their variants for processing sequential data with memory and temporal dependencies.

## Directory Structure
- **`demo/`** - Demonstration notebook showing RNN, LSTM, and GRU implementations from scratch
- **`exercises/`** - Hands-on sequential modeling exercises
  - **`starter/`** - Exercise templates with TODOs
  - **`solution/`** - Complete solutions

## Learning Objectives
- Understand recurrent neural network architectures and the vanishing gradient problem
- Implement LSTM and GRU cells with forget gates and memory mechanisms
- Compare performance of different RNN variants on sequence prediction tasks
- Apply RNNs to real-world sequential data (sentiment analysis, time series)
- Master backpropagation through time (BPTT) for training recurrent models

### 🎯 Learning Progression
1. **Demo first** → Understand why RNNs need memory and how gates solve vanishing gradients
2. **Starter exercise** → Build RNN, LSTM, and GRU cells from scratch
3. **Solution reference** → See optimized implementations and comparison analysis
4. **Real applications** → Apply to sentiment analysis and sequence prediction

## Key Concepts

### The Memory Challenge
```
Traditional NN: f(x) → y (no memory)
                ↓
RNN: f(x_t, h_{t-1}) → h_t, y_t (sequential memory)
```

### RNN Architecture Evolution
- **Vanilla RNN** - Simple recurrence, suffers from vanishing gradients
- **LSTM** - Long Short-Term Memory with sophisticated gating mechanisms
- **GRU** - Gated Recurrent Unit, simplified but effective alternative

### Gate Mechanisms in LSTMs
```python
# Forget Gate: What to throw away from cell state
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# Input Gate: What new information to store
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# Update Cell State
C_t = f_t * C_{t-1} + i_t * C̃_t

# Output Gate: What parts to output
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

### Real Dataset
- **IMDB Movie Reviews** - 25,000 movie reviews for sentiment classification
- **Time Series Data** - Stock prices or weather data for sequence prediction
- Demonstrates RNN effectiveness on variable-length sequential data

## Extensions & Next Steps

After mastering RNNs:
1. **Bidirectional RNNs** - Process sequences in both directions
2. **Sequence-to-sequence models** - Encoder-decoder architectures
3. **Attention mechanisms** - Focus on relevant parts of input sequences
4. **Transformer architecture** - Self-attention as RNN alternative

## Assessment

Complete the exercises and ensure you can:
- [ ] Explain the vanishing gradient problem in vanilla RNNs
- [ ] Implement LSTM and GRU forward passes with proper gate computations
- [ ] Compare RNN variants on sequence modeling benchmarks
- [ ] Train recurrent models for sentiment analysis and sequence prediction
- [ ] Debug common RNN training issues (exploding gradients, slow convergence)

**Success Metric**: You can build a sentiment classifier that accurately processes variable-length movie reviews and explain why LSTMs outperform vanilla RNNs on long sequences.

---

🎯 **Remember**: RNNs introduce the concept of sequential memory to neural networks, laying the foundation for all modern sequence models including Transformers.
