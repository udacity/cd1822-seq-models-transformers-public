# Lesson 4: Sequence-to-Sequence Models

## Overview
Building encoder-decoder architectures for transforming one sequence into another, enabling tasks like translation and summarization.

## Directory Structure
- **`demo/`** - Demonstration notebook showing seq2seq model implementation from scratch
- **`exercises/`** - Hands-on sequence transformation exercises
  - **`starter/`** - Exercise templates with TODOs
  - **`solution/`** - Complete solutions

## Learning Objectives
- Understand encoder-decoder architecture for sequence transformation tasks
- Implement sequence-to-sequence models with RNN-based encoders and decoders
- Handle variable-length input/output sequences with padding and masking
- Apply seq2seq models to neural machine translation and text summarization
- Master teacher forcing and inference strategies for sequence generation

### 🎯 Learning Progression
1. **Demo first** → Understand encoder-decoder paradigm and sequence-to-sequence mapping
2. **Starter exercise** → Build translation model with LSTM encoder-decoder
3. **Solution reference** → See production-quality seq2seq implementation
4. **Real applications** → Apply to French-English translation and text summarization

## Key Concepts

### The Encoder-Decoder Paradigm
```
Input Sequence → [Encoder] → Context Vector → [Decoder] → Output Sequence
"Bonjour"     →    RNN     →   Hidden State  →   RNN    →   "Hello"
```

### Seq2Seq Architecture Components
- **Encoder RNN** - Processes input sequence into fixed-size context vector
- **Context Vector** - Compressed representation of entire input sequence
- **Decoder RNN** - Generates output sequence one token at a time
- **Teacher Forcing** - Training strategy using ground truth at each step

### Training vs. Inference
```python
# Training (Teacher Forcing)
for t in range(target_length):
    decoder_output = decoder(target[t-1], hidden_state)  # Use ground truth
    
# Inference (Autoregressive Generation)
for t in range(max_length):
    decoder_output = decoder(previous_output, hidden_state)  # Use model output
    previous_output = decoder_output.argmax()
```

### Real Dataset
- **Multi30K** - 30,000 English-German sentence pairs for translation
- **CNN/DailyMail** - News article summarization dataset
- **WMT Translation** - Professional translation benchmarks
- Demonstrates seq2seq effectiveness on real language pairs

## Extensions & Next Steps

After mastering seq2seq:
1. **Attention mechanisms** - Allow decoder to focus on different encoder states
2. **Beam search** - Better decoding strategies than greedy generation
3. **Copy mechanisms** - Handle out-of-vocabulary words and proper nouns
4. **Transformer seq2seq** - Replace RNNs with self-attention

## Assessment

Complete the exercises and ensure you can:
- [ ] Implement encoder-decoder architecture with proper sequence handling
- [ ] Train seq2seq models using teacher forcing for stable convergence
- [ ] Handle variable-length sequences with appropriate padding strategies
- [ ] Generate sequences during inference without ground truth labels
- [ ] Evaluate translation quality using BLEU scores and human assessment

**Success Metric**: You can build a neural machine translator that converts French sentences to English and achieves reasonable BLEU scores on test data.

---

🎯 **Remember**: Seq2seq models introduced the encoder-decoder paradigm that powers modern translation, summarization, and conversation systems, paving the way for Transformer architectures.
