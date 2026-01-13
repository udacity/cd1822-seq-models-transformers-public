# Video: Self-Attention, Multi-Head Attention, and Positional Encoding
*Lesson 6, Video 1 | Topic: Transformers and the Power of Self-Attention*

---

## Compelling Hook / Opening Question

> *"Your content moderation system needs to process millions of comments in real-time. With RNNs, you process words one at a time — a 100-word comment requires 100 sequential steps. But what if you could process ALL words simultaneously? What if every word could directly 'see' every other word in one parallel computation? This is the Transformer — and it's the architecture behind GPT, BERT, and every modern language model."*

---

## Introduction
RNNs process sequences step by step. Attention helped by letting the decoder look back at encoder states. But Transformers take a radical approach: remove recurrence entirely. Instead, use **self-attention** to let every position attend to every other position — all at once, in parallel. This simple change revolutionized NLP.

## The Limitation of Sequential Processing

RNNs must process tokens one at a time. Each step waits for the previous step to complete. For a 100-word sentence, that's 100 sequential operations — even on a GPU with thousands of cores, most sit idle.

Additionally, information from early tokens must survive through many transformations to reach later tokens. Long-range dependencies are hard to capture.

Transformers solve both problems by processing all positions simultaneously.

## Self-Attention: Every Word Sees Every Word

In encoder-decoder attention, the decoder attends to the encoder. In **self-attention**, a sequence attends to ITSELF.

For the sentence "The bank can refuse to lend money":
- "bank" can directly attend to "lend" and "money" (understanding it's financial)
- "refuse" can attend to "can" (understanding the modal verb relationship)
- Every word has direct access to every other word — no information bottleneck

Self-attention computes three things for each word:
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What information do I provide?"

Each word's query matches against all keys to determine attention weights, then retrieves a weighted combination of all values. This happens for ALL words in parallel.

## Multi-Head Attention: Multiple Perspectives

A single attention mechanism learns ONE pattern. But language has many types of relationships:
- Syntactic (subject-verb agreement)
- Semantic (meaning similarity)
- Positional (nearby words)
- Long-range (coreference, topic)

**Multi-head attention** runs multiple attention mechanisms in parallel — each "head" can specialize in a different pattern. BERT uses 12 heads per layer, giving it 12 different perspectives on every relationship in the sequence.

When you visualize different heads, you see striking diversity:
- One head focuses on adjacent words
- Another captures subject-verb connections
- Another finds semantic relationships across the sentence

The outputs from all heads are combined, giving the model a rich, multi-faceted understanding.

## Positional Encoding: Restoring Order

Here's a problem: self-attention treats all positions equally. It sees the sentence as a "bag of words" — "dog bites man" and "man bites dog" would look identical!

RNNs inherently know order because they process sequentially. Transformers need another solution: **positional encoding**.

Before self-attention, we ADD position information to each word embedding. Each position gets a unique pattern that tells the model "this is position 1, this is position 2, etc."

The standard approach uses sine and cosine waves of different frequencies. The result: the model knows word order without needing sequential processing.

## Parallel Processing: The Speed Advantage

With self-attention and positional encoding, Transformers process all positions simultaneously:

- RNN: Process token 1, then token 2, then token 3... (sequential)
- Transformer: Process ALL tokens at once (parallel)

For a 100-token sequence on a GPU, the Transformer can be orders of magnitude faster. This is why modern language models can scale to billions of parameters — the architecture is inherently parallelizable.

## Contextualized Embeddings: Dynamic Meaning

A powerful consequence of self-attention: word representations become **context-dependent**.

Consider: "The bank can refuse to lend money to the person by the river bank."

Word2Vec gives "bank" the same vector both times. But in a Transformer:
- First "bank" attends to "lend", "money", "refuse" → financial meaning
- Second "bank" attends to "river" → geographical meaning

Same word, different embeddings based on context. This is why BERT and GPT understand language so much better than earlier models.

## Closing — The Foundation of Modern AI
Transformers replaced recurrence with self-attention, enabling parallel processing and direct long-range connections. Multi-head attention provides multiple perspectives. Positional encoding preserves word order.

This architecture powers BERT, GPT, and every major language model today. Understanding these components is essential for working with modern NLP.

---

## Important Topics

| Topic | Key Concept |
|-------|-------------|
| **Self-Attention** | Every position attends to every other position |
| **Query, Key, Value** | The three components of attention computation |
| **Multi-Head Attention** | Multiple attention patterns in parallel |
| **Positional Encoding** | Add position information since there's no recurrence |
| **Parallel Processing** | All positions computed simultaneously |
| **Contextualized Embeddings** | Same word gets different embeddings based on context |

---

## Use Cases & Examples to Discuss

1. **Word Sense Disambiguation** — "bank" (financial) vs "bank" (river) get different representations
2. **Content Moderation** — Process millions of comments in real-time with parallel computation
3. **Attention Head Specialization** — Different heads learn syntax, semantics, coreference
4. **Transfer Learning** — Pretrained transformers (BERT, GPT) leverage billions of words of training
