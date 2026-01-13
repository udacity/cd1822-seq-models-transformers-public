# Video: Tokenization and Vocabulary Building
*Lesson 1, Video 2 | Topic: Breaking Text into Tokens*

---

## Compelling Hook / Opening Question

> *"Neural networks are just math — they multiply matrices and apply functions. They have no idea what 'hello' or 'world' means. So how does ChatGPT generate coherent sentences? How does Google Translate convert English to Japanese? The secret is tokenization — the art of breaking text into pieces and converting those pieces to numbers. But here's the twist: the WAY you tokenize dramatically affects everything downstream — from model accuracy to API costs. A single OpenAI API call can cost $0.01 or $0.10 depending on how text gets tokenized. Let's understand why."*

---

## Introduction
Before any neural network can process text, we need to convert words to numbers. This process — tokenization — seems simple but hides surprising complexity. Today we'll explore the tokenization pipeline, understand the trade-offs between different approaches, and see why modern systems like GPT and BERT made specific tokenization choices.

## The Tokenization Pipeline
Every NLP system follows this pipeline:
```
Raw Text → Tokens → Vocabulary → Indices → Tensors → Model
"Hello world!" → ["Hello", "world", "!"] → {Hello:0, world:1, !:2} → [0, 1, 2] → tensor([0, 1, 2])
```

**Step 1: Tokenization** — Split text into units (words, subwords, or characters)
**Step 2: Vocabulary** — Create a mapping from tokens to unique integers
**Step 3: Encoding** — Convert tokens to their integer indices
**Step 4: Tensorization** — Create PyTorch/TensorFlow tensors for the model

## Word-Level Tokenization — Simple but Flawed
The simplest approach: split on whitespace and punctuation.

```python
"I love machine learning!" → ["I", "love", "machine", "learning", "!"]
```

**Advantages:**
- Intuitive and easy to implement
- Each token has clear meaning

**Critical Flaw — The Unknown Word Problem:**
What happens with "iPhone-15-Pro" or "ChatGPT" or "COVID-19"?
- If not in vocabulary → becomes `<UNK>` (unknown)
- Customer writes "smartwatch" → model sees `<UNK>` → loses critical information!

With a 50,000 word vocabulary, 5-10% of real-world text becomes `<UNK>`. That's massive information loss.

## Character-Level Tokenization — Complete but Slow
Alternative: tokenize every character individually.

```python
"Hello" → ["H", "e", "l", "l", "o"]
```

**Advantages:**
- Vocabulary is tiny (just ~100 characters)
- Zero unknown tokens — can handle ANY text

**Critical Flaw:**
- "Hello world" becomes 11 tokens instead of 2
- Sequences become 5-10x longer
- Models are slower, context windows fill up faster
- Harder to learn word-level patterns

## Subword Tokenization — The Goldilocks Solution
Modern models use subword tokenization — a brilliant middle ground.

**How it works:**
- Common words stay whole: "the", "is", "learning"
- Rare words split into pieces: "tokenization" → ["token", "##ization"]
- Novel words handled gracefully: "smartwatch" → ["smart", "##watch"]

**Popular Algorithms:**
- **BPE (Byte Pair Encoding)** — Used by GPT models
- **WordPiece** — Used by BERT
- **SentencePiece** — Used by T5, handles any language

**Example:**
```python
# BERT tokenizer
"unhappiness" → ["un", "##happiness"]
"COVID-19" → ["CO", "##VI", "##D", "-", "19"]
```

**Why this matters for cost:**
- GPT-4 charges per token
- Efficient tokenization = fewer tokens = lower cost
- A poorly tokenized prompt might cost 2x more!

## Special Tokens — The Vocabulary Essentials
Every vocabulary needs special tokens:

| Token | Purpose |
|-------|---------|
| `<PAD>` | Padding shorter sequences to fixed length |
| `<UNK>` | Unknown words (hopefully rare with subword!) |
| `<BOS>` / `<EOS>` | Beginning/End of sequence markers |
| `[CLS]` | BERT's classification token |
| `[SEP]` | BERT's separator between segments |
| `[MASK]` | BERT's masked language modeling token |

## Vocabulary Size Trade-offs
Choosing vocabulary size involves trade-offs:

| Small Vocab (5K) | Large Vocab (50K) |
|------------------|-------------------|
| More tokens per text | Fewer tokens per text |
| Better unknown handling | More `<UNK>` tokens |
| Larger embedding table | Smaller embedding table |
| Longer sequences | Shorter sequences |

BERT uses ~30,000 tokens. GPT-4 uses ~100,000 tokens. These choices affect model size, speed, and cost.

## Closing — Connecting to Embeddings
Tokenization converts text to integer indices. But integers alone don't capture meaning — "cat" (index 5) isn't mathematically similar to "dog" (index 7). That's where embeddings come in — converting these indices into rich vector representations where similar words are close together. That's our next lesson.

---

## Important Topics

| Topic | Key Concept |
|-------|-------------|
| **Tokenization Pipeline** | Text → Tokens → Vocabulary → Indices → Tensors |
| **Word-Level Tokenization** | Simple but loses information on unknown words |
| **Character-Level Tokenization** | No unknowns but sequences become very long |
| **Subword Tokenization** | Best of both worlds — BPE, WordPiece, SentencePiece |
| **Unknown Token Problem** | `<UNK>` represents information loss |
| **Special Tokens** | `<PAD>`, `<UNK>`, `[CLS]`, `[SEP]`, `[MASK]` |
| **Vocabulary Size Trade-offs** | Smaller = longer sequences; Larger = more unknowns |
| **Cost Implications** | Token count directly affects API pricing |

---

## Use Cases & Examples to Discuss

1. **Customer Support Bot** — Must handle product names like "iPhone-15-Pro-Max" without losing info
2. **OpenAI API Costs** — Same text can be 100 or 200 tokens depending on tokenizer
3. **Multilingual Systems** — SentencePiece handles Japanese, Arabic, Chinese without spaces
4. **Code Generation** — Copilot must tokenize variable names like `getUserById` intelligently
5. **Medical NLP** — Drug names like "acetaminophen" need subword handling
6. **Social Media Analysis** — Hashtags, emojis, slang require robust tokenization
