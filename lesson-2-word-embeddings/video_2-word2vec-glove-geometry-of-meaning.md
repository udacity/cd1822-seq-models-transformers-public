# Video: Word2Vec, GloVe, and the Geometry of Meaning
*Lesson 2, Video 2 | Topic: Finding Meaning with Word Embeddings*

---

## Compelling Hook / Opening Question

> *"King minus man plus woman equals... queen. Paris minus France plus Italy equals... Rome. This isn't magic — it's mathematics. Word embeddings don't just place similar words near each other; they capture relationships as geometric directions. And why does 'king - man + woman ≈ queen' actually work?"*

---

## Introduction
We know embeddings place similar words close together. But the real magic is deeper: embeddings capture semantic RELATIONSHIPS as consistent geometric patterns. Today we'll explore how embeddings learn these patterns, and why word arithmetic reveals the hidden structure of language.

## How Embeddings Are Learned

**Word2Vec** learns by predicting context — words appearing in similar contexts get similar vectors:
- "dog" and "cat" both appear near "pet", "furry", "animal"
- So their vectors become similar!

**GloVe** analyzes how often words appear together across the entire corpus, then factorizes these patterns into dense vectors.

Both produce dense 100-300 dimensional vectors with semantic similarity built in.

## The Geometry of Meaning

The breakthrough discovery: semantic relationships form consistent geometric patterns.

**Word Analogies as Vector Arithmetic:**
```
king - man + woman ≈ queen
```

Why does this work?
- The vector from "man" to "king" captures "royalty"
- The vector from "woman" to "queen" captures the SAME direction
- Gender, tense, geography — all become consistent directions in the space

**More Examples:**
```
paris - france + italy = rome       (capital city relationship)
walked - walk + swim = swam         (past tense relationship)
```

## Pretrained Embeddings

Ready-to-use embeddings trained on massive corpora:
- **GloVe**: Wikipedia + Gigaword (6B tokens)
- **Word2Vec**: Google News (100B tokens)
- **FastText**: Handles subwords, better for rare words

These capture general language patterns and work well for most NLP tasks.

## Closing — The Foundation for Sequence Models
Word embeddings transform discrete symbols into a continuous space where meaning has geometry. This is the foundation for RNNs, attention, and transformers.

But static embeddings have a limitation: "bank" has the same vector whether it means "river bank" or "financial bank." Contextual embeddings (BERT, GPT) solve this — but that's a later lesson.

---

## Important Topics

| Topic | Key Concept |
|-------|-------------|
| **Word2Vec** | Learn embeddings by predicting context |
| **GloVe** | Learn from global co-occurrence statistics |
| **Word Analogies** | Semantic relationships as vector arithmetic |
| **Pretrained Embeddings** | GloVe, Word2Vec trained on billions of words |
| **Static vs. Contextual** | Same vector regardless of context (limitation) |

---

## Use Cases & Examples to Discuss

1. **Analogy Tests** — "king - man + woman = ?" used to evaluate embedding quality
2. **News Recommendation** — Find semantically similar articles using embedding similarity
3. **Semantic Search** — Find relevant documents even without exact keyword matches
4. **Transfer Learning** — Use pretrained embeddings as starting point for new tasks
