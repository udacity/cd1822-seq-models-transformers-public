# Video: Popular Metrics for Sequence Models — EM, F1, Recall@K, BLEU/ROUGE
*Lesson 7, Video 2 | Topic: Key Evaluation Metrics*

---

## Compelling Hook / Opening Question

> *"The model predicted 'New York City' but the ground truth was 'New York'. Is that wrong? What about 'NYC'? And if you're ranking search results, does it matter if the right answer is #1 or #3? Different metrics answer different questions about your model. Choosing the wrong metric means optimizing for the wrong thing."*

---

## Introduction
Different sequence tasks need different metrics. Question answering cares about exact correctness and partial overlap. Translation cares about n-gram similarity. Retrieval cares about ranking quality. Understanding what each metric measures — and what it misses — is essential for evaluating your model fairly.

## Exact Match (EM) — The Strict Standard

Exact Match is binary: 100% if the prediction exactly matches the ground truth (after normalization), 0% otherwise.

**Normalization** makes EM practical:
- Convert to lowercase
- Remove punctuation
- Remove articles (a, an, the)
- Strip extra whitespace

After normalization, "The Answer" matches "answer" — both become "answer".

**When EM works well**: Short, factual answers with one clear correct form.

**When EM fails**: "New York City" vs "New York" gets 0% even though it's mostly correct. EM is too harsh when partial credit matters.

## F1 Score — Partial Credit

F1 measures token-level overlap between prediction and ground truth, giving partial credit for partially correct answers.

**How it works**:
- **Precision**: What fraction of predicted tokens are in the ground truth?
- **Recall**: What fraction of ground truth tokens appear in the prediction?
- **F1**: Harmonic mean of precision and recall

For "New York City" vs "New York":
- Precision: 2/3 (two of three predicted words are correct)
- Recall: 2/2 (both ground truth words appear)
- F1: 80%

Compare to EM's harsh 0%. F1 better reflects actual answer quality when answers can be partially correct.

## Recall@K and Precision@K — For Ranked Results

When your model returns multiple results (like search or retrieval), you care about ranking quality.

**Recall@K**: What fraction of relevant items appear in the top K results?
- Recall@5 = 0.8 means 80% of relevant items are in the top 5

**Precision@K**: What fraction of the top K results are relevant?
- Precision@5 = 0.6 means 3 out of 5 top results are relevant

**MRR (Mean Reciprocal Rank)**: Where does the first correct answer appear?
- If correct answer is #1, score = 1.0
- If correct answer is #3, score = 0.33

These metrics matter when users see ranked lists and position affects user experience.

## BLEU — For Translation

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between generated text and reference translations.

**How it works**:
- Count matching unigrams, bigrams, trigrams, 4-grams
- Apply brevity penalty for outputs that are too short
- Combine into a single score (0-100)

**When to use**: Machine translation, where you're comparing generated text against reference translations.

**Limitations**: BLEU rewards surface-level similarity. Two sentences with the same meaning but different words score poorly.

## ROUGE — For Summarization

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated summaries and reference summaries.

**Variants**:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence

**When to use**: Text summarization, where you want to measure content coverage.

**Difference from BLEU**: ROUGE emphasizes recall (did you capture the important content?), while BLEU emphasizes precision (is your output accurate?).

## Choosing the Right Metric

| Task | Primary Metrics |
|------|-----------------|
| Question Answering | EM, F1 |
| Retrieval/Search | Recall@K, MRR |
| Translation | BLEU |
| Summarization | ROUGE |

Often you'll use multiple metrics together — EM for strict correctness, F1 for partial credit, both giving complementary views of model quality.

## Closing — Metrics Shape What You Optimize
The metrics you choose determine what your model learns to do well. If you only measure EM, you optimize for exact string matching. If you only measure BLEU, you optimize for n-gram overlap. Understanding what each metric captures — and what it misses — helps you evaluate fairly and improve effectively.

---

## Important Topics

| Topic | Key Concept |
|-------|-------------|
| **Exact Match (EM)** | Binary score — exact match or nothing |
| **F1 Score** | Token overlap with partial credit |
| **Normalization** | Lowercase, remove punctuation/articles for fair comparison |
| **Recall@K** | Fraction of relevant items in top K |
| **MRR** | Position of first correct answer |
| **BLEU** | N-gram overlap for translation |
| **ROUGE** | Content overlap for summarization |

---

## Use Cases & Examples to Discuss

1. **Q&A Evaluation** — EM + F1 together give strict and lenient views
2. **Search Ranking** — MRR tells you if users find answers quickly
3. **Translation Quality** — BLEU correlates with human judgment (imperfectly)
4. **Summary Coverage** — ROUGE measures if key information is captured
