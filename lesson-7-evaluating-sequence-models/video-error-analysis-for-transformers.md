# Video: Error Analysis for Transformers — Qualitative Inspection
*Lesson 7, Video 3 | Topic: Finding Model Blind Spots*

---

## Compelling Hook / Opening Question

> *"Your model achieves 75% F1 — not bad! But WHERE is it failing? Is it struggling with long contexts? Ambiguous questions? Questions that have no answer? Aggregate metrics hide crucial details. A model that's 90% accurate on easy questions and 20% accurate on hard ones looks the same as a model that's uniformly 55% accurate. Error analysis reveals the difference — and shows you exactly what to fix."*

---

## Introduction
Aggregate metrics tell you HOW WELL your model performs. Error analysis tells you WHERE and WHY it fails. By systematically examining errors, you discover patterns: certain question types, context lengths, or edge cases where the model struggles. This qualitative inspection is essential for targeted improvement.

## Answerable vs Unanswerable Questions

Modern Q&A datasets like SQuAD 2.0 include questions that CANNOT be answered from the given context. The model should recognize these and return nothing.

This is surprisingly hard. Models are trained to find answers, so they often hallucinate plausible-sounding responses even when no answer exists.

**Error pattern**: Model gives confident answers to unanswerable questions.

**What it reveals**: The model hasn't learned to say "I don't know" — a critical capability for real-world deployment.

Breaking down metrics by answerable vs unanswerable questions often reveals dramatic differences in performance.

## Span Mismatch Errors

In extractive Q&A, the model selects a span from the context. Common span errors include:

**Too short**: Ground truth is "the United States of America", model predicts "United States"
- Partial credit via F1, but EM fails
- Model is close but not precise

**Too long**: Ground truth is "Paris", model predicts "Paris, France"
- Extra information isn't wrong, but doesn't match
- Suggests model isn't learning span boundaries well

**Off by one**: Prediction includes or excludes a word at the boundary
- Often punctuation or articles
- Normalization helps, but indicates imprecise span detection

Examining span mismatches reveals whether errors are minor boundary issues or fundamental misunderstandings.

## Question Type Analysis

Different question types have different difficulty:

**Factoid questions** ("What year...?", "Who is...?") — Usually easier, clear answers

**Reasoning questions** ("Why did...?", "How does...?") — Harder, require inference

**Comparison questions** ("What's the difference...?") — Require synthesizing multiple facts

**Yes/No questions** — Model must infer answer not explicitly stated

Breaking down performance by question type reveals which reasoning capabilities your model lacks.

## Context Length Effects

Transformers handle long contexts better than RNNs, but they're not immune to length effects:

**Short contexts**: Model performs well — answer is easy to locate

**Long contexts**: Performance may drop — more distractors, harder to focus

**Answer position**: Does performance depend on whether the answer is early or late in the context?

If your model struggles with long contexts, you might need longer training examples or better attention mechanisms.

## Systematic Error Patterns

Look for patterns that repeat across multiple errors:

**Domain gaps**: Model trained on Wikipedia fails on technical documentation

**Linguistic patterns**: Struggles with negation ("Who did NOT attend?") or superlatives ("the largest...")

**Formatting issues**: Dates, numbers, or named entities consistently wrong

**Ambiguity handling**: When multiple valid answers exist, which does the model choose?

These patterns point to specific weaknesses you can address through data augmentation, architectural changes, or fine-tuning.

## From Errors to Improvements

Error analysis isn't just diagnosis — it guides improvement:

**Data augmentation**: If the model fails on negation, add more negation examples

**Architecture changes**: If long contexts fail, consider hierarchical attention

**Training adjustments**: If unanswerable detection is poor, weight those examples higher

**Post-processing**: If span boundaries are off, add boundary refinement

Without error analysis, you're improving blindly. With it, you know exactly what to fix.

## Closing — Metrics Tell You What, Errors Tell You Why
Aggregate metrics are essential for measuring progress, but they hide the details. Error analysis reveals the patterns behind the numbers — which question types fail, which contexts confuse the model, which edge cases break it. This qualitative inspection is how you turn a good model into a great one.

---

## Important Topics

| Topic | Key Concept |
|-------|-------------|
| **Answerable vs Unanswerable** | Can the model say "I don't know"? |
| **Span Mismatch** | Too short, too long, or off-by-one predictions |
| **Question Type Analysis** | Factoid vs reasoning vs comparison |
| **Context Length Effects** | Performance degradation on long inputs |
| **Systematic Patterns** | Repeating errors reveal specific weaknesses |
| **Error-Driven Improvement** | Use error patterns to guide fixes |

---

## Use Cases & Examples to Discuss

1. **SQuAD 2.0 Analysis** — Answerable questions often score much higher than unanswerable
2. **Span Boundary Inspection** — "New York City" vs "New York" — close but not exact
3. **Question Type Breakdown** — "What" questions easier than "Why" questions
4. **Production Debugging** — User complaints often cluster around specific failure modes
