# Lesson 8: Fine-tuning Transformers

## Overview
Practical fine-tuning of pre-trained transformer models for downstream tasks, bridging research and production deployment.

## Directory Structure
- **`demo/`** - DistilBERT fine-tuning demonstration for question answering
  - `finetune_distilbert_qa.ipynb` - Complete fine-tuning walkthrough with SQuAD 2.0
  - `qa_utils.py` - Utility functions for QA tasks and evaluation metrics
- **`exercises/`** - Hands-on fine-tuning exercises
  - **`starter/`** - Exercise templates with educational TODOs for learning
  - **`solution/`** - Complete solutions with production best practices

## Learning Objectives
- Fine-tune pre-trained transformers (DistilBERT) on domain-specific tasks (SQuAD 2.0 QA)
- Implement custom tokenization and data preprocessing for question-answering tasks
- Configure training parameters, learning rates, and hyperparameter optimization strategies
- Evaluate fine-tuned models with domain-specific metrics (exact match, F1 score)
- Understand transfer learning principles and when fine-tuning vs. feature extraction is appropriate

### 🎯 Learning Progression
1. **Demo first** → Understand pre-training + fine-tuning paradigm with real QA example
2. **Starter exercise** → Implement tokenization, training loop, and evaluation for DistilBERT
3. **Solution reference** → See production-quality fine-tuning with proper error handling
4. **Real applications** → Apply to SQuAD 2.0 dataset with 100,000+ question-answer pairs

## Key Concepts

### Transfer Learning Paradigm
```
Pre-training: Large corpus → General language understanding
                      ↓
Fine-tuning: Task-specific data → Specialized performance
```

### Fine-tuning vs. Feature Extraction
- **Fine-tuning** - Update all model parameters (better performance, more data needed)
- **Feature Extraction** - Freeze base model, train only task head (faster, less data)
- **Gradual Unfreezing** - Progressively unfreeze layers during training

### DistilBERT Architecture
```python
# Question Answering with DistilBERT
[CLS] question [SEP] context [SEP] → DistilBERT → Linear layers → start/end logits

# Example tokenization
input_text = "[CLS] What is AI? [SEP] Artificial intelligence is... [SEP]"
tokens = tokenizer(input_text, return_tensors="pt")
start_logits, end_logits = model(**tokens)  # Predict answer span
```

### SQuAD 2.0 Evaluation
```python
# Key metrics for question answering
def evaluate_qa(predictions, ground_truth):
    exact_match = (prediction == ground_truth)  # Exact string match
    f1_score = compute_f1(prediction_tokens, ground_truth_tokens)  # Token overlap
    return {"exact_match": exact_match, "f1": f1_score}
```

### Real Dataset
- **SQuAD 2.0** - 100,000+ question-answer pairs from Wikipedia articles
- **Includes unanswerable questions** - Model must predict when no answer exists
- **Challenging evaluation** - Requires exact match and F1 score computation
- Demonstrates fine-tuning effectiveness on complex reasoning tasks

## Extensions & Next Steps

After mastering fine-tuning:
1. **Multi-task learning** - Fine-tune on multiple tasks simultaneously
2. **Domain adaptation** - Adapt models to specific industries or domains
3. **Efficient fine-tuning** - Parameter-efficient methods (LoRA, adapters)
4. **Model deployment** - Optimize fine-tuned models for production serving

## Assessment

Complete the exercises and ensure you can:
- [ ] Set up proper tokenization for question-answering with context and questions
- [ ] Configure training loops with appropriate learning rates and batch sizes
- [ ] Implement evaluation metrics specific to QA tasks (exact match, F1)
- [ ] Fine-tune DistilBERT to achieve competitive performance on SQuAD 2.0
- [ ] Analyze model predictions and identify common error patterns

**Success Metric**: You can fine-tune DistilBERT on SQuAD 2.0 to achieve >80% exact match and >85% F1 score, demonstrating effective transfer learning from pre-trained representations.

---

🎯 **Remember**: Fine-tuning bridges the gap between large-scale pre-training and task-specific performance, making powerful transformer models accessible for specialized applications.
