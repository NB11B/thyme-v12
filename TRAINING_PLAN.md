# Thyme LLM Comprehensive Training Plan

## Current Status

The initial training on 8 classic books (~4.8 MB) achieved:
- **Perplexity**: 57.1
- **State Size**: 2,304 bytes (constant)
- **Training Time**: 110 minutes

## Goal: Push Training Further

To significantly improve the model, we need:
1. **Larger corpus** - More diverse, comprehensive text
2. **Larger vocabulary** - Better tokenization coverage
3. **Deeper architecture** - More axiom processing layers
4. **Longer training** - More epochs with better scheduling

---

## Recommended Corpus Options

### Option A: Expanded Gutenberg (Recommended Start)

| Metric | Value |
|--------|-------|
| Source | HuggingFace `manu/project_gutenberg` |
| Size | 14.4 GB |
| Books | 75,570 |
| Languages | English (primary), 10+ others |
| Download Time | ~30 minutes |

**Command:**
```bash
python prepare_corpus.py --gutenberg --language en
```

**Expected Results:**
- PPL improvement: 57 → ~35-40
- Training time: ~8-12 hours (2 GPUs)

### Option B: Gutenberg + Wikipedia

| Metric | Value |
|--------|-------|
| Gutenberg | 14.4 GB |
| Wikipedia | 6.4 GB |
| Total | ~20 GB |
| Download Time | ~1 hour |

**Command:**
```bash
python prepare_corpus.py --all --language en
```

**Expected Results:**
- PPL improvement: 57 → ~25-30
- Training time: ~16-24 hours (2 GPUs)

### Option C: Full Pile Subset (Advanced)

| Component | Size | Purpose |
|-----------|------|---------|
| Books3/PG-19 | 10 GB | Narrative |
| Wikipedia | 6 GB | Knowledge |
| ArXiv | 10 GB | Scientific |
| StackExchange | 10 GB | Q&A |
| **Total** | ~36 GB | |

**Expected Results:**
- PPL improvement: 57 → ~15-20
- Training time: ~48-72 hours (2 GPUs)

---

## Training Configuration

### Small Scale (Quick Test)
```bash
python train_corpus.py --epochs 5 --batch-size 32 --max-files 100
```

### Medium Scale (Recommended)
```bash
python train_corpus.py --epochs 20 --batch-size 16 --lr 1e-3
```

### Large Scale (Best Results)
```bash
python train_corpus.py --epochs 50 --batch-size 8 --lr 5e-4
```

---

## Architecture Improvements

### Current (train_books.py)
| Parameter | Value |
|-----------|-------|
| Vocab Size | 10,000 |
| Embed Dim | 256 |
| Layers | 1 |
| State | 576 (24×24) |

### Enhanced (train_corpus.py)
| Parameter | Value |
|-----------|-------|
| Vocab Size | 32,000 |
| Embed Dim | 512 |
| Layers | 2 |
| State | 576 (24×24) |
| Position Embeddings | Yes |
| Gradient Accumulation | 4 |
| Mixed Precision | Yes |

---

## Thyme-Aligned Corpus Design

The corpus should cover all **7 Content Axioms**:

| Axiom | Domain | Sources |
|-------|--------|---------|
| Entity | Concrete/Abstract | Wikipedia, encyclopedias |
| Animacy | Living/Non-living | Biology texts, nature writing |
| Valence | Positive/Negative | Literature, emotional texts |
| Sociality | Individual/Collective | Social texts, dialogues |
| Modality | Physical/Mental | Philosophy, psychology |
| Scale | Small/Large | Scientific texts (physics, astronomy) |
| Openness | Bounded/Unbounded | Mathematics, logic |

---

## Step-by-Step Instructions

### 1. Prepare Environment
```bash
cd /home/ubuntu/thyme-v12
pip install datasets tokenizers tqdm torch
```

### 2. Download Corpus
```bash
# Option A: Just Gutenberg
python prepare_corpus.py --gutenberg --language en

# Option B: Gutenberg + Wikipedia
python prepare_corpus.py --all --language en

# Check what you have
python prepare_corpus.py --stats
```

### 3. Train Model
```bash
# Quick test (1-2 hours)
python train_corpus.py --epochs 5 --max-files 1000

# Full training (8-24 hours)
python train_corpus.py --epochs 20
```

### 4. Evaluate
```bash
# Generate samples
python -c "
from train_corpus import *
model = ThymeLM(32000)
model.load_state_dict(torch.load('checkpoints/thyme_corpus_best.pt'))
# ... generate
"
```

---

## Expected Improvements

| Corpus Size | Expected PPL | Training Time | Memory |
|-------------|--------------|---------------|--------|
| 5 MB (current) | 57 | 2 hours | 2 GB |
| 500 MB | ~40 | 8 hours | 4 GB |
| 5 GB | ~30 | 24 hours | 8 GB |
| 20 GB | ~20 | 72 hours | 16 GB |

The key insight: **State size remains constant at 2,304 bytes** regardless of corpus size or training duration.

---

## The Formula Governs Training

**π² = (7φ² + √2) / 2**

The 12-axiom structure (7 content + 5 relational) creates a natural compression that:
- Captures semantic meaning in 12 dimensions
- Expands to 24 composites for interaction
- Maintains 576-dimensional state (24×24)

This is why the model can learn from massive corpora while maintaining constant memory.

---

## Files Created

| File | Purpose |
|------|---------|
| `prepare_corpus.py` | Download and prepare training data |
| `train_corpus.py` | Enhanced training script |
| `train_books.py` | Original training script |
| `TRAINING_PLAN.md` | This document |

---

*Last updated: January 2, 2026*
