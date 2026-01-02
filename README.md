# Thyme: 12-Axiom Language Model

A novel language model architecture based on semantic compression through 12 fundamental axioms.

## Key Features

- **Constant Memory**: 2,304 bytes state regardless of context length
- **O(1) Complexity**: Fixed computation per token
- **12-Axiom Decomposition**: 7 content + 5 relational semantic primitives
- **Lie Algebra Structure**: so(2,1) ⊕ ℝ² governs semantic transformations

## Architecture

```
Input Token → Embedding (256-dim)
           → Axiom Decomposition (12-dim)
           → Composite Expansion (24-dim)
           → State Update (24×24 = 576-dim)
           → Output Projection → Logits
```

## Installation

```bash
pip install torch tokenizers numpy
```

## Quick Start

### Training on Books

```bash
# Download books (or use your own corpus)
mkdir -p books
# Add .txt files to books/

# Train
python train_books.py
```

### Using a Trained Model

```python
import torch
from thyme_torch import ThymeLM
from tokenizers import Tokenizer

# Load
tokenizer = Tokenizer.from_file("checkpoints/tokenizer_books.json")
model = ThymeLM(vocab_size=tokenizer.get_vocab_size())
model.load_state_dict(torch.load("checkpoints/thyme_books_best.pt"))

# Generate
prompt = "It was a dark and stormy night"
prompt_ids = torch.tensor([[2] + tokenizer.encode(prompt).ids])  # 2 = BOS
generated = model.generate(prompt_ids, max_new=50, temp=0.7)
print(tokenizer.decode(generated[0].tolist()))
```

## Files

| File | Description |
|------|-------------|
| `thyme_core.py` | Core architecture (NumPy reference) |
| `thyme_torch.py` | PyTorch implementation |
| `thyme_bpe.py` | BPE tokenization version |
| `train_books.py` | Book corpus training script |
| `benchmark.py` | Thyme vs Transformer comparison |
| `geometric_operators.py` | Lie algebra derivation |
| `lie_algebra_proper.py` | Full so(2,1) structure |

## Theory

### The 12 Axioms

**Content (7):**
1. Entity - concrete vs abstract
2. Animacy - living vs non-living
3. Valence - positive vs negative
4. Sociality - individual vs collective
5. Modality - physical vs mental
6. Scale - small vs large
7. Openness - bounded vs unbounded

**Relational (5):**
1. Cardinality - quantity/scaling
2. Gradability - comparison/projection
3. Deixis - reference frame
4. Dynamism - action/rotation
5. Temporality - time ordering

### Key Formula

```
π² ≈ (7φ² + √2) / 2
```

Where:
- 7 = content axioms
- φ = golden ratio
- √2 ≈ 7/5 = content/relational ratio

### Lie Algebra

The 5 relational operators form **so(2,1) ⊕ ℝ²**:
- so(2,1) ≅ sl(2,ℝ): Deixis, Gradability, Dynamism
- ℝ²: Temporality, Cardinality (center)

This is the same symmetry structure as 2+1 dimensional spacetime.

## Benchmarks

| Metric | Thyme | Transformer |
|--------|-------|-------------|
| Parameters | ~8M | ~8M |
| Inference Speed | 1,534 tok/s | 411 tok/s |
| Memory @1K tokens | 2.3 KB | 1 MB |
| Memory @10K tokens | 2.3 KB | 10 MB |
| Memory Scaling | O(1) | O(n) |

## License

MIT

## Citation

```
@misc{thyme2025,
  title={Thyme: Semantic Compression via 12-Axiom Decomposition},
  year={2025}
}
```
