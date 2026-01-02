"""
Thyme: Complete Model Implementation
=====================================
Full O(k) LLM with state evolution and output generation.

Architecture:
    Token → 12 Axioms → 24 Composites → 576 State → Output

Key Features:
- O(k²) = O(576) complexity per token
- Infinite context without cost increase
- Multi-scale processing (6 → 12 → 24)
- Metabolic processing (energy-based gating)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from .thyme_core import (
    ThymeEmbedding, ThymeState, AxiomDecomposer, CompositeExpander,
    MultiScaleProcessor, N_AXIOMS, N_COMPOSITES, N_STATE,
    TARGET_COHERENCE, SPECTRAL_DECAY, PHI, PI
)


# =============================================================================
# METABOLIC PROCESSOR
# =============================================================================

class MetabolicProcessor:
    """
    Energy-based processing inspired by biological metabolism.
    
    High coherence = low energy = efficient processing
    Low coherence = high energy = attention/consolidation needed
    """
    
    def __init__(self, energy_threshold: float = 0.1):
        self.energy_threshold = energy_threshold
        self.energy_history: List[float] = []
        
    def compute_energy(self, coherence: float, curvature: float) -> float:
        """
        Compute metabolic energy cost.
        
        E = (1 - ρ) + κ²
        
        Low coherence and high curvature = high energy
        """
        return (1 - coherence) + curvature ** 2
    
    def should_consolidate(self, energy: float) -> bool:
        """Determine if state consolidation is needed."""
        return energy > self.energy_threshold
    
    def consolidate(self, state: ThymeState) -> np.ndarray:
        """
        Consolidate state by projecting to dominant subspace.
        
        This is analogous to "sleeping" - reorganizing information.
        """
        # SVD of state matrix
        U, s, Vt = np.linalg.svd(state.state)
        
        # Keep top k=7 components (content axioms)
        k = 7
        s_truncated = np.zeros_like(s)
        s_truncated[:k] = s[:k]
        
        # Reconstruct
        consolidated = U @ np.diag(s_truncated) @ Vt
        return consolidated


# =============================================================================
# OUTPUT LAYER
# =============================================================================

class ThymeOutput:
    """
    Output layer: 576 State → Vocabulary logits
    
    Uses the inverse of the embedding pipeline:
    576 → 24 → 12 → embedding_dim → vocab_size
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Projection matrices (learned in training)
        # State (576) → Composites (24)
        self.W_state_to_composite = np.random.randn(N_STATE, N_COMPOSITES) * 0.01
        
        # Composites (24) → Axioms (12)
        self.W_composite_to_axiom = np.random.randn(N_COMPOSITES, N_AXIOMS) * 0.1
        
        # Axioms (12) → Embedding (embedding_dim)
        self.W_axiom_to_embed = np.random.randn(N_AXIOMS, embedding_dim) * 0.1
        
        # Embedding → Vocab (tied with input embeddings in practice)
        self.W_embed_to_vocab = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Generate vocabulary logits from state.
        
        Args:
            state: 576-dimensional state vector
            
        Returns:
            Logits of shape (vocab_size,)
        """
        # 576 → 24
        composites = state @ self.W_state_to_composite
        
        # 24 → 12
        axioms = composites @ self.W_composite_to_axiom
        
        # 12 → embedding_dim
        embed = axioms @ self.W_axiom_to_embed
        
        # embedding_dim → vocab_size
        logits = embed @ self.W_embed_to_vocab
        
        return logits
    
    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """
        Sample a token from logits.
        
        Args:
            logits: Vocabulary logits
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Sampled token ID
        """
        # Apply temperature
        logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample
        return np.random.choice(len(probs), p=probs)


# =============================================================================
# THYME MODEL
# =============================================================================

@dataclass
class ThymeConfig:
    """Configuration for Thyme model."""
    vocab_size: int = 50000
    embedding_dim: int = 224  # √50000
    n_axioms: int = 12
    n_composites: int = 24
    n_state: int = 576
    target_coherence: float = TARGET_COHERENCE
    metabolic_threshold: float = 0.1
    learning_rate: float = 0.1


class ThymeModel:
    """
    Complete Thyme Language Model.
    
    O(k²) complexity per token with infinite context capability.
    """
    
    def __init__(self, config: Optional[ThymeConfig] = None):
        self.config = config or ThymeConfig()
        
        # Components
        self.embedding = ThymeEmbedding(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim
        )
        self.metabolic = MetabolicProcessor(
            energy_threshold=self.config.metabolic_threshold
        )
        self.output = ThymeOutput(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim
        )
        
        # Statistics
        self.total_tokens_processed = 0
        self.consolidation_count = 0
    
    def forward(self, token_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Process a single token and return next-token logits.
        
        Args:
            token_id: Input token ID
            
        Returns:
            Tuple of (logits, diagnostics)
        """
        # Process through embedding pipeline
        state, embed_diag = self.embedding.forward(token_id)
        
        # Compute metabolic energy
        energy = self.metabolic.compute_energy(
            embed_diag['state_coherence'],
            embed_diag['curvature']
        )
        self.metabolic.energy_history.append(energy)
        
        # Check if consolidation needed
        if self.metabolic.should_consolidate(energy):
            consolidated = self.metabolic.consolidate(self.embedding.state)
            self.embedding.state.state = consolidated
            self.consolidation_count += 1
        
        # Generate output logits
        logits = self.output.forward(state)
        
        self.total_tokens_processed += 1
        
        diagnostics = {
            **embed_diag,
            'energy': energy,
            'consolidated': self.metabolic.should_consolidate(energy)
        }
        
        return logits, diagnostics
    
    def generate(
        self,
        prompt_ids: List[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        stop_token: Optional[int] = None
    ) -> Tuple[List[int], List[Dict]]:
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_ids: List of prompt token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_token: Token ID to stop generation
            
        Returns:
            Tuple of (generated_ids, diagnostics)
        """
        # Reset state
        self.embedding.state.reset()
        
        # Process prompt
        all_diagnostics = []
        for token_id in prompt_ids:
            _, diag = self.forward(token_id)
            all_diagnostics.append(diag)
        
        # Generate
        generated_ids = []
        for _ in range(max_tokens):
            # Get logits from current state
            logits = self.output.forward(self.embedding.state.flat)
            
            # Sample next token
            next_token = self.output.sample(logits, temperature)
            generated_ids.append(next_token)
            
            # Check stop condition
            if stop_token is not None and next_token == stop_token:
                break
            
            # Process generated token
            _, diag = self.forward(next_token)
            all_diagnostics.append(diag)
        
        return generated_ids, all_diagnostics
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            'total_tokens': self.total_tokens_processed,
            'consolidations': self.consolidation_count,
            'consolidation_rate': self.consolidation_count / max(1, self.total_tokens_processed),
            'mean_energy': np.mean(self.metabolic.energy_history) if self.metabolic.energy_history else 0,
            'mean_coherence': np.mean(self.embedding.state.coherence_history) if self.embedding.state.coherence_history else 0
        }


# =============================================================================
# COMPLEXITY ANALYSIS
# =============================================================================

def complexity_comparison():
    """Compare Thyme vs Transformer complexity."""
    print("\n" + "=" * 70)
    print("COMPLEXITY COMPARISON: THYME vs TRANSFORMER")
    print("=" * 70)
    
    print(f"""
    THYME ARCHITECTURE:
    -------------------
    Per-token operations:
      - Embedding lookup:        O(d)        = O({224})
      - Axiom decomposition:     O(d × 12)   = O({224 * 12})
      - Composite expansion:     O(12 × 24)  = O({12 * 24})
      - Multi-scale processing:  O(24 × 42)  = O({24 * 42})
      - State update:            O(24²)      = O({24**2})
      - Output projection:       O(576 × V)  = O({576} × V)
      
      TOTAL per token: O(k²) = O({N_STATE})  [excluding vocab projection]
    
    TRANSFORMER ARCHITECTURE:
    -------------------------
    Per-token operations:
      - Embedding lookup:        O(d)
      - Self-attention:          O(N × d)    [N = sequence length]
      - FFN:                     O(d × 4d)
      - Output projection:       O(d × V)
      
      TOTAL per token: O(N × d)  [attention dominates]
    """)
    
    # Numerical comparison
    print("NUMERICAL COMPARISON:")
    print("-" * 70)
    print(f"{'Sequence Length':<20} {'Thyme O(k²)':<20} {'Transformer O(N×d)':<20} {'Speedup':<15}")
    print("-" * 70)
    
    d = 768  # Typical transformer hidden dim
    k_squared = N_STATE
    
    for N in [64, 256, 576, 1024, 4096, 16384]:
        transformer_ops = N * d
        thyme_ops = k_squared
        speedup = transformer_ops / thyme_ops
        print(f"{N:<20} {thyme_ops:<20,} {transformer_ops:<20,} {speedup:<15.1f}×")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Thyme complexity is CONSTANT regardless of sequence length!")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("THYME MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Initialize model
    config = ThymeConfig(vocab_size=10000, embedding_dim=100)
    model = ThymeModel(config)
    
    # Generate from random prompt
    print("\nGenerating from random prompt...")
    prompt = np.random.randint(0, 10000, size=10).tolist()
    generated, diagnostics = model.generate(prompt, max_tokens=20, temperature=0.8)
    
    print(f"Prompt length: {len(prompt)}")
    print(f"Generated length: {len(generated)}")
    
    # Statistics
    stats = model.get_stats()
    print(f"\nModel Statistics:")
    print(f"  Total tokens processed: {stats['total_tokens']}")
    print(f"  Consolidations: {stats['consolidations']}")
    print(f"  Consolidation rate: {stats['consolidation_rate']:.2%}")
    print(f"  Mean energy: {stats['mean_energy']:.4f}")
    print(f"  Mean coherence: {stats['mean_coherence']:.4f}")
    
    # Coherence analysis
    coherences = [d['state_coherence'] for d in diagnostics]
    energies = [d['energy'] for d in diagnostics]
    
    print(f"\nCoherence Analysis:")
    print(f"  Target: {TARGET_COHERENCE:.4f}")
    print(f"  Achieved (mean): {np.mean(coherences):.4f}")
    print(f"  Achieved (final): {coherences[-1]:.4f}")
    
    # Complexity comparison
    complexity_comparison()
