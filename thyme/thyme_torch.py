"""
Thyme PyTorch Implementation
=============================
Full differentiable implementation with proper backprop through all layers.

Architecture:
    Input → Embedding → Axiom Decomposition (12) → Composite Expansion (24) 
    → State Update (24×24=576) → Output Projection → Logits

The key insight: the 576-dim state is a FIXED-SIZE representation
regardless of sequence length, giving O(1) memory per token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import json
import os
from openai import OpenAI

# =============================================================================
# CONSTANTS (from theory)
# =============================================================================

N_AXIOMS = 12          # 7 content + 5 relational
N_COMPOSITES = 24      # 12 × 2 (positive/negative aspects)
N_STATE = 576          # 24 × 24 (full interaction matrix)
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


# =============================================================================
# THYME LAYERS
# =============================================================================

class AxiomDecomposer(nn.Module):
    """
    Decompose embeddings into 12 semantic axioms.
    
    The 12 axioms are:
    Content (7): Entity, Animacy, Valence, Sociality, Modality, Scale, Openness
    Relational (5): Cardinality, Gradability, Deixis, Dynamism, Temporality
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, N_AXIOMS)
        self.norm = nn.LayerNorm(N_AXIOMS)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, embed_dim) → (batch, 12)"""
        axioms = self.linear(x)
        axioms = self.norm(axioms)
        axioms = torch.tanh(axioms)  # Bounded [-1, 1]
        return axioms


class CompositeExpander(nn.Module):
    """
    Expand 12 axioms to 24 composites.
    
    Each axiom has positive and negative aspects,
    creating a richer representation.
    """
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(N_AXIOMS, N_COMPOSITES)
        self.norm = nn.LayerNorm(N_COMPOSITES)
        
    def forward(self, axioms: torch.Tensor) -> torch.Tensor:
        """axioms: (batch, 12) → (batch, 24)"""
        composites = self.linear(axioms)
        composites = self.norm(composites)
        composites = torch.tanh(composites)
        return composites


class StateManager(nn.Module):
    """
    Manage the 576-dimensional state (24×24 matrix).
    
    The state captures all pairwise interactions between composites.
    It's updated with exponential moving average for stability.
    """
    
    def __init__(self, decay: float = 0.9):
        super().__init__()
        self.decay = decay
        # Learnable mixing weights
        self.mix_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, composites: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        """
        composites: (batch, 24)
        prev_state: (batch, 24, 24)
        returns: (batch, 24, 24)
        """
        # Outer product: (batch, 24, 1) × (batch, 1, 24) → (batch, 24, 24)
        new_contribution = torch.bmm(
            composites.unsqueeze(2),
            composites.unsqueeze(1)
        )
        
        # Exponential moving average
        mix = torch.sigmoid(self.mix_weight)  # Learnable mixing rate
        new_state = self.decay * prev_state + mix * new_contribution
        
        return new_state


class ThymeCore(nn.Module):
    """
    Core Thyme module: processes one token and updates state.
    """
    
    def __init__(self, embed_dim: int, decay: float = 0.9):
        super().__init__()
        self.axiom_decomposer = AxiomDecomposer(embed_dim)
        self.composite_expander = CompositeExpander()
        self.state_manager = StateManager(decay)
        
    def forward(
        self, 
        embedding: torch.Tensor, 
        prev_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        embedding: (batch, embed_dim)
        prev_state: (batch, 24, 24)
        returns: (state_flat, new_state) where state_flat is (batch, 576)
        """
        axioms = self.axiom_decomposer(embedding)
        composites = self.composite_expander(axioms)
        new_state = self.state_manager(composites, prev_state)
        state_flat = new_state.view(new_state.size(0), -1)  # (batch, 576)
        
        return state_flat, new_state


# =============================================================================
# FULL THYME MODEL
# =============================================================================

class ThymeLM(nn.Module):
    """
    Thyme Language Model.
    
    Full model with embedding, core processing, and output projection.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        decay: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Core Thyme processing
        self.core = ThymeCore(embed_dim, decay)
        
        # Output projection: 576 → vocab_size
        self.output_proj = nn.Sequential(
            nn.Linear(N_STATE, N_STATE // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(N_STATE // 2, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize the 24×24 state matrix."""
        return torch.zeros(batch_size, N_COMPOSITES, N_COMPOSITES, device=device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: (batch, seq_len)
        state: (batch, 24, 24) or None
        
        returns: (logits, final_state)
            logits: (batch, seq_len, vocab_size)
            final_state: (batch, 24, 24)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if state is None:
            state = self.init_state(batch_size, device)
        
        # Get embeddings
        embeds = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        embeds = self.embed_dropout(embeds)
        
        # Process sequence
        all_logits = []
        for t in range(seq_len):
            embed_t = embeds[:, t, :]  # (batch, embed_dim)
            state_flat, state = self.core(embed_t, state)
            logits_t = self.output_proj(state_flat)  # (batch, vocab_size)
            all_logits.append(logits_t)
        
        logits = torch.stack(all_logits, dim=1)  # (batch, seq_len, vocab_size)
        
        return logits, state
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.size(0)
        
        # Process prompt
        state = self.init_state(batch_size, device)
        
        with torch.no_grad():
            # Encode prompt
            for t in range(prompt_ids.size(1)):
                embed_t = self.embedding(prompt_ids[:, t])
                _, state = self.core(embed_t, state)
            
            # Generate
            generated = [prompt_ids]
            current_token = prompt_ids[:, -1:]
            
            for _ in range(max_new_tokens):
                embed_t = self.embedding(current_token.squeeze(1))
                state_flat, state = self.core(embed_t, state)
                logits = self.output_proj(state_flat)
                
                # Temperature scaling
                logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token)
                current_token = next_token
        
        return torch.cat(generated, dim=1)


# =============================================================================
# TOKENIZER
# =============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self):
        self.char_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_char = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        
        # Add printable ASCII
        for i in range(32, 127):
            char = chr(i)
            idx = len(self.char_to_id)
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_id)
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
    
    def encode(self, text: str) -> List[int]:
        return [self.bos_id] + [self.char_to_id.get(c, 1) for c in text] + [self.eos_id]
    
    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids 
                      if i not in [0, 2, 3])


# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, texts: List[str], tokenizer: CharTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        for text in texts:
            ids = tokenizer.encode(text)
            # Create overlapping windows
            for i in range(0, len(ids) - 1, max_len // 2):
                chunk = ids[i:i + max_len + 1]
                if len(chunk) > 2:
                    self.samples.append(chunk)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Pad if needed
        if len(ids) < self.max_len + 1:
            ids = ids + [self.tokenizer.pad_id] * (self.max_len + 1 - len(ids))
        
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        
        return input_ids, target_ids


# =============================================================================
# TRAINING
# =============================================================================

def generate_corpus(n_samples: int = 100) -> List[str]:
    """Generate training corpus from teacher LLM."""
    client = OpenAI()
    
    prompts = [
        "The quick brown fox", "Once upon a time", "In the beginning",
        "Hello, my name is", "The weather today", "I think that",
        "Science shows that", "The most important", "When I was young",
        "In conclusion", "Technology has changed", "Nature is beautiful",
        "Learning to code", "Music helps people", "Books are important",
        "The history of", "A simple recipe", "The meaning of life",
        "Robots will help", "Ancient civilizations", "The future holds",
        "Mathematics is", "Art expresses", "Philosophy asks",
        "The universe is", "Human nature", "Society needs",
        "Education should", "Health requires", "Happiness comes from"
    ]
    
    corpus = []
    print(f"Generating {n_samples} training samples...")
    
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "Write a short, natural continuation. Keep it under 150 characters."},
                    {"role": "user", "content": f"Continue: {prompt}"}
                ],
                max_tokens=100,
                temperature=0.9
            )
            text = prompt + " " + response.choices[0].message.content
            corpus.append(text[:200])  # Truncate
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{n_samples}")
        except Exception as e:
            corpus.append(prompt)
    
    return corpus


def train_model(
    model: ThymeLM,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu')
) -> List[float]:
    """Train the Thyme model."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(input_ids)
            
            # Cross-entropy loss (ignore padding)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1),
                ignore_index=0  # Ignore PAD
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Generate sample
        model.eval()
        sample_prompt = "The "
        sample_ids = torch.tensor([train_loader.dataset.tokenizer.encode(sample_prompt)[:-1]], device=device)
        generated = model.generate(sample_ids, max_new_tokens=40, temperature=0.7)
        sample_text = train_loader.dataset.tokenizer.decode(generated[0].tolist())
        print(f"  Sample: {sample_text}")
    
    return losses


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("THYME PYTORCH TRAINING")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Tokenizer
    tokenizer = CharTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Generate or load corpus
    cache_path = "/home/ubuntu/thyme/corpus_cache.json"
    if os.path.exists(cache_path):
        print("Loading cached corpus...")
        with open(cache_path, 'r') as f:
            corpus = json.load(f)
    else:
        corpus = generate_corpus(n_samples=100)
        with open(cache_path, 'w') as f:
            json.dump(corpus, f)
    
    print(f"Corpus size: {len(corpus)} texts")
    print(f"Sample: {corpus[0][:80]}...")
    
    # Dataset
    dataset = TextDataset(corpus, tokenizer, max_len=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Model
    model = ThymeLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        decay=0.9,
        dropout=0.1
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    
    losses = train_model(
        model,
        train_loader,
        epochs=15,
        lr=2e-3,
        device=device
    )
    
    # Save model
    save_path = "/home/ubuntu/thyme/checkpoints/thyme_torch.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 128,
        'losses': losses
    }, save_path)
    print(f"\nModel saved to {save_path}")
    
    # Final generation
    print("\n" + "=" * 70)
    print("FINAL GENERATION SAMPLES")
    print("=" * 70)
    
    model.eval()
    test_prompts = ["The ", "Once ", "I think ", "Hello ", "In the "]
    
    for prompt in test_prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)
        generated = model.generate(prompt_ids, max_new_tokens=60, temperature=0.6)
        text = tokenizer.decode(generated[0].tolist())
        print(f"'{prompt}' → {text}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
