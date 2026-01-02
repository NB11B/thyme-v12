"""
Thyme with BPE Tokenization
============================
BPE (Byte-Pair Encoding) dramatically improves training efficiency
by working with subword units instead of characters.

"The quick brown fox" → ["The", " quick", " brown", " fox"] (4 tokens)
vs character: ["T","h","e"," ","q",...] (19 tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from typing import List, Tuple, Optional
import json
import os
from openai import OpenAI

# =============================================================================
# CONSTANTS
# =============================================================================

N_AXIOMS = 12
N_COMPOSITES = 24
N_STATE = 576
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# BPE TOKENIZER
# =============================================================================

class BPETokenizer:
    """BPE tokenizer wrapper."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
    def train(self, texts: List[str]):
        """Train BPE on corpus."""
        # Save texts to temp file
        temp_path = "/tmp/bpe_train.txt"
        with open(temp_path, 'w') as f:
            for text in texts:
                f.write(text + "\n")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Train
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            min_frequency=2
        )
        self.tokenizer.train([temp_path], trainer)
        
        # Update special token IDs
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.unk_id = self.tokenizer.token_to_id("<UNK>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        
        self.actual_vocab_size = self.tokenizer.get_vocab_size()
        print(f"BPE vocab size: {self.actual_vocab_size}")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text).ids
        return [self.bos_id] + ids + [self.eos_id]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        # Filter special tokens
        filtered = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]
        return self.tokenizer.decode(filtered)
    
    def save(self, path: str):
        """Save tokenizer."""
        self.tokenizer.save(path)
        
    def load(self, path: str):
        """Load tokenizer."""
        self.tokenizer = Tokenizer.from_file(path)
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.unk_id = self.tokenizer.token_to_id("<UNK>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.actual_vocab_size = self.tokenizer.get_vocab_size()


# =============================================================================
# THYME MODEL (same architecture, different vocab)
# =============================================================================

class AxiomDecomposer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, N_AXIOMS)
        self.norm = nn.LayerNorm(N_AXIOMS)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(self.linear(x)))


class CompositeExpander(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(N_AXIOMS, N_COMPOSITES)
        self.norm = nn.LayerNorm(N_COMPOSITES)
        
    def forward(self, axioms: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(self.linear(axioms)))


class StateManager(nn.Module):
    def __init__(self, decay: float = 0.9):
        super().__init__()
        self.decay = decay
        self.mix = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, composites: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        new_contrib = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
        return self.decay * prev_state + torch.sigmoid(self.mix) * new_contrib


class ThymeLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.axiom = AxiomDecomposer(embed_dim)
        self.composite = CompositeExpander()
        self.state_mgr = StateManager(0.9)
        
        # Larger output head for BPE vocab
        self.output = nn.Sequential(
            nn.Linear(N_STATE, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, vocab_size)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, N_COMPOSITES, N_COMPOSITES, device=device)
    
    def forward(self, input_ids: torch.Tensor, state: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if state is None:
            state = self.init_state(batch_size, device)
        
        embeds = self.embed_dropout(self.embedding(input_ids))
        
        all_logits = []
        for t in range(seq_len):
            embed_t = embeds[:, t, :]
            axioms = self.axiom(embed_t)
            composites = self.composite(axioms)
            state = self.state_mgr(composites, state)
            state_flat = state.view(batch_size, -1)
            logits_t = self.output(state_flat)
            all_logits.append(logits_t)
        
        return torch.stack(all_logits, dim=1), state
    
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 30, temp: float = 0.8, top_k: int = 40):
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.size(0)
        state = self.init_state(batch_size, device)
        
        with torch.no_grad():
            # Process prompt
            embeds = self.embedding(prompt_ids)
            for t in range(prompt_ids.size(1)):
                axioms = self.axiom(embeds[:, t, :])
                composites = self.composite(axioms)
                state = self.state_mgr(composites, state)
            
            # Generate
            generated = [prompt_ids]
            last_token = prompt_ids[:, -1:]
            
            for _ in range(max_new):
                embed = self.embedding(last_token.squeeze(1))
                axioms = self.axiom(embed)
                composites = self.composite(axioms)
                state = self.state_mgr(composites, state)
                state_flat = state.view(batch_size, -1)
                logits = self.output(state_flat) / temp
                
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token)
                last_token = next_token
        
        return torch.cat(generated, dim=1)


# =============================================================================
# DATASET
# =============================================================================

class BPEDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BPETokenizer, max_len: int = 64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > 2:
                # Truncate or use as-is
                if len(ids) > max_len + 1:
                    ids = ids[:max_len + 1]
                self.samples.append(ids)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Pad
        if len(ids) < self.max_len + 1:
            ids = ids + [self.tokenizer.pad_id] * (self.max_len + 1 - len(ids))
        
        return (
            torch.tensor(ids[:-1], dtype=torch.long),
            torch.tensor(ids[1:], dtype=torch.long)
        )


# =============================================================================
# TRAINING
# =============================================================================

def generate_corpus(n_samples: int = 150) -> List[str]:
    """Generate training corpus."""
    client = OpenAI()
    
    prompts = [
        "The quick brown fox", "Once upon a time", "In the beginning",
        "Hello my name is", "The weather today is", "I think that",
        "Science has shown", "The most important thing", "When I was young",
        "In conclusion we can say", "Technology has changed how", 
        "Nature is beautiful because", "Learning to code helps",
        "Music helps people feel", "Books are important for",
        "The history of humanity", "A simple recipe for success",
        "The meaning of life is", "Robots will help humans",
        "Ancient civilizations built", "The future holds many",
        "Mathematics describes the", "Art expresses human",
        "Philosophy asks important", "The universe contains",
        "Human nature includes both", "Society needs people who",
        "Education should focus on", "Health requires balance",
        "Happiness comes from within", "Love is the foundation"
    ]
    
    corpus = []
    print(f"Generating {n_samples} training samples...")
    
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "Continue naturally in 1-2 sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=80,
                temperature=0.9
            )
            text = prompt + " " + response.choices[0].message.content
            corpus.append(text[:250])
            
            if (i + 1) % 30 == 0:
                print(f"  Generated {i + 1}/{n_samples}")
        except:
            corpus.append(prompt)
    
    return corpus


def train(model, loader, epochs, lr, device):
    """Train model."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            opt.zero_grad()
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1),
                ignore_index=loader.dataset.tokenizer.pad_id
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
            n += 1
        
        sched.step()
        avg_loss = total_loss / n
        losses.append(avg_loss)
        
        # Generate sample
        model.eval()
        prompt = "The "
        prompt_ids = torch.tensor([loader.dataset.tokenizer.encode(prompt)[:-1]], device=device)
        gen = model.generate(prompt_ids, max_new=25, temp=0.7)
        text = loader.dataset.tokenizer.decode(gen[0].tolist())
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Sample: {text[:60]}")
    
    return losses


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("THYME WITH BPE TOKENIZATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Generate corpus
    cache_path = "/home/ubuntu/thyme/corpus_bpe.json"
    if os.path.exists(cache_path):
        print("Loading cached corpus...")
        with open(cache_path, 'r') as f:
            corpus = json.load(f)
    else:
        corpus = generate_corpus(n_samples=150)
        with open(cache_path, 'w') as f:
            json.dump(corpus, f)
    
    print(f"Corpus: {len(corpus)} texts")
    
    # Train BPE tokenizer
    print("\nTraining BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(corpus)
    
    # Test tokenization
    test = "The quick brown fox jumps"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)
    print(f"Test: '{test}' → {len(encoded)} tokens → '{decoded}'")
    
    # Dataset
    dataset = BPEDataset(corpus, tokenizer, max_len=48)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Dataset: {len(dataset)} samples")
    
    # Model
    model = ThymeLM(
        vocab_size=tokenizer.actual_vocab_size,
        embed_dim=256,
        dropout=0.1
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    
    losses = train(model, loader, epochs=25, lr=3e-3, device=device)
    
    # Save
    save_path = "/home/ubuntu/thyme/checkpoints/thyme_bpe.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'losses': losses,
        'vocab_size': tokenizer.actual_vocab_size
    }, save_path)
    tokenizer.save("/home/ubuntu/thyme/checkpoints/tokenizer.json")
    print(f"\nSaved to {save_path}")
    
    # Final samples
    print("\n" + "=" * 70)
    print("FINAL GENERATION")
    print("=" * 70)
    
    model.eval()
    for prompt in ["The ", "Once upon ", "I think ", "Science ", "The future "]:
        ids = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)
        gen = model.generate(ids, max_new=30, temp=0.6)
        print(f"'{prompt}' → {tokenizer.decode(gen[0].tolist())}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
