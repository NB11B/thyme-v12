"""
Train Thyme on Book Corpus
===========================
Real literature training with 10K vocab.
Cross-platform compatible (Linux/Windows/Mac).
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
import os
import glob
import re
import time
import tempfile

# =============================================================================
# CONSTANTS
# =============================================================================

N_AXIOMS = 12
N_COMPOSITES = 24
N_STATE = 576
VOCAB_SIZE = 10000
EMBED_DIM = 256
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_books(book_dir: str) -> str:
    """Load and clean all books."""
    texts = []
    
    for path in glob.glob(os.path.join(book_dir, "*.txt")):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Remove Gutenberg header/footer
        start_markers = ["*** START OF", "***START OF"]
        end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]
        
        for marker in start_markers:
            if marker in text:
                text = text.split(marker, 1)[-1]
                break
        
        for marker in end_markers:
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        
        # Clean
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        texts.append(text.strip())
        print(f"  Loaded: {os.path.basename(path)} ({len(text):,} chars)")
    
    return '\n\n'.join(texts)


def create_chunks(text: str, chunk_size: int = 500) -> list:
    """Split text into chunks for training."""
    paragraphs = text.split('\n\n')
    
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current) + len(para) < chunk_size:
            current += " " + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    
    if current:
        chunks.append(current)
    
    return chunks


# =============================================================================
# TOKENIZER
# =============================================================================

class BookTokenizer:
    def __init__(self, vocab_size=10000):
        self.target_vocab_size = vocab_size
        self.tokenizer = None
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
    
    def train(self, texts):
        # Use cross-platform temp directory
        temp_path = os.path.join(tempfile.gettempdir(), "books_train.txt")
        with open(temp_path, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(t + "\n")
        
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            min_frequency=2
        )
        self.tokenizer.train([temp_path], trainer)
        
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode(self, text):
        return [self.bos_id] + self.tokenizer.encode(text).ids + [self.eos_id]
    
    def decode(self, ids):
        filtered = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]
        return self.tokenizer.decode(filtered)
    
    def save(self, path):
        self.tokenizer.save(path)


# =============================================================================
# DATASET
# =============================================================================

class BookDataset(Dataset):
    def __init__(self, chunks, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        for chunk in chunks:
            ids = tokenizer.encode(chunk)
            for i in range(0, max(1, len(ids) - max_len), max_len // 2):
                window = ids[i:i + max_len + 1]
                if len(window) > 10:
                    self.samples.append(window)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ids = self.samples[idx]
        if len(ids) < self.max_len + 1:
            ids = ids + [self.tokenizer.pad_id] * (self.max_len + 1 - len(ids))
        else:
            ids = ids[:self.max_len + 1]
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)


# =============================================================================
# THYME MODEL
# =============================================================================

class ThymeLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 12 Axiom decomposition
        self.axiom = nn.Sequential(
            nn.Linear(embed_dim, N_AXIOMS),
            nn.LayerNorm(N_AXIOMS),
            nn.Tanh()
        )
        
        # 24 Composite expansion
        self.composite = nn.Sequential(
            nn.Linear(N_AXIOMS, N_COMPOSITES),
            nn.LayerNorm(N_COMPOSITES),
            nn.Tanh()
        )
        
        # State dynamics
        self.decay = nn.Parameter(torch.tensor(0.9))
        self.mix = nn.Parameter(torch.tensor(0.1))
        
        # Output projection
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
    
    def init_state(self, B, device):
        return torch.zeros(B, N_COMPOSITES, N_COMPOSITES, device=device)
    
    def forward(self, input_ids, state=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        if state is None:
            state = self.init_state(B, device)
        
        embeds = self.dropout(self.embedding(input_ids))
        decay = torch.sigmoid(self.decay)
        mix = torch.sigmoid(self.mix)
        
        all_logits = []
        for t in range(T):
            axioms = self.axiom(embeds[:, t, :])
            composites = self.composite(axioms)
            new = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
            state = decay * state + mix * new
            logits = self.output(state.view(B, -1))
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1), state
    
    def generate(self, prompt_ids, max_new=50, temp=0.8, top_k=40):
        self.eval()
        device = prompt_ids.device
        B = prompt_ids.size(0)
        state = self.init_state(B, device)
        decay = torch.sigmoid(self.decay)
        mix = torch.sigmoid(self.mix)
        
        with torch.no_grad():
            embeds = self.embedding(prompt_ids)
            for t in range(prompt_ids.size(1)):
                axioms = self.axiom(embeds[:, t, :])
                composites = self.composite(axioms)
                new = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
                state = decay * state + mix * new
            
            generated = [prompt_ids]
            last = prompt_ids[:, -1:]
            
            for _ in range(max_new):
                e = self.embedding(last.squeeze(1))
                axioms = self.axiom(e)
                composites = self.composite(axioms)
                new = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
                state = decay * state + mix * new
                
                logits = self.output(state.view(B, -1)) / temp
                
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                generated.append(next_tok)
                last = next_tok
        
        return torch.cat(generated, dim=1)


# =============================================================================
# TRAINING
# =============================================================================

def train(model, loader, val_loader, epochs, lr, device, save_dir):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, 'thyme_books_best.pt')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, n = 0, 0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad()
            logits, _ = model(inp)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), tgt.view(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n += 1
        
        sched.step()
        train_loss /= n
        
        # Validate
        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits, _ = model(inp)
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), tgt.view(-1), ignore_index=0)
                val_loss += loss.item()
                vn += 1
        val_loss /= vn
        
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Train: {train_loss:.3f} (PPL {train_ppl:.1f}) | Val: {val_loss:.3f} (PPL {val_ppl:.1f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
    
    return best_val_loss, best_model_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("THYME TRAINING ON BOOK CORPUS")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup directories (relative to script location)
    book_dir = os.path.join(SCRIPT_DIR, "books")
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load books
    print(f"\nLoading books from: {book_dir}")
    text = load_books(book_dir)
    print(f"Total: {len(text):,} characters")
    
    if len(text) == 0:
        print("\nERROR: No books found!")
        print(f"Please add .txt files to: {book_dir}")
        exit(1)
    
    # Create chunks
    print("\nCreating chunks...")
    chunks = create_chunks(text, chunk_size=500)
    print(f"Chunks: {len(chunks):,}")
    
    # Split train/val
    np.random.seed(42)
    np.random.shuffle(chunks)
    split = int(len(chunks) * 0.9)
    train_chunks = chunks[:split]
    val_chunks = chunks[split:]
    print(f"Train: {len(train_chunks):,} | Val: {len(val_chunks):,}")
    
    # Train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = BookTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(chunks)
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Test
    test = "It is a truth universally acknowledged"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)
    print(f"Test: '{test}' → {len(encoded)} tokens → '{decoded}'")
    
    # Datasets
    print("\nCreating datasets...")
    train_ds = BookDataset(train_chunks, tokenizer, max_len=MAX_LEN)
    val_ds = BookDataset(val_chunks, tokenizer, max_len=MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples: {len(val_ds):,}")
    
    # Model
    print("\nCreating model...")
    model = ThymeLM(vocab_size=tokenizer.vocab_size, embed_dim=EMBED_DIM, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"State size: {N_STATE * 4:,} bytes (constant)")
    
    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    
    start = time.time()
    best_loss, best_model_path = train(
        model, train_loader, val_loader, 
        epochs=EPOCHS, lr=2e-3, device=device, 
        save_dir=checkpoint_dir
    )
    elapsed = time.time() - start
    
    print(f"\nTraining time: {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.3f} (PPL {np.exp(best_loss):.1f})")
    
    # Load best and generate
    print("\n" + "-" * 70)
    print("GENERATION SAMPLES")
    print("-" * 70)
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    prompts = [
        "It is a truth universally acknowledged",
        "Call me Ishmael",
        "It was the best of times",
        "The night was dark",
        "She looked at him"
    ]
    
    for prompt in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)
        gen = model.generate(prompt_ids, max_new=40, temp=0.7)
        text = tokenizer.decode(gen[0].tolist())
        print(f"\n'{prompt}'")
        print(f"  → {text[:100]}...")
    
    # Save tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer_books.json')
    tokenizer.save(tokenizer_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model: {best_model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"State size: 2,304 bytes (constant, regardless of context)")
