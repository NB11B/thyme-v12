#!/usr/bin/env python3
"""
Thyme LLM Training on Large Corpus
==================================
Enhanced training script for comprehensive text corpus.
Supports multi-GPU, gradient accumulation, and larger models.

Key improvements:
- Larger vocabulary (32K BPE tokens)
- Deeper axiom processing
- Gradient accumulation for effective larger batches
- Learning rate warmup
- COMPREHENSIVE LOGGING AND PROGRESS TRACKING
- Periodic checkpointing (not just at epoch end)
- Batch-level progress with ETA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
import os
import glob
import re
import time
import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path, log_name: str = "training"):
    """Setup comprehensive logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('thyme')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Also create a simple progress file that can be tailed
    progress_path = log_dir / "progress.log"
    progress_handler = logging.FileHandler(progress_path, mode='w')
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.addHandler(progress_handler)
    
    return logger, progress_logger, log_path, progress_path

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Architecture (Thyme 7/5/2)
    N_AXIOMS = 12          # 7 content + 5 relational
    N_COMPOSITES = 24      # 12 * 2 (paired interactions)
    N_STATE = 576          # 24 * 24 (full state matrix)
    
    # Model size
    VOCAB_SIZE = 32000     # Larger vocabulary for diverse corpus
    EMBED_DIM = 512        # Increased embedding dimension
    HIDDEN_DIM = 1024      # Hidden layer size
    N_LAYERS = 2           # Number of axiom processing layers
    DROPOUT = 0.1
    
    # Training
    MAX_LEN = 256          # Longer sequences
    BATCH_SIZE = 16        # Per-GPU batch size
    GRAD_ACCUM = 4         # Gradient accumulation steps
    EPOCHS = 20
    LR = 1e-3
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.01
    
    # Logging & Checkpointing
    LOG_EVERY_N_BATCHES = 100      # Log progress every N batches
    CHECKPOINT_EVERY_N_BATCHES = 5000  # Save checkpoint every N batches
    SAVE_HISTORY_EVERY_N_BATCHES = 1000  # Update history file every N batches
    
    # Paths
    SCRIPT_DIR = Path(__file__).parent
    CORPUS_DIR = SCRIPT_DIR / "corpus"
    CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
    LOG_DIR = SCRIPT_DIR / "logs"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_time(seconds):
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def estimate_vram_usage(batch_size, seq_len, vocab_size, embed_dim, n_gpus=1):
    """Estimate VRAM usage in GB."""
    # Rough estimation
    params_memory = (vocab_size * embed_dim + embed_dim * 1024 + 1024 * vocab_size) * 4 / 1e9
    activation_memory = batch_size * seq_len * embed_dim * 4 * 10 / 1e9  # ~10x for gradients
    total = (params_memory + activation_memory) * n_gpus
    return total

def suggest_batch_size(available_vram_gb, seq_len=256, vocab_size=32000, embed_dim=512):
    """Suggest optimal batch size based on available VRAM."""
    for bs in [128, 64, 32, 16, 8]:
        estimated = estimate_vram_usage(bs, seq_len, vocab_size, embed_dim)
        if estimated < available_vram_gb * 0.8:  # Leave 20% headroom
            return bs
    return 8

# =============================================================================
# DATA LOADING
# =============================================================================

def load_corpus(corpus_dir: Path, max_files: int = None, logger=None) -> str:
    """Load all text files from corpus directory."""
    log = logger.info if logger else print
    
    texts = []
    total_chars = 0
    
    # Load books
    books_dir = corpus_dir / "books"
    if books_dir.exists():
        book_files = sorted(books_dir.glob("*.txt"))
        if max_files:
            book_files = book_files[:max_files]
        
        for path in tqdm(book_files, desc="Loading books", disable=logger is None):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            texts.append(text)
            total_chars += len(text)
        log(f"Loaded {len(book_files)} books ({total_chars/1e6:.1f}M chars)")
    
    # Load Wikipedia
    wiki_dir = corpus_dir / "wikipedia"
    if wiki_dir.exists():
        wiki_files = sorted(wiki_dir.glob("*.txt"))
        wiki_chars = 0
        for path in tqdm(wiki_files, desc="Loading wiki", disable=logger is None):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            texts.append(text)
            wiki_chars += len(text)
        total_chars += wiki_chars
        log(f"Loaded {len(wiki_files)} wiki batches ({wiki_chars/1e6:.1f}M chars)")
    
    # Fallback to books directory if corpus not prepared
    if not texts:
        books_dir = corpus_dir.parent / "books"
        if books_dir.exists():
            for path in sorted(books_dir.glob("*.txt")):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                texts.append(text)
                total_chars += len(text)
            log(f"Loaded {len(texts)} books from fallback ({total_chars/1e6:.1f}M chars)")
    
    log(f"Total corpus: {total_chars/1e9:.2f} GB")
    return '\n\n'.join(texts)

def create_chunks(text: str, chunk_size: int = 1000) -> list:
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
            if current and len(current) > 100:
                chunks.append(current)
            current = para
    
    if current and len(current) > 100:
        chunks.append(current)
    
    return chunks

# =============================================================================
# TOKENIZER
# =============================================================================

class CorpusTokenizer:
    def __init__(self, vocab_size=32000):
        self.target_vocab_size = vocab_size
        self.tokenizer = None
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
    
    def train(self, texts, save_path=None):
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), "corpus_train.txt")
        
        # Sample if too large
        if len(texts) > 100000:
            indices = np.random.choice(len(texts), 100000, replace=False)
            sample = [texts[i] for i in indices]
        else:
            sample = texts
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            for t in sample:
                f.write(t + "\n")
        
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            min_frequency=2,
            show_progress=True
        )
        self.tokenizer.train([temp_path], trainer)
        
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        if save_path:
            self.tokenizer.save(str(save_path))
        
        return self
    
    def encode(self, text):
        return [self.bos_id] + self.tokenizer.encode(text).ids + [self.eos_id]
    
    def decode(self, ids):
        filtered = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]
        return self.tokenizer.decode(filtered)
    
    def save(self, path):
        self.tokenizer.save(str(path))
    
    @classmethod
    def load(cls, path):
        obj = cls()
        obj.tokenizer = Tokenizer.from_file(str(path))
        obj.pad_id = obj.tokenizer.token_to_id("<PAD>")
        obj.bos_id = obj.tokenizer.token_to_id("<BOS>")
        obj.eos_id = obj.tokenizer.token_to_id("<EOS>")
        obj.vocab_size = obj.tokenizer.get_vocab_size()
        return obj

# =============================================================================
# DATASET
# =============================================================================

class CorpusDataset(Dataset):
    def __init__(self, chunks, tokenizer, max_len=256, logger=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        log = logger.info if logger else print
        log("Tokenizing chunks...")
        
        for chunk in tqdm(chunks, desc="Tokenizing", disable=logger is None):
            ids = tokenizer.encode(chunk)
            # Create overlapping windows
            stride = max_len // 2
            for i in range(0, max(1, len(ids) - max_len), stride):
                window = ids[i:i + max_len + 1]
                if len(window) > 10:
                    self.samples.append(window)
        
        log(f"Created {len(self.samples):,} training samples")
    
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
# THYME MODEL (Enhanced)
# =============================================================================

class ThymeLayer(nn.Module):
    """Single Thyme processing layer."""
    def __init__(self, embed_dim, n_axioms=12, n_composites=24, dropout=0.1):
        super().__init__()
        
        # Axiom decomposition
        self.axiom = nn.Sequential(
            nn.Linear(embed_dim, n_axioms * 2),
            nn.LayerNorm(n_axioms * 2),
            nn.GELU(),
            nn.Linear(n_axioms * 2, n_axioms),
            nn.Tanh()
        )
        
        # Composite expansion
        self.composite = nn.Sequential(
            nn.Linear(n_axioms, n_composites),
            nn.LayerNorm(n_composites),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        axioms = self.axiom(x)
        composites = self.composite(axioms)
        return self.dropout(composites)

class ThymeLM(nn.Module):
    """Enhanced Thyme Language Model."""
    def __init__(self, vocab_size, embed_dim=512, n_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(Config.MAX_LEN, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Thyme layers
        self.layers = nn.ModuleList([
            ThymeLayer(embed_dim, Config.N_AXIOMS, Config.N_COMPOSITES, dropout)
            for _ in range(n_layers)
        ])
        
        # State dynamics
        self.decay = nn.Parameter(torch.tensor(0.9))
        self.mix = nn.Parameter(torch.tensor(0.1))
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(Config.N_STATE, Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(Config.HIDDEN_DIM, vocab_size)
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
        return torch.zeros(B, Config.N_COMPOSITES, Config.N_COMPOSITES, device=device)
    
    def forward(self, input_ids, state=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        if state is None:
            state = self.init_state(B, device)
        
        # Embeddings with position
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        embeds = self.dropout(self.embedding(input_ids) + self.pos_embedding(positions))
        
        decay = torch.sigmoid(self.decay)
        mix = torch.sigmoid(self.mix)
        
        all_logits = []
        for t in range(T):
            x = embeds[:, t, :]
            
            # Process through Thyme layers
            for layer in self.layers:
                composites = layer(x)
            
            # State update
            new = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
            state = decay * state + mix * new
            
            # Output
            logits = self.output(state.view(B, -1))
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1), state
    
    def generate(self, prompt_ids, max_new=100, temp=0.8, top_k=50, top_p=0.9):
        self.eval()
        device = prompt_ids.device
        B = prompt_ids.size(0)
        state = self.init_state(B, device)
        decay = torch.sigmoid(self.decay)
        mix = torch.sigmoid(self.mix)
        
        with torch.no_grad():
            # Process prompt
            positions = torch.arange(prompt_ids.size(1), device=device).unsqueeze(0)
            embeds = self.embedding(prompt_ids) + self.pos_embedding(positions)
            
            for t in range(prompt_ids.size(1)):
                x = embeds[:, t, :]
                for layer in self.layers:
                    composites = layer(x)
                new = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
                state = decay * state + mix * new
            
            # Generate
            generated = [prompt_ids]
            last = prompt_ids[:, -1:]
            pos = prompt_ids.size(1)
            
            for _ in range(max_new):
                pos_emb = self.pos_embedding(torch.tensor([[min(pos, Config.MAX_LEN-1)]], device=device))
                e = self.embedding(last.squeeze(1)) + pos_emb.squeeze(1)
                
                for layer in self.layers:
                    composites = layer(e)
                new = torch.bmm(composites.unsqueeze(2), composites.unsqueeze(1))
                state = decay * state + mix * new
                
                logits = self.output(state.view(B, -1)) / temp
                
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                generated.append(next_tok)
                last = next_tok
                pos += 1
        
        return torch.cat(generated, dim=1)

# =============================================================================
# TRAINING WITH COMPREHENSIVE LOGGING
# =============================================================================

def get_lr(step, warmup_steps, max_lr, total_steps):
    """Learning rate schedule with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr * 0.5 * (1 + np.cos(np.pi * progress))

def train(model, train_loader, val_loader, config, device, save_dir, logger, progress_logger):
    """Training loop with comprehensive logging and periodic checkpointing."""
    model = model.to(device)
    
    # Multi-GPU
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = GradScaler()
    
    total_batches = len(train_loader)
    total_steps = total_batches * config.EPOCHS // config.GRAD_ACCUM
    
    logger.info(f"Total batches per epoch: {total_batches:,}")
    logger.info(f"Total optimizer steps: {total_steps:,}")
    logger.info(f"Effective batch size: {config.BATCH_SIZE * config.GRAD_ACCUM * max(1, n_gpus)}")
    
    best_val_loss = float('inf')
    best_model_path = save_dir / 'thyme_corpus_best.pt'
    latest_model_path = save_dir / 'thyme_corpus_latest.pt'
    history = {
        'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': [],
        'batch_losses': [], 'learning_rates': [], 'timestamps': []
    }
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        batch_losses = []
        
        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch+1}/{config.EPOCHS}")
        logger.info(f"{'='*70}")
        
        pbar = tqdm(enumerate(train_loader), total=total_batches, 
                    desc=f"Epoch {epoch+1}", leave=True)
        
        for batch_idx, (inp, tgt) in pbar:
            inp, tgt = inp.to(device), tgt.to(device)
            
            # Update learning rate
            lr = get_lr(global_step, config.WARMUP_STEPS, config.LR, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward with mixed precision
            with autocast():
                logits, _ = model(inp)
                base_model = model.module if hasattr(model, 'module') else model
                loss = F.cross_entropy(
                    logits.view(-1, base_model.vocab_size), 
                    tgt.view(-1), 
                    ignore_index=0
                )
                loss = loss / config.GRAD_ACCUM
            
            # Backward
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            batch_loss = loss.item() * config.GRAD_ACCUM
            train_loss += batch_loss
            batch_losses.append(batch_loss)
            n_batches += 1
            
            # Update progress bar
            avg_loss = train_loss / n_batches
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'ppl': f'{np.exp(avg_loss):.1f}',
                'lr': f'{lr:.2e}'
            })
            
            # Periodic logging
            if (batch_idx + 1) % config.LOG_EVERY_N_BATCHES == 0:
                elapsed = time.time() - start_time
                batches_done = epoch * total_batches + batch_idx + 1
                batches_total = config.EPOCHS * total_batches
                eta = elapsed / batches_done * (batches_total - batches_done)
                
                progress_msg = (
                    f"E{epoch+1} B{batch_idx+1}/{total_batches} | "
                    f"Loss: {avg_loss:.3f} | PPL: {np.exp(avg_loss):.1f} | "
                    f"LR: {lr:.2e} | ETA: {format_time(eta)}"
                )
                progress_logger.info(progress_msg)
                
                # Save to history
                history['batch_losses'].append(batch_loss)
                history['learning_rates'].append(lr)
                history['timestamps'].append(time.time() - start_time)
            
            # Periodic checkpoint
            if (batch_idx + 1) % config.CHECKPOINT_EVERY_N_BATCHES == 0:
                logger.info(f"Saving periodic checkpoint at batch {batch_idx+1}...")
                base_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'global_step': global_step
                }, latest_model_path)
                
                # Also save history periodically
                with open(save_dir / 'training_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
            
            # Save history periodically (more frequent than checkpoints)
            if (batch_idx + 1) % config.SAVE_HISTORY_EVERY_N_BATCHES == 0:
                with open(save_dir / 'training_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
        
        train_loss /= n_batches
        epoch_time = time.time() - epoch_start
        
        # Validation
        logger.info(f"Running validation...")
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for inp, tgt in tqdm(val_loader, desc="Validation", leave=False):
                inp, tgt = inp.to(device), tgt.to(device)
                logits, _ = model(inp)
                base_model = model.module if hasattr(model, 'module') else model
                loss = F.cross_entropy(
                    logits.view(-1, base_model.vocab_size), 
                    tgt.view(-1), 
                    ignore_index=0
                )
                val_loss += loss.item()
                vn += 1
        val_loss /= vn
        
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)
        
        # Log epoch summary
        marker = "★ NEW BEST" if val_loss < best_val_loss else ""
        epoch_summary = (
            f"Epoch {epoch+1:2d}/{config.EPOCHS} COMPLETE | "
            f"Train: {train_loss:.3f} (PPL {train_ppl:.1f}) | "
            f"Val: {val_loss:.3f} (PPL {val_ppl:.1f}) | "
            f"Time: {format_time(epoch_time)} | "
            f"LR: {lr:.2e} {marker}"
        )
        logger.info(epoch_summary)
        progress_logger.info(epoch_summary)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            base_model = model.module if hasattr(model, 'module') else model
            torch.save(base_model.state_dict(), best_model_path)
            logger.info(f"Saved new best model to {best_model_path}")
        
        # Save history after each epoch
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal training time: {format_time(total_time)}")
    
    return best_val_loss, best_model_path, history

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Thyme on large corpus")
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LR)
    parser.add_argument('--max-files', type=int, default=None, help='Limit corpus files')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--log-every', type=int, default=Config.LOG_EVERY_N_BATCHES, 
                        help='Log every N batches')
    parser.add_argument('--checkpoint-every', type=int, default=Config.CHECKPOINT_EVERY_N_BATCHES,
                        help='Save checkpoint every N batches')
    args = parser.parse_args()
    
    # Update config
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.LOG_EVERY_N_BATCHES = args.log_every
    Config.CHECKPOINT_EVERY_N_BATCHES = args.checkpoint_every
    
    # Setup logging
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger, progress_logger, log_path, progress_path = setup_logging(Config.LOG_DIR)
    
    logger.info("="*70)
    logger.info("THYME TRAINING ON LARGE CORPUS")
    logger.info("="*70)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Progress file: {progress_path}")
    logger.info(f"  (Monitor with: tail -f {progress_path})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        logger.info(f"GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
        
        # Suggest batch size
        total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)) / 1e9
        suggested_bs = suggest_batch_size(total_vram / n_gpus)
        logger.info(f"  Total VRAM: {total_vram:.1f} GB")
        logger.info(f"  Suggested batch size: {suggested_bs} (current: {Config.BATCH_SIZE})")
    
    # Load corpus
    logger.info("\nLoading corpus...")
    text = load_corpus(Config.CORPUS_DIR, args.max_files, logger)
    
    if len(text) == 0:
        logger.error("\nERROR: No corpus found!")
        logger.error(f"Please run: python prepare_corpus.py --gutenberg")
        logger.error(f"Or add .txt files to: {Config.CORPUS_DIR}")
        return
    
    # Create chunks
    logger.info("\nCreating chunks...")
    chunks = create_chunks(text, chunk_size=1000)
    logger.info(f"Chunks: {len(chunks):,}")
    
    # Split
    np.random.seed(42)
    np.random.shuffle(chunks)
    split = int(len(chunks) * 0.95)
    train_chunks = chunks[:split]
    val_chunks = chunks[split:]
    logger.info(f"Train: {len(train_chunks):,} | Val: {len(val_chunks):,}")
    
    # Tokenizer
    tokenizer_path = Config.CHECKPOINT_DIR / 'tokenizer_corpus.json'
    if tokenizer_path.exists() and not args.resume:
        logger.info("\nLoading existing tokenizer...")
        tokenizer = CorpusTokenizer.load(tokenizer_path)
    else:
        logger.info("\nTraining tokenizer...")
        tokenizer = CorpusTokenizer(vocab_size=Config.VOCAB_SIZE)
        tokenizer.train(chunks, tokenizer_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Datasets
    logger.info("\nCreating datasets...")
    train_ds = CorpusDataset(train_chunks, tokenizer, max_len=Config.MAX_LEN, logger=logger)
    val_ds = CorpusDataset(val_chunks, tokenizer, max_len=Config.MAX_LEN, logger=logger)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
    
    # Model
    logger.info("\nCreating model...")
    model = ThymeLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=Config.EMBED_DIM,
        n_layers=Config.N_LAYERS,
        dropout=Config.DROPOUT
    )
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"  Resumed from epoch {checkpoint.get('epoch', '?')}, batch {checkpoint.get('batch', '?')}")
        else:
            model.load_state_dict(checkpoint)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    logger.info(f"State size: {Config.N_STATE * 4:,} bytes (constant)")
    
    # Training config summary
    logger.info("\n" + "-"*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("-"*70)
    logger.info(f"Epochs: {Config.EPOCHS}")
    logger.info(f"Batch size: {Config.BATCH_SIZE} (per GPU)")
    logger.info(f"Gradient accumulation: {Config.GRAD_ACCUM}")
    logger.info(f"Learning rate: {Config.LR}")
    logger.info(f"Log every: {Config.LOG_EVERY_N_BATCHES} batches")
    logger.info(f"Checkpoint every: {Config.CHECKPOINT_EVERY_N_BATCHES} batches")
    
    # Train
    logger.info("\n" + "-"*70)
    logger.info("TRAINING")
    logger.info("-"*70)
    
    start = time.time()
    best_loss, best_path, history = train(
        model, train_loader, val_loader, Config, device, 
        Config.CHECKPOINT_DIR, logger, progress_logger
    )
    elapsed = time.time() - start
    
    logger.info(f"\nTraining time: {format_time(elapsed)}")
    logger.info(f"Best validation loss: {best_loss:.3f} (PPL {np.exp(best_loss):.1f})")
    
    # Generate samples
    logger.info("\n" + "-"*70)
    logger.info("GENERATION SAMPLES")
    logger.info("-"*70)
    
    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    prompts = [
        "It is a truth universally acknowledged",
        "The universe is",
        "In the beginning",
        "The most important thing",
        "Science has shown that"
    ]
    
    for prompt in prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt)[:-1]], device=device)
        gen = model.generate(prompt_ids, max_new=50, temp=0.7)
        text = tokenizer.decode(gen[0].tolist())
        logger.info(f"\n'{prompt}'")
        logger.info(f"  → {text[:150]}...")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Model: {best_path}")
    logger.info(f"Tokenizer: {tokenizer_path}")
    logger.info(f"State size: {Config.N_STATE * 4:,} bytes (constant)")
    logger.info(f"Log: {log_path}")
    logger.info(f"Progress: {progress_path}")

if __name__ == "__main__":
    main()
