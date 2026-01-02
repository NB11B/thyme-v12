#!/usr/bin/env python3
"""
Thyme LLM Corpus Preparation
============================
Downloads and prepares a comprehensive training corpus aligned with the 7/5/2 axiom structure.

Corpus Strategy:
- Project Gutenberg (14.4 GB) - Long-form narrative, diverse genres
- Wikipedia (6.4 GB) - Encyclopedic knowledge across all domains
- Optional: OpenWebText, ArXiv subsets

The corpus is designed to cover all 7 Content axioms:
1. Entity (concrete/abstract) - Wikipedia articles
2. Animacy (living/non-living) - Biology, nature texts
3. Valence (positive/negative) - Literature with emotional range
4. Sociality (individual/collective) - Social texts, dialogues
5. Modality (physical/mental) - Philosophy, psychology
6. Scale (small/large) - Scientific texts
7. Openness (bounded/unbounded) - Mathematical, logical texts
"""

import os
import sys
import argparse
from pathlib import Path

def check_dependencies():
    """Check and install required packages."""
    required = ['datasets', 'tqdm', 'tokenizers']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing missing packages: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("Packages installed. Please re-run the script.")
        sys.exit(0)

check_dependencies()

from datasets import load_dataset
from tqdm import tqdm
import re

# Configuration
SCRIPT_DIR = Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpus"
BOOKS_DIR = CORPUS_DIR / "books"
WIKI_DIR = CORPUS_DIR / "wikipedia"

def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg headers/footers and clean text."""
    # Remove header
    start_markers = ["*** START OF", "***START OF", "*** START OF THE PROJECT", 
                     "*END*THE SMALL PRINT", "End of the Project Gutenberg"]
    for marker in start_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                text = parts[1]
                # Skip to next paragraph
                if '\n\n' in text:
                    text = text.split('\n\n', 1)[1]
                break
    
    # Remove footer
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg", 
                   "End of the Project Gutenberg"]
    for marker in end_markers:
        if marker in text:
            text = text.split(marker)[0]
            break
    
    # Clean whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def download_gutenberg(max_books: int = None, language: str = "en"):
    """Download Project Gutenberg books from HuggingFace."""
    print("\n" + "="*70)
    print("DOWNLOADING PROJECT GUTENBERG")
    print("="*70)
    
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from HuggingFace (language: {language})...")
    
    try:
        # Load with streaming for memory efficiency
        ds = load_dataset("manu/project_gutenberg", split=language, streaming=True)
        
        total_chars = 0
        book_count = 0
        
        for item in tqdm(ds, desc="Processing books"):
            if max_books and book_count >= max_books:
                break
            
            book_id = item.get('id', f'book_{book_count}')
            text = item.get('text', '')
            
            if len(text) < 10000:  # Skip very short texts
                continue
            
            # Clean the text
            cleaned = clean_gutenberg_text(text)
            
            if len(cleaned) < 5000:  # Skip if too short after cleaning
                continue
            
            # Save to file
            output_path = BOOKS_DIR / f"{book_id}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            total_chars += len(cleaned)
            book_count += 1
            
            if book_count % 1000 == 0:
                print(f"  Processed {book_count} books, {total_chars/1e9:.2f} GB")
        
        print(f"\nCompleted: {book_count} books, {total_chars/1e9:.2f} GB total")
        return book_count, total_chars
        
    except Exception as e:
        print(f"Error downloading Gutenberg: {e}")
        print("Trying alternative method...")
        return download_gutenberg_alternative(max_books)

def download_gutenberg_alternative(max_books: int = None):
    """Alternative: Download from pgcorpus/gutenberg GitHub."""
    print("Using pgcorpus/gutenberg method...")
    
    # This would clone and process the standardized corpus
    # For now, return placeholder
    print("Alternative download not implemented - use HuggingFace method")
    return 0, 0

def download_wikipedia(max_articles: int = None, language: str = "en"):
    """Download Wikipedia articles from HuggingFace."""
    print("\n" + "="*70)
    print("DOWNLOADING WIKIPEDIA")
    print("="*70)
    
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Wikipedia dataset (language: {language})...")
    
    try:
        # Wikipedia dataset from HuggingFace
        ds = load_dataset("wikipedia", f"20220301.{language}", streaming=True, split="train")
        
        total_chars = 0
        article_count = 0
        
        # Combine articles into larger files for efficiency
        batch_size = 100
        batch_texts = []
        batch_num = 0
        
        for item in tqdm(ds, desc="Processing articles"):
            if max_articles and article_count >= max_articles:
                break
            
            text = item.get('text', '')
            title = item.get('title', '')
            
            if len(text) < 500:  # Skip stubs
                continue
            
            # Format article
            formatted = f"# {title}\n\n{text}\n\n"
            batch_texts.append(formatted)
            total_chars += len(formatted)
            article_count += 1
            
            # Save batch
            if len(batch_texts) >= batch_size:
                output_path = WIKI_DIR / f"wiki_batch_{batch_num:05d}.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n---\n'.join(batch_texts))
                batch_texts = []
                batch_num += 1
            
            if article_count % 10000 == 0:
                print(f"  Processed {article_count} articles, {total_chars/1e9:.2f} GB")
        
        # Save remaining
        if batch_texts:
            output_path = WIKI_DIR / f"wiki_batch_{batch_num:05d}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n---\n'.join(batch_texts))
        
        print(f"\nCompleted: {article_count} articles, {total_chars/1e9:.2f} GB total")
        return article_count, total_chars
        
    except Exception as e:
        print(f"Error downloading Wikipedia: {e}")
        return 0, 0

def create_corpus_stats():
    """Generate statistics about the downloaded corpus."""
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)
    
    stats = {
        'books': {'count': 0, 'chars': 0, 'files': []},
        'wikipedia': {'count': 0, 'chars': 0, 'files': []}
    }
    
    # Count books
    if BOOKS_DIR.exists():
        for f in BOOKS_DIR.glob("*.txt"):
            stats['books']['files'].append(f)
            stats['books']['count'] += 1
            stats['books']['chars'] += f.stat().st_size
    
    # Count Wikipedia
    if WIKI_DIR.exists():
        for f in WIKI_DIR.glob("*.txt"):
            stats['wikipedia']['files'].append(f)
            stats['wikipedia']['count'] += 1
            stats['wikipedia']['chars'] += f.stat().st_size
    
    print(f"\nBooks:")
    print(f"  Files: {stats['books']['count']}")
    print(f"  Size: {stats['books']['chars']/1e9:.2f} GB")
    
    print(f"\nWikipedia:")
    print(f"  Batches: {stats['wikipedia']['count']}")
    print(f"  Size: {stats['wikipedia']['chars']/1e9:.2f} GB")
    
    total = stats['books']['chars'] + stats['wikipedia']['chars']
    print(f"\nTotal Corpus: {total/1e9:.2f} GB")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Prepare Thyme training corpus")
    parser.add_argument('--gutenberg', action='store_true', help='Download Project Gutenberg')
    parser.add_argument('--wikipedia', action='store_true', help='Download Wikipedia')
    parser.add_argument('--all', action='store_true', help='Download all sources')
    parser.add_argument('--max-books', type=int, default=None, help='Max books to download')
    parser.add_argument('--max-articles', type=int, default=None, help='Max Wikipedia articles')
    parser.add_argument('--stats', action='store_true', help='Show corpus statistics')
    parser.add_argument('--language', type=str, default='en', help='Language code (default: en)')
    
    args = parser.parse_args()
    
    if args.stats:
        create_corpus_stats()
        return
    
    if not any([args.gutenberg, args.wikipedia, args.all]):
        print("Usage: python prepare_corpus.py [--gutenberg] [--wikipedia] [--all] [--stats]")
        print("\nOptions:")
        print("  --gutenberg     Download Project Gutenberg books")
        print("  --wikipedia     Download Wikipedia articles")
        print("  --all           Download all sources")
        print("  --max-books N   Limit number of books")
        print("  --max-articles N Limit number of articles")
        print("  --stats         Show corpus statistics")
        print("  --language XX   Language code (default: en)")
        return
    
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.gutenberg or args.all:
        download_gutenberg(args.max_books, args.language)
    
    if args.wikipedia or args.all:
        download_wikipedia(args.max_articles, args.language)
    
    create_corpus_stats()
    
    print("\n" + "="*70)
    print("CORPUS PREPARATION COMPLETE")
    print("="*70)
    print(f"Corpus saved to: {CORPUS_DIR}")
    print("\nNext steps:")
    print("  1. Run: python train_corpus.py")
    print("  2. Or use the corpus with train_books.py")

if __name__ == "__main__":
    main()
