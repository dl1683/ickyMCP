#!/usr/bin/env python3
"""Simple chunking test."""
import time
from pathlib import Path
from src.parsers import parse_document

# Parse the small PDF
print("Parsing RL.pdf...")
doc = parse_document(Path('docs/RL.pdf'))
print(f"Parsed: {len(doc.text)} chars")

# Simple chunking - FIXED
print("\nChunking...")
start = time.time()

chunk_size = 20000  # ~5K tokens
overlap = 2000      # ~500 tokens
chunks = []

i = 0
while i < len(doc.text):
    end = min(i + chunk_size, len(doc.text))
    chunks.append(doc.text[i:end])

    # If we hit the end, break
    if end >= len(doc.text):
        break

    # Otherwise advance with overlap
    i = end - overlap

print(f"Chunked in {time.time() - start:.4f}s")
print(f"Chunks: {len(chunks)}")
