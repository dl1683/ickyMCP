#!/usr/bin/env python3
"""Fast parallel indexer for ickyMCP."""

import sys
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DB_PATH
from src.database import VectorDatabase
from src.parsers import parse_document, is_supported
from src.chunker import TextChunker
from src.embedder import get_embedder


@dataclass
class FileResult:
    path: str
    chunks: list
    page_count: int
    file_type: str
    file_size: int
    modified_time: float
    error: Optional[str] = None


def parse_file(file_path: Path) -> FileResult:
    """Parse a single file - runs in thread pool."""
    try:
        stat = file_path.stat()
        parsed = parse_document(file_path)

        if parsed is None or not parsed.text.strip():
            return FileResult(
                path=str(file_path),
                chunks=[],
                page_count=0,
                file_type="unknown",
                file_size=stat.st_size,
                modified_time=stat.st_mtime,
                error="No content extracted"
            )

        # Chunk the text
        chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = chunker.chunk_text(parsed.text)

        return FileResult(
            path=str(file_path),
            chunks=chunks,
            page_count=parsed.page_count,
            file_type=parsed.file_type,
            file_size=stat.st_size,
            modified_time=stat.st_mtime
        )
    except Exception as e:
        stat = file_path.stat()
        return FileResult(
            path=str(file_path),
            chunks=[],
            page_count=0,
            file_type="unknown",
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            error=str(e)
        )


def fast_index(target_dir: str, patterns: list = None, max_workers: int = 8, batch_size: int = 64):
    """Fast parallel indexing with thread pool for parsing and batched embeddings."""

    if patterns is None:
        patterns = ["*.pdf"]

    target_path = Path(target_dir)

    print(f"=== ickyMCP Fast Indexer ===")
    print(f"Target: {target_dir}")
    print(f"Workers: {max_workers}, Batch size: {batch_size}")
    print()

    # Collect files
    files = []
    for pattern in patterns:
        files.extend(list(target_path.rglob(pattern)))
    files = [f for f in files if f.is_file() and is_supported(f)]

    print(f"Found {len(files)} files to index")
    print()

    # Initialize components
    print("Loading embedding model...")
    start_time = time.time()
    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()
    print(f"Model loaded in {time.time() - start_time:.1f}s")
    print()

    # Clear and initialize database
    db = VectorDatabase(DB_PATH)
    db.connect()
    db.delete_all()

    # Phase 1: Parse all files in parallel
    print(f"Phase 1: Parsing {len(files)} files with {max_workers} workers...")
    parse_start = time.time()

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(parse_file, f): f for f in files}

        completed = 0
        for future in as_completed(future_to_file):
            result = future.result()
            completed += 1

            if result.error:
                errors.append(f"{result.path}: {result.error}")
            elif result.chunks:
                results.append(result)

            if completed % 50 == 0:
                print(f"  Parsed {completed}/{len(files)} files...")

    parse_time = time.time() - parse_start
    print(f"Parsing complete: {len(results)} successful, {len(errors)} errors in {parse_time:.1f}s")
    print()

    # Phase 2: Collect all chunks and embed in batches
    print("Phase 2: Generating embeddings...")
    embed_start = time.time()

    # Flatten all chunks with metadata
    all_chunks = []
    for result in results:
        for chunk in result.chunks:
            all_chunks.append({
                'result': result,
                'chunk': chunk
            })

    print(f"  Total chunks: {len(all_chunks)}")

    # Embed in batches
    all_texts = [item['chunk'].text for item in all_chunks]
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        batch_embeddings = embedder.embed_documents(batch, show_progress=False)
        all_embeddings.extend(batch_embeddings)

        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i+batch_size, len(all_texts))}/{len(all_texts)} chunks...")

    embed_time = time.time() - embed_start
    print(f"Embedding complete in {embed_time:.1f}s")
    print()

    # Phase 3: Bulk insert into database
    print("Phase 3: Storing in database...")
    store_start = time.time()

    # Group chunks by document
    doc_chunks = {}
    for item, embedding in zip(all_chunks, all_embeddings):
        path = item['result'].path
        if path not in doc_chunks:
            doc_chunks[path] = {
                'result': item['result'],
                'chunks_data': []
            }
        doc_chunks[path]['chunks_data'].append({
            'chunk': item['chunk'],
            'embedding': embedding
        })

    # Insert documents and chunks
    for path, data in doc_chunks.items():
        result = data['result']

        doc_id = db.add_document(
            path=path,
            file_type=result.file_type,
            file_size=result.file_size,
            modified_time=result.modified_time,
            page_count=result.page_count
        )

        for chunk_data in data['chunks_data']:
            chunk = chunk_data['chunk']
            embedding = chunk_data['embedding']

            db.add_chunk(
                document_id=doc_id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                token_count=chunk.token_count,
                embedding=embedding,
                page_number=chunk.page_number,
                start_char=chunk.start_char,
                end_char=chunk.end_char
            )

        db.update_document_chunk_count(doc_id, len(data['chunks_data']))

    db.conn.commit()
    store_time = time.time() - store_start
    print(f"Storage complete in {store_time:.1f}s")
    print()

    # Summary
    total_time = time.time() - start_time
    stats = db.get_stats()

    print("=== INDEXING COMPLETE ===")
    print(f"Documents indexed: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Database size: {stats['index_size_mb']:.1f} MB")
    print(f"Errors: {len(errors)}")
    print()
    print(f"Time breakdown:")
    print(f"  Model load: {start_time:.1f}s")
    print(f"  Parsing:    {parse_time:.1f}s")
    print(f"  Embedding:  {embed_time:.1f}s")
    print(f"  Storage:    {store_time:.1f}s")
    print(f"  TOTAL:      {total_time:.1f}s")

    if errors and len(errors) <= 10:
        print()
        print("Errors:")
        for e in errors:
            print(f"  - {e[:100]}")

    db.close()
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast parallel indexer for ickyMCP")
    parser.add_argument("target_dir", help="Directory to index")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--patterns", nargs="+", default=["*.pdf"], help="File patterns to match")

    args = parser.parse_args()

    fast_index(args.target_dir, args.patterns, args.workers, args.batch)
