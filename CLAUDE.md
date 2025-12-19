# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ickyMCP is a RAG (Retrieval-Augmented Generation) MCP server for semantic document search. It indexes documents into a SQLite vector database and provides semantic search via the MCP protocol.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
# Or as editable package
pip install -e .

# Run the MCP server
python run.py

# Fast parallel indexing (for bulk document ingestion)
python fast_index.py <target_dir> --workers 8 --batch 64 --patterns "*.pdf" "*.docx"

# Run tests
pytest
pytest tests/test_specific.py -k "test_name"
```

## Architecture

### Data Flow
1. **Parsing** (`src/parsers.py`): Documents (PDF, DOCX, PPTX, XLSX, TXT, MD) are parsed to extract text
2. **Chunking** (`src/chunker.py`): Text is split into 4K token chunks with 500 token overlap using tiktoken
3. **Embedding** (`src/embedder.py`): Chunks are embedded using either local model (nomic-embed-text-v1.5) or Voyage AI API
4. **Storage** (`src/database.py`): Embeddings stored in SQLite with sqlite-vec extension for vector similarity search
5. **Search**: Query is embedded and compared using cosine similarity

### Key Components

- **`src/server.py`**: MCP server entry point. Exposes 7 tools: `index`, `search`, `similar`, `refresh`, `list`, `delete`, `status`
- **`src/embedder.py`**: Dual embedding support - `LocalEmbedder` (sentence-transformers) and `VoyageEmbedder` (API). Selected via `EMBEDDING_PROVIDER` env var
- **`src/database.py`**: `VectorDatabase` class wraps SQLite + sqlite-vec. Three tables: `documents`, `chunks`, `chunk_embeddings` (virtual table)
- **`src/chunker.py`**: `TextChunker` uses tiktoken for accurate token counting. Finds smart break points (paragraph > sentence > word)
- **`fast_index.py`**: Standalone CLI for bulk indexing with thread-pool parsing and batched embeddings

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ICKY_EMBEDDING_PROVIDER` | `voyage` | `local` or `voyage` |
| `ICKY_CHUNK_SIZE` | `4000` | Tokens per chunk |
| `ICKY_CHUNK_OVERLAP` | `500` | Overlap between chunks |
| `ICKY_DB_PATH` | `./icky_voyage.db` or `./icky.db` | SQLite database path |
| `ICKY_VOYAGE_API_KEY` | (set in config) | Voyage AI API key |
| `ICKY_VOYAGE_MODEL` | `voyage-3.5-lite` | Voyage model name |
| `ICKY_VOYAGE_DIMENSIONS` | `1024` | Voyage embedding dimensions |
| `ICKY_LOCAL_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Local model name |

### Database Schema

- `documents`: path (unique), file_type, file_size, modified_time, indexed_at, chunk_count, page_count
- `chunks`: document_id (FK), chunk_index, chunk_text, token_count, page_number, start_char, end_char
- `chunk_embeddings`: Virtual table using sqlite-vec for vector similarity (768 or 1024 dimensions)

### Embedding Prefixes (Local Model Only)

The nomic model requires prefixes:
- Query: `search_query: <text>`
- Document: `search_document: <text>`

These are applied automatically in `LocalEmbedder.embed_query()` and `embed_documents()`.
