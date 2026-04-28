# ickyMCP

RAG MCP server for semantic document search. It indexes local document collections into SQLite with `sqlite-vec`, then exposes MCP tools for indexing, searching, refreshing, listing, and deleting indexed documents.

## Features

- Semantic search over PDF, Word, PowerPoint, Excel, Markdown, and text files
- Per-user database isolation with optional legacy single-database mode
- Document ID filters for chat or matter-scoped retrieval
- Incremental indexing based on file size and modified time
- Voyage AI embeddings by default, with an offline local sentence-transformers backend
- Portable SQLite storage with ignored local database artifacts

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create local environment settings from the example:

```bash
copy .env.example .env
```

Set `ICKY_VOYAGE_API_KEY` when using the default Voyage backend. For offline use, set `ICKY_EMBEDDING_PROVIDER=local`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ICKY_EMBEDDING_PROVIDER` | `voyage` | `voyage` or `local` |
| `ICKY_VOYAGE_API_KEY` | unset | Required when using Voyage embeddings |
| `ICKY_VOYAGE_MODEL` | `voyage-3.5-lite` | Voyage embedding model |
| `ICKY_VOYAGE_DIMENSIONS` | `1024` | Voyage output dimension |
| `ICKY_VOYAGE_WORKERS` | `8` | Reserved for parallel Voyage workflows |
| `ICKY_LOCAL_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Local sentence-transformers model |
| `ICKY_CHUNK_SIZE` | `5000` | Approximate tokens per chunk |
| `ICKY_CHUNK_OVERLAP` | `500` | Approximate token overlap between chunks |
| `ICKY_DATA_DIR` | `./data` | Base directory for per-user databases |
| `ICKY_DB_PATH` | provider-specific | Legacy single database path |

## Claude MCP Configuration

```json
{
  "mcpServers": {
    "ickyMCP": {
      "command": "python",
      "args": ["C:\\Users\\devan\\OneDrive\\Desktop\\Projects\\ickyMCP\\run.py"],
      "env": {
        "ICKY_EMBEDDING_PROVIDER": "voyage",
        "ICKY_VOYAGE_API_KEY": "YOUR_VOYAGE_API_KEY",
        "ICKY_CHUNK_SIZE": "5000",
        "ICKY_CHUNK_OVERLAP": "500"
      }
    }
  }
}
```

## Tools

- `index`: index a file or directory; accepts `user_id`, `patterns`, `exclude`, and `force`
- `search`: semantic query over indexed chunks; accepts `document_ids`, `path_filter`, and `file_types`
- `similar`: find chunks similar to supplied text
- `refresh`: re-index changed files and remove deleted files from the index
- `list`: list indexed documents and their IDs
- `delete`: delete by path, document IDs, or all documents
- `status`: return database, embedding, and chunking status

## Verification

```bash
python -m compileall -q src run.py fast_index.py
python -c "from src.config import EMBEDDING_PROVIDER; print(EMBEDDING_PROVIDER)"
```

The root `test_*.py` files are integration scripts that expect local documents under `docs/` and, for the default backend, a configured Voyage API key.

## Data Hygiene

Generated databases and local document folders are ignored by git: `*.db`, `*.db.bak`, `*.sqlite`, `*.sqlite3`, `data/`, and `docs/`. Keep real API keys in environment variables or `.env`, not in tracked files.

## License

MIT
