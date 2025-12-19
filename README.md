# ickyMCP

RAG MCP Server for Document Search. Built for legal professionals and business users who need to search across large document collections.

## Features

- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **Document Support**: PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), Markdown, Text
- **4K Token Chunks**: Large chunks preserve context for legal and business documents
- **Incremental Indexing**: Only re-index changed files
- **Local Embeddings**: Uses nomic-embed-text-v1.5 (no API costs)
- **SQLite Storage**: Single portable database file

## Installation

```bash
# Clone or copy the project
cd ickyMCP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ICKY_CHUNK_SIZE` | 4000 | Tokens per chunk |
| `ICKY_CHUNK_OVERLAP` | 500 | Overlap between chunks |
| `ICKY_DB_PATH` | `./icky.db` | Path to SQLite database |
| `ICKY_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |

### Claude Code Configuration

Add to your `claude_desktop_config.json` or MCP settings:

```json
{
  "mcpServers": {
    "ickyMCP": {
      "command": "python",
      "args": ["/path/to/ickyMCP/run.py"],
      "env": {
        "ICKY_CHUNK_SIZE": "4000",
        "ICKY_CHUNK_OVERLAP": "500",
        "ICKY_DB_PATH": "/path/to/icky.db"
      }
    }
  }
}
```

## Usage

### Tools Available

#### `index`
Index documents from a file or directory.

```
index(path="/contracts/2024", patterns=["*.pdf", "*.docx"])
```

#### `search`
Semantic search across indexed documents.

```
search(query="indemnification clause", top_k=10, file_types=["pdf"])
```

#### `similar`
Find chunks similar to a given text.

```
similar(chunk_text="The parties agree to...", top_k=5)
```

#### `refresh`
Re-index only files that have changed.

```
refresh(path="/contracts")
```

#### `list`
List all indexed documents.

```
list(path_filter="/contracts")
```

#### `delete`
Remove documents from the index.

```
delete(path="/contracts/old")
delete(all=true)  # Clear entire index
```

#### `status`
Get server status and statistics.

```
status()
```

## How It Works

1. **Indexing**: Documents are parsed, split into 4K token chunks with 500 token overlap
2. **Embedding**: Each chunk is embedded using nomic-embed-text-v1.5 (768 dimensions)
3. **Storage**: Embeddings stored in SQLite with sqlite-vec for fast vector search
4. **Search**: Query is embedded, compared against all chunks using cosine similarity
5. **Results**: Top-K most similar chunks returned with full text and metadata

## System Requirements

- Python 3.10+
- 4GB RAM (2GB for model + headroom)
- ~1GB disk space (model + database)

## License

MIT
