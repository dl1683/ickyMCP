"""ickyMCP - RAG MCP Server for Document Search.

Multi-tenant architecture:
- Each user has their own database (user_id required for all operations)
- Documents are isolated per user
- Search can be filtered to specific document IDs for chat/matter isolation
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS, DB_PATH,
    EMBEDDING_PROVIDER, VOYAGE_MODEL, LOCAL_EMBEDDING_MODEL,
    get_user_db_path
)
from .database import VectorDatabase
from .parsers import parse_document, is_supported
from .chunker import TextChunker
from .embedder import get_embedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ickyMCP")

# Initialize server
server = Server("ickyMCP")

# Global instances - chunker is shared, but DB is per-user
chunker: Optional[TextChunker] = None

# Cache of user databases to avoid reconnecting
_user_dbs: dict[str, VectorDatabase] = {}


def get_db(user_id: Optional[str] = None) -> VectorDatabase:
    """Get or create database connection for a user.

    Args:
        user_id: User identifier. If None, uses legacy single-DB mode.

    Returns:
        VectorDatabase instance for the user
    """
    if user_id is None:
        # Legacy mode - single shared database
        if "_legacy" not in _user_dbs:
            db = VectorDatabase(DB_PATH)
            db.connect()
            _user_dbs["_legacy"] = db
        return _user_dbs["_legacy"]

    # Multi-tenant mode - per-user database
    if user_id not in _user_dbs:
        db_path = get_user_db_path(user_id)
        db = VectorDatabase(db_path)
        db.connect()
        _user_dbs[user_id] = db
        logger.info(f"Initialized database for user {user_id} at {db_path}")

    return _user_dbs[user_id]


def get_chunker() -> TextChunker:
    """Get or create chunker instance."""
    global chunker
    if chunker is None:
        chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return chunker


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="index",
            description="Index documents from a file or directory for semantic search. Supports: .txt, .md, .pdf, .docx, .pptx, .xlsx",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory path to index"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation. Each user has their own database."
                    },
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to match (e.g., ['*.pdf', '*.docx']). Default: all supported types",
                        "default": ["*"]
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns to exclude",
                        "default": []
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-index even if file unchanged",
                        "default": False
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="search",
            description="Semantic search across indexed documents. Returns relevant text chunks. Use document_ids to filter to specific documents (e.g., for chat/matter isolation).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation"
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter to only search within specific document IDs. Use this to limit search to documents selected for a particular chat or matter."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "path_filter": {
                        "type": "string",
                        "description": "Filter results to paths starting with this prefix"
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by file types (e.g., ['pdf', 'docx'])"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="similar",
            description="Find chunks similar to a given text. Useful for finding related content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_text": {
                        "type": "string",
                        "description": "Text to find similar chunks for"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation"
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Filter to only search within specific document IDs"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "exclude_same_doc": {
                        "type": "boolean",
                        "description": "Exclude chunks from the same document",
                        "default": True
                    }
                },
                "required": ["chunk_text"]
            }
        ),
        Tool(
            name="refresh",
            description="Re-index only files that have changed since last indexing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation"
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional: refresh only this path/directory"
                    }
                }
            }
        ),
        Tool(
            name="list",
            description="List all indexed documents. Returns document IDs that can be used for filtering searches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation"
                    },
                    "path_filter": {
                        "type": "string",
                        "description": "Filter to paths starting with this prefix"
                    }
                }
            }
        ),
        Tool(
            name="delete",
            description="Remove documents from the index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to delete (file or directory prefix)"
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Delete specific documents by ID"
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Delete entire index (requires confirmation)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="status",
            description="Get server status and index statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier for multi-tenant isolation"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "index":
            result = await handle_index(arguments)
        elif name == "search":
            result = await handle_search(arguments)
        elif name == "similar":
            result = await handle_similar(arguments)
        elif name == "refresh":
            result = await handle_refresh(arguments)
        elif name == "list":
            result = await handle_list(arguments)
        elif name == "delete":
            result = await handle_delete(arguments)
        elif name == "status":
            result = await handle_status(arguments)
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=str(result))]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_index(args: dict) -> dict:
    """Index documents from a path (file or directory).

    When given a directory, recursively indexes all supported files within it.
    This is useful for indexing a "matter" folder containing multiple documents.

    Returns document IDs that can be used for filtering searches.
    """
    path = Path(args["path"])
    user_id = args.get("user_id")
    patterns = args.get("patterns", ["*"])
    exclude = args.get("exclude", [])
    force = args.get("force", False)

    if not path.exists():
        return {"error": f"Path does not exist: {path}"}

    database = get_db(user_id)
    text_chunker = get_chunker()
    embedder = get_embedder()

    # Ensure model is loaded (only for local embedder)
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    indexed = 0
    skipped = 0
    errors = []
    indexed_documents = []  # Track indexed docs with their IDs

    # Collect files to process
    files_to_process = []

    if path.is_file():
        if is_supported(path):
            files_to_process.append(path)
    else:
        # Directory - recursively find all matching files
        for pattern in patterns:
            for file_path in path.rglob(pattern):
                if file_path.is_file() and is_supported(file_path):
                    # Check exclusions
                    excluded = any(file_path.match(ex) for ex in exclude)
                    if not excluded and file_path not in files_to_process:
                        files_to_process.append(file_path)

    # Process each file
    for file_path in files_to_process:
        try:
            stat = file_path.stat()

            # Check if needs re-indexing
            if not force and not database.document_needs_reindex(
                str(file_path), stat.st_mtime, stat.st_size
            ):
                skipped += 1
                continue

            # Delete existing if re-indexing
            database.delete_document(str(file_path))

            # Parse document
            parsed = parse_document(file_path)
            if parsed is None:
                errors.append(f"Could not parse: {file_path}")
                continue

            # Chunk the text
            chunks = text_chunker.chunk_text(parsed.text)
            if not chunks:
                errors.append(f"No content extracted: {file_path}")
                continue

            # Generate embeddings for all chunks
            chunk_texts = [c.text for c in chunks]
            embeddings = embedder.embed_documents(chunk_texts, show_progress=False)

            # Store document
            doc_id = database.add_document(
                path=str(file_path),
                file_type=parsed.file_type,
                file_size=stat.st_size,
                modified_time=stat.st_mtime,
                page_count=parsed.page_count
            )

            # Store chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                database.add_chunk(
                    document_id=doc_id,
                    chunk_index=chunk.chunk_index,
                    chunk_text=chunk.text,
                    token_count=chunk.token_count,
                    embedding=embedding,
                    page_number=chunk.page_number,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char
                )

            database.update_document_chunk_count(doc_id, len(chunks))
            indexed += 1

            # Track indexed document with its ID for return
            indexed_documents.append({
                "id": doc_id,
                "path": str(file_path),
                "file_type": parsed.file_type,
                "chunks": len(chunks),
                "pages": parsed.page_count
            })

            logger.info(f"Indexed {file_path}: {len(chunks)} chunks (doc_id={doc_id})")

        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")
            logger.exception(f"Error indexing {file_path}")

    return {
        "indexed": indexed,
        "skipped": skipped,
        "errors": errors if errors else None,
        "total_files_found": len(files_to_process),
        "documents": indexed_documents  # List of indexed docs with IDs
    }


async def handle_search(args: dict) -> dict:
    """Search indexed documents."""
    query = args["query"]
    user_id = args.get("user_id")
    document_ids = args.get("document_ids")
    top_k = args.get("top_k", 10)
    path_filter = args.get("path_filter")
    file_types = args.get("file_types")

    database = get_db(user_id)
    embedder = get_embedder()

    # Ensure model is loaded (only for local embedder)
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    # Generate query embedding
    query_embedding = embedder.embed_query(query)

    # Search with document_ids filter
    results = database.search(
        query_embedding=query_embedding,
        top_k=top_k,
        path_filter=path_filter,
        file_types=file_types,
        document_ids=document_ids
    )

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "filtered_to_documents": document_ids if document_ids else "all"
    }


async def handle_similar(args: dict) -> dict:
    """Find similar chunks to given text."""
    chunk_text = args["chunk_text"]
    user_id = args.get("user_id")
    document_ids = args.get("document_ids")
    top_k = args.get("top_k", 10)
    exclude_same_doc = args.get("exclude_same_doc", True)

    database = get_db(user_id)
    embedder = get_embedder()

    # Ensure model is loaded (only for local embedder)
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    # Generate embedding for the input text
    text_embedding = embedder.embed_for_similarity(chunk_text)

    # Search for similar (get extra results if excluding same doc)
    search_k = top_k * 2 if exclude_same_doc else top_k
    results = database.search(
        query_embedding=text_embedding,
        top_k=search_k,
        document_ids=document_ids
    )

    # Filter out same document if requested
    if exclude_same_doc and results:
        # Try to identify source document from the input text
        # (this is a heuristic - exact match on first few chars)
        filtered = []
        for r in results:
            # Simple check: if chunk text is too similar, skip
            if chunk_text[:100] not in r["chunk_text"][:100]:
                filtered.append(r)
            if len(filtered) >= top_k:
                break
        results = filtered

    return {
        "input_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
        "results": results[:top_k],
        "count": len(results[:top_k])
    }


async def handle_refresh(args: dict) -> dict:
    """Refresh index for changed files."""
    user_id = args.get("user_id")
    path = args.get("path")

    database = get_db(user_id)

    # Get all indexed documents
    docs = database.list_documents(path_filter=path)

    updated = 0
    added = 0
    removed = 0

    for doc in docs:
        doc_path = Path(doc["path"])

        if not doc_path.exists():
            # File was deleted
            database.delete_document(doc["path"])
            removed += 1
        elif database.document_needs_reindex(doc["path"], doc_path.stat().st_mtime, doc_path.stat().st_size):
            # File was modified - re-index it
            result = await handle_index({"path": doc["path"], "user_id": user_id, "force": True})
            if result.get("indexed", 0) > 0:
                updated += 1

    return {
        "updated": updated,
        "added": added,
        "removed": removed,
        "checked": len(docs)
    }


async def handle_list(args: dict) -> dict:
    """List indexed documents with their IDs for document selection."""
    user_id = args.get("user_id")
    path_filter = args.get("path_filter")

    database = get_db(user_id)
    docs = database.list_documents(path_filter=path_filter)

    return {
        "documents": [
            {
                "id": d["id"],  # Document ID for use with document_ids filter
                "path": d["path"],
                "file_type": d["file_type"],
                "chunks": d["chunk_count"],
                "pages": d["page_count"],
                "indexed_at": d["indexed_at"]
            }
            for d in docs
        ],
        "count": len(docs)
    }


async def handle_delete(args: dict) -> dict:
    """Delete documents from index."""
    user_id = args.get("user_id")
    path = args.get("path")
    document_ids = args.get("document_ids")
    delete_all = args.get("all", False)

    database = get_db(user_id)

    if delete_all:
        count = database.delete_all()
        return {"deleted_chunks": count, "message": "Deleted entire index"}

    # Delete by document IDs if provided
    if document_ids:
        deleted_docs = 0
        deleted_chunks = 0
        for doc_id in document_ids:
            doc = database.get_document_by_id(doc_id)
            if doc:
                chunks = database.delete_document(doc["path"])
                deleted_chunks += chunks
                deleted_docs += 1
        return {
            "deleted_documents": deleted_docs,
            "deleted_chunks": deleted_chunks
        }

    if not path:
        return {"error": "Must specify 'path', 'document_ids', or 'all: true'"}

    # Check if it's a prefix (directory) or exact file
    docs = database.list_documents(path_filter=path)

    deleted = 0
    for doc in docs:
        chunks = database.delete_document(doc["path"])
        deleted += chunks

    return {
        "deleted_documents": len(docs),
        "deleted_chunks": deleted
    }


async def handle_status(args: dict) -> dict:
    """Get server status."""
    user_id = args.get("user_id")
    database = get_db(user_id)
    embedder = get_embedder()

    stats = database.get_stats()

    model_info = embedder.get_model_info()
    return {
        **stats,
        "user_id": user_id if user_id else "legacy_mode",
        "provider": model_info.get("provider", EMBEDDING_PROVIDER),
        "model": model_info.get("model_name", VOYAGE_MODEL if EMBEDDING_PROVIDER == "voyage" else LOCAL_EMBEDDING_MODEL),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "supported_types": list(SUPPORTED_EXTENSIONS.keys()),
        "embedding_dimensions": embedder.dimensions
    }


async def main():
    """Run the MCP server."""
    logger.info("Starting ickyMCP server...")

    # Initialize embedder
    embedder = get_embedder()
    model_info = embedder.get_model_info()
    logger.info(f"Embedder: {model_info.get('provider')} - {model_info.get('model_name')}")

    # Pre-load local model if using local embeddings
    if hasattr(embedder, 'load_model'):
        embedder.load_model()
        logger.info("Local model loaded successfully")

    # Initialize database
    database = get_db()
    logger.info(f"Database initialized at: {DB_PATH}")

    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
