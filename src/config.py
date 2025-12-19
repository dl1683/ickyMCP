"""Configuration for ickyMCP server."""

import os
from pathlib import Path

# Chunking settings
CHUNK_SIZE = int(os.getenv("ICKY_CHUNK_SIZE", "5000"))
CHUNK_OVERLAP = int(os.getenv("ICKY_CHUNK_OVERLAP", "500"))

# Embedding provider: "local" or "voyage"
EMBEDDING_PROVIDER = os.getenv("ICKY_EMBEDDING_PROVIDER", "voyage")

# Local embedding model (sentence-transformers)
LOCAL_EMBEDDING_MODEL = os.getenv("ICKY_LOCAL_MODEL", "nomic-ai/nomic-embed-text-v1.5")
LOCAL_EMBEDDING_DIMENSIONS = 768

# Voyage AI settings
VOYAGE_API_KEY = os.getenv("ICKY_VOYAGE_API_KEY", "pa-YRGTaoUoqv9rgCElaL7RAg0T8FdfIaOpLNCpLZ75fWP")
VOYAGE_MODEL = os.getenv("ICKY_VOYAGE_MODEL", "voyage-3.5-lite")
VOYAGE_DIMENSIONS = int(os.getenv("ICKY_VOYAGE_DIMENSIONS", "1024"))
VOYAGE_MAX_BATCH = 100  # Texts per API call (starts high for speed, adaptive fallback on failure)
VOYAGE_MAX_WORKERS = int(os.getenv("ICKY_VOYAGE_WORKERS", "8"))

# Active embedding dimensions (depends on provider)
EMBEDDING_DIMENSIONS = VOYAGE_DIMENSIONS if EMBEDDING_PROVIDER == "voyage" else LOCAL_EMBEDDING_DIMENSIONS

# Multi-tenant database settings
# Base directory for all user databases
DATA_DIR = Path(os.getenv("ICKY_DATA_DIR", "./data"))

# Legacy single-DB path (for backwards compatibility / local testing)
_default_db = "./icky_voyage.db" if EMBEDDING_PROVIDER == "voyage" else "./icky.db"
DB_PATH = Path(os.getenv("ICKY_DB_PATH", _default_db))


def get_user_db_path(user_id: str) -> Path:
    """Get the database path for a specific user.

    Args:
        user_id: Unique identifier for the user

    Returns:
        Path to the user's database file
    """
    # Sanitize user_id to prevent directory traversal
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    if not safe_user_id:
        raise ValueError("Invalid user_id")

    user_dir = DATA_DIR / safe_user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    db_name = "icky_voyage.db" if EMBEDDING_PROVIDER == "voyage" else "icky.db"
    return user_dir / db_name

# Supported file types
SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "markdown",
    ".pdf": "pdf",
    ".docx": "word",
    ".pptx": "powerpoint",
    ".xlsx": "excel",
}

# Query/document prefixes for nomic model (local only)
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "
