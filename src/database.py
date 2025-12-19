"""SQLite + sqlite-vec database for vector storage."""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlite_vec

from .config import DB_PATH, EMBEDDING_DIMENSIONS

logger = logging.getLogger("ickyMCP")


class VectorDatabase:
    """Vector database using SQLite + sqlite-vec."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.dimensions = EMBEDDING_DIMENSIONS

    def connect(self) -> None:
        """Initialize database connection and schema.

        Handles dimension mismatch by recreating the database if needed.
        """
        db_exists = self.db_path.exists()

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Check for dimension mismatch if DB exists
        if db_exists:
            if not self._check_dimensions():
                logger.warning(
                    f"Embedding dimension mismatch detected. "
                    f"Expected {EMBEDDING_DIMENSIONS}, recreating database."
                )
                self._recreate_database()

        self._create_schema()

    def _check_dimensions(self) -> bool:
        """Check if the database embedding dimensions match the current config."""
        try:
            # Check if chunk_embeddings table exists
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
            )
            if not cursor.fetchone():
                return True  # No embeddings table yet, OK to create

            # Check dimensions by looking at table info
            # sqlite-vec stores dimension info in the virtual table definition
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
            )
            row = cursor.fetchone()
            if row and row[0]:
                sql = row[0]
                # Parse FLOAT[dimensions] from the SQL
                import re
                match = re.search(r'FLOAT\[(\d+)\]', sql)
                if match:
                    db_dimensions = int(match.group(1))
                    return db_dimensions == EMBEDDING_DIMENSIONS

            return True  # Can't determine, assume OK
        except Exception as e:
            logger.warning(f"Error checking dimensions: {e}")
            return True  # Assume OK on error

    def _recreate_database(self) -> None:
        """Recreate the database with correct dimensions."""
        self.conn.close()

        # Backup old database
        backup_path = self.db_path.with_suffix('.db.bak')
        if backup_path.exists():
            backup_path.unlink()
        self.db_path.rename(backup_path)
        logger.info(f"Old database backed up to {backup_path}")

        # Reconnect to fresh database
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _create_schema(self) -> None:
        """Create database tables if they don't exist."""
        # Documents table - tracks indexed files
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                modified_time REAL NOT NULL,
                indexed_at TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                page_count INTEGER DEFAULT 0
            )
        """)

        # Chunks table - stores text chunks with metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                page_number INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Vector table for embeddings using sqlite-vec
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIMENSIONS}]
            )
        """)

        # Indexes for faster queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")

        self.conn.commit()

    def add_document(
        self,
        path: str,
        file_type: str,
        file_size: int,
        modified_time: float,
        page_count: int = 0
    ) -> int:
        """Add a document to the database. Returns document ID."""
        cursor = self.conn.execute("""
            INSERT INTO documents (path, file_type, file_size, modified_time, indexed_at, page_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_type = excluded.file_type,
                file_size = excluded.file_size,
                modified_time = excluded.modified_time,
                indexed_at = excluded.indexed_at,
                page_count = excluded.page_count
        """, (path, file_type, file_size, modified_time, datetime.utcnow().isoformat(), page_count))

        self.conn.commit()

        # Get the document ID
        cursor = self.conn.execute("SELECT id FROM documents WHERE path = ?", (path,))
        return cursor.fetchone()[0]

    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        chunk_text: str,
        token_count: int,
        embedding: list[float],
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None
    ) -> int:
        """Add a chunk with its embedding. Returns chunk ID."""
        # Insert chunk metadata
        cursor = self.conn.execute("""
            INSERT INTO chunks (document_id, chunk_index, chunk_text, token_count, page_number, start_char, end_char)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (document_id, chunk_index, chunk_text, token_count, page_number, start_char, end_char))

        chunk_id = cursor.lastrowid

        # Insert embedding
        embedding_blob = sqlite_vec.serialize_float32(embedding)
        self.conn.execute("""
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
        """, (chunk_id, embedding_blob))

        return chunk_id

    def update_document_chunk_count(self, document_id: int, chunk_count: int) -> None:
        """Update the chunk count for a document."""
        self.conn.execute(
            "UPDATE documents SET chunk_count = ? WHERE id = ?",
            (chunk_count, document_id)
        )
        self.conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        path_filter: Optional[str] = None,
        file_types: Optional[list[str]] = None,
        document_ids: Optional[list[int]] = None
    ) -> list[dict]:
        """Search for similar chunks using vector similarity.

        Filter-first approach: We first identify which chunk IDs match the filters,
        then perform vector search only on those chunks. This ensures we never
        compute embeddings for irrelevant documents.

        Args:
            query_embedding: The embedding vector for the query
            top_k: Number of results to return
            path_filter: Filter results to paths starting with this prefix
            file_types: Filter by file types (e.g., ['pdf', 'docx'])
            document_ids: Filter to only search within specific document IDs.
                         This is the primary filter for multi-tenant document selection.

        Returns:
            List of matching chunks with metadata and similarity scores
        """
        embedding_blob = sqlite_vec.serialize_float32(query_embedding)

        # Check if any filters are applied
        has_filters = document_ids or path_filter or file_types

        if has_filters:
            # FILTER-FIRST APPROACH:
            # Step 1: Build subquery to get chunk IDs matching filters
            filter_subquery = """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE 1=1
            """
            filter_params = []

            if document_ids:
                placeholders = ",".join("?" * len(document_ids))
                filter_subquery += f" AND d.id IN ({placeholders})"
                filter_params.extend(document_ids)

            if path_filter:
                filter_subquery += " AND d.path LIKE ?"
                filter_params.append(f"{path_filter}%")

            if file_types:
                placeholders = ",".join("?" * len(file_types))
                filter_subquery += f" AND d.file_type IN ({placeholders})"
                filter_params.extend(file_types)

            # Step 2: Vector search only on filtered chunk IDs
            query = f"""
                SELECT
                    c.id,
                    c.chunk_text,
                    c.chunk_index,
                    c.page_number,
                    c.token_count,
                    d.id as document_id,
                    d.path,
                    d.file_type,
                    vec_distance_cosine(ce.embedding, ?) as distance
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE ce.chunk_id IN ({filter_subquery})
                ORDER BY distance ASC
                LIMIT ?
            """
            params = [embedding_blob] + filter_params + [top_k]

        else:
            # No filters - search all chunks
            query = """
                SELECT
                    c.id,
                    c.chunk_text,
                    c.chunk_index,
                    c.page_number,
                    c.token_count,
                    d.id as document_id,
                    d.path,
                    d.file_type,
                    vec_distance_cosine(ce.embedding, ?) as distance
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                JOIN documents d ON d.id = c.document_id
                ORDER BY distance ASC
                LIMIT ?
            """
            params = [embedding_blob, top_k]

        cursor = self.conn.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                "chunk_id": row["id"],
                "chunk_text": row["chunk_text"],
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "token_count": row["token_count"],
                "document_id": row["document_id"],
                "path": row["path"],
                "file_type": row["file_type"],
                "score": 1 - row["distance"]  # Convert distance to similarity score
            })

        return results

    def get_document_by_path(self, path: str) -> Optional[dict]:
        """Get document metadata by path."""
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE path = ?", (path,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def document_needs_reindex(self, path: str, modified_time: float, file_size: int) -> bool:
        """Check if a document needs to be re-indexed."""
        doc = self.get_document_by_path(path)
        if not doc:
            return True
        return doc["modified_time"] != modified_time or doc["file_size"] != file_size

    def delete_document(self, path: str) -> int:
        """Delete a document and its chunks. Returns number of chunks deleted."""
        doc = self.get_document_by_path(path)
        if not doc:
            return 0

        # Get chunk IDs for this document
        cursor = self.conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (doc["id"],)
        )
        chunk_ids = [row["id"] for row in cursor.fetchall()]

        # Delete embeddings
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            self.conn.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )

        # Delete chunks (cascade will handle this if FK is set up)
        self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc["id"],))

        # Delete document
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc["id"],))

        self.conn.commit()
        return len(chunk_ids)

    def delete_all(self) -> int:
        """Delete all documents and chunks. Returns total chunks deleted."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]

        self.conn.execute("DELETE FROM chunk_embeddings")
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()

        return count

    def list_documents(self, path_filter: Optional[str] = None) -> list[dict]:
        """List all indexed documents with their IDs.

        Returns documents with 'id' field that can be used for document_ids filtering.
        """
        query = "SELECT * FROM documents"
        params = []

        if path_filter:
            query += " WHERE path LIKE ?"
            params.append(f"{path_filter}%")

        query += " ORDER BY indexed_at DESC"

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_documents_by_ids(self, document_ids: list[int]) -> list[dict]:
        """Get documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            List of document dictionaries
        """
        if not document_ids:
            return []

        placeholders = ",".join("?" * len(document_ids))
        query = f"SELECT * FROM documents WHERE id IN ({placeholders}) ORDER BY indexed_at DESC"

        cursor = self.conn.execute(query, document_ids)
        return [dict(row) for row in cursor.fetchall()]

    def get_document_by_id(self, document_id: int) -> Optional[dict]:
        """Get a single document by ID.

        Args:
            document_id: The document ID

        Returns:
            Document dictionary or None if not found
        """
        cursor = self.conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT MAX(indexed_at) FROM documents")
        last_indexed = cursor.fetchone()[0]

        # Get database file size
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "index_size_mb": round(db_size_mb, 2),
            "last_indexed": last_indexed
        }

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
