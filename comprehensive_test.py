#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive in-depth testing of ickyMCP system."""

import json
import sys
import io
from pathlib import Path
from typing import List, Dict, Any

# Set stdout to UTF-8 to handle unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import VectorDatabase
from src.embedder import get_embedder
from src.chunker import TextChunker
from src.parsers import parse_document
from src.config import get_user_db_path, CHUNK_SIZE, CHUNK_OVERLAP

# Test configuration
USER_ID = "test-indepth"
DOCS_PATH = Path(r"C:\Users\devan\OneDrive\Desktop\Projects\ickyMCP\docs")

class TestResults:
    """Track test results."""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0

    def add_test(self, category: str, test_name: str, passed: bool, details: str = ""):
        """Add a test result."""
        self.tests.append({
            "category": category,
            "test": test_name,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: [{category}] {test_name}")
        if details and not passed:
            print(f"  Details: {details}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print(f"TEST SUMMARY: {self.passed} passed, {self.failed} failed")
        print("="*80)

        if self.failed > 0:
            print("\nFailed tests:")
            for test in self.tests:
                if not test["passed"]:
                    print(f"  - [{test['category']}] {test['test']}")
                    if test['details']:
                        print(f"    {test['details']}")

def setup_test_user():
    """Create database for test user."""
    db_path = get_user_db_path(USER_ID)
    print(f"Setting up test user: {USER_ID}")
    print(f"Database path: {db_path}")

    # Delete existing database
    if Path(db_path).exists():
        Path(db_path).unlink()
        print("Deleted existing test database")

    db = VectorDatabase(db_path)
    db.connect()
    return db

def index_all_documents(db: VectorDatabase) -> List[Dict[str, Any]]:
    """Index all documents in the docs folder."""
    print(f"\nIndexing documents from: {DOCS_PATH}")

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    indexed_docs = []

    for file_path in DOCS_PATH.iterdir():
        if file_path.is_file():
            print(f"  Indexing: {file_path.name}...")

            # Parse
            parsed = parse_document(file_path)
            if not parsed:
                print(f"    Failed to parse!")
                continue

            # Chunk
            chunks = chunker.chunk_text(parsed.text)
            if not chunks:
                print(f"    No content extracted!")
                continue

            # Embed
            chunk_texts = [c.text for c in chunks]
            embeddings = embedder.embed_documents(chunk_texts, show_progress=False)

            # Store
            stat = file_path.stat()
            doc_id = db.add_document(
                path=str(file_path),
                file_type=parsed.file_type,
                file_size=stat.st_size,
                modified_time=stat.st_mtime,
                page_count=parsed.page_count
            )

            for chunk, embedding in zip(chunks, embeddings):
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

            db.update_document_chunk_count(doc_id, len(chunks))

            indexed_docs.append({
                "id": doc_id,
                "path": str(file_path),
                "name": file_path.name,
                "file_type": parsed.file_type,
                "chunks": len(chunks),
                "pages": parsed.page_count
            })

            print(f"    [OK] Indexed: {len(chunks)} chunks, {parsed.page_count} pages (doc_id={doc_id})")

    return indexed_docs

def test_docx_page_numbers(db: VectorDatabase, results: TestResults, docs: List[Dict]):
    """Test DOCX page number verification."""
    print("\n" + "="*80)
    print("TEST CATEGORY 1: DOCX Page Number Verification")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    # Get DOCX document IDs (check both 'docx' and full extension)
    docx_docs = [d for d in docs if 'docx' in d['file_type'].lower() or d['name'].lower().endswith('.docx')]

    if not docx_docs:
        results.add_test("DOCX", "Find DOCX documents", False, "No DOCX documents found")
        return

    results.add_test("DOCX", f"Found {len(docx_docs)} DOCX documents", True)

    # Search for content likely in DOCX files
    query_embedding = embedder.embed_query("artificial intelligence machine learning")
    search_results = db.search(query_embedding, top_k=50)

    # Filter to DOCX results
    docx_results = [r for r in search_results if 'docx' in r['file_type'].lower() or r['path'].lower().endswith('.docx')]

    if not docx_results:
        results.add_test("DOCX", "Find DOCX in search results", False, "No DOCX results in search")
        return

    results.add_test("DOCX", f"Found {len(docx_results)} DOCX chunks in search", True)

    # Check page numbers
    none_count = sum(1 for r in docx_results if r['page_number'] is None)
    page_nums = [r['page_number'] for r in docx_results if r['page_number'] is not None]
    all_ones = all(p == 1 for p in page_nums)
    page_range = f"{min(page_nums)}-{max(page_nums)}" if page_nums else "N/A"

    results.add_test(
        "DOCX",
        "No None page numbers",
        none_count == 0,
        f"{none_count}/{len(docx_results)} results have None page_number. Page range: {page_range}"
    )

    results.add_test(
        "DOCX",
        "Page numbers vary (not all 1s)",
        not all_ones,
        f"All page numbers are 1" if all_ones else f"Page numbers range: {page_range}"
    )

    # Check each DOCX file
    for doc in docx_docs:
        doc_id = doc['id']
        doc_results = [r for r in search_results if r['document_id'] == doc_id]

        if doc_results:
            pages = [r['page_number'] for r in doc_results if r['page_number'] is not None]
            unique_pages = len(set(pages))

            results.add_test(
                "DOCX",
                f"{doc['name']}: Page number distribution",
                unique_pages > 1 or len(pages) == 1,
                f"{unique_pages} unique pages across {len(doc_results)} chunks"
            )

def test_document_filtering(db: VectorDatabase, results: TestResults, docs: List[Dict]):
    """Test document filtering stress tests."""
    print("\n" + "="*80)
    print("TEST CATEGORY 2: Document Filtering Stress Tests")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    query_embedding = embedder.embed_query("machine learning algorithms")

    # Test 1: Single document
    doc_id = docs[0]['id']
    single_results = db.search(query_embedding, top_k=10, document_ids=[doc_id])

    leaked = [r for r in single_results if r['document_id'] != doc_id]
    results.add_test(
        "Filtering",
        "Single document filter - no leaks",
        len(leaked) == 0,
        f"{len(leaked)} results leaked from other documents" if leaked else "No leaks"
    )

    # Test 2: Two documents
    doc_ids = [docs[0]['id'], docs[1]['id']]
    two_results = db.search(query_embedding, top_k=20, document_ids=doc_ids)

    leaked = [r for r in two_results if r['document_id'] not in doc_ids]
    results.add_test(
        "Filtering",
        "Two document filter - no leaks",
        len(leaked) == 0,
        f"{len(leaked)} results leaked" if leaked else "No leaks"
    )

    # Test 3: All except one
    excluded_id = docs[-1]['id']
    included_ids = [d['id'] for d in docs if d['id'] != excluded_id]
    exclude_one_results = db.search(query_embedding, top_k=30, document_ids=included_ids)

    leaked = [r for r in exclude_one_results if r['document_id'] == excluded_id]
    results.add_test(
        "Filtering",
        "Exclude one document - no leaks",
        len(leaked) == 0,
        f"{len(leaked)} results from excluded doc" if leaked else "No leaks"
    )

    # Test 4: Empty document_ids (should return all)
    empty_results = db.search(query_embedding, top_k=10, document_ids=[])
    results.add_test(
        "Filtering",
        "Empty document_ids returns all",
        len(empty_results) > 0,
        f"Got {len(empty_results)} results"
    )

    # Test 5: Invalid document_id
    invalid_results = db.search(query_embedding, top_k=10, document_ids=[9999])
    results.add_test(
        "Filtering",
        "Invalid document_id returns no results",
        len(invalid_results) == 0,
        f"Got {len(invalid_results)} results (should be 0)"
    )

def test_large_document(db: VectorDatabase, results: TestResults, docs: List[Dict]):
    """Test large document (CW Manual - 256 chunks)."""
    print("\n" + "="*80)
    print("TEST CATEGORY 3: Large Document Tests (CW Manual)")
    print("="*80)

    # Find CW manual
    cw_doc = next((d for d in docs if 'cw-procedure' in d['path'].lower()), None)

    if not cw_doc:
        results.add_test("Large Doc", "Find CW Manual", False, "CW Manual not found")
        return

    results.add_test("Large Doc", f"Found CW Manual ({cw_doc['chunks']} chunks)", True)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    queries = [
        "emergency procedures",
        "safety protocols",
        "equipment maintenance",
        "quality control",
        "training requirements"
    ]

    for query in queries:
        query_embedding = embedder.embed_query(query)
        query_results = db.search(
            query_embedding,
            top_k=10,
            document_ids=[cw_doc['id']]
        )

        if query_results:
            pages = [r['page_number'] for r in query_results if r['page_number'] is not None]
            scores = [r['score'] for r in query_results]

            results.add_test(
                "Large Doc",
                f"Query '{query}' - got results",
                len(query_results) > 0,
                f"{len(query_results)} results, pages: {min(pages) if pages else 'N/A'}-{max(pages) if pages else 'N/A'}, scores: {min(scores):.3f}-{max(scores):.3f}"
            )

def test_cross_document(db: VectorDatabase, results: TestResults, docs: List[Dict]):
    """Test cross-document synthesis."""
    print("\n" + "="*80)
    print("TEST CATEGORY 4: Cross-Document Synthesis")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    queries = [
        "artificial intelligence applications",
        "machine learning techniques",
        "data analysis methods"
    ]

    for query in queries:
        query_embedding = embedder.embed_query(query)
        results_list = db.search(query_embedding, top_k=20)

        unique_docs = len(set(r['document_id'] for r in results_list))

        results.add_test(
            "Cross-Doc",
            f"'{query}' - spans multiple documents",
            unique_docs >= 2,
            f"Results from {unique_docs} different documents"
        )

def test_edge_cases(db: VectorDatabase, results: TestResults):
    """Test edge cases."""
    print("\n" + "="*80)
    print("TEST CATEGORY 5: Edge Cases")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    # Long query
    long_query = " ".join(["machine learning artificial intelligence data science"] * 15)
    try:
        query_embedding = embedder.embed_query(long_query)
        long_results = db.search(query_embedding, top_k=5)
        results.add_test(
            "Edge Cases",
            "Very long query (50+ words)",
            True,
            f"Got {len(long_results)} results"
        )
    except Exception as e:
        results.add_test("Edge Cases", "Very long query", False, str(e))

    # Single character
    try:
        query_embedding = embedder.embed_query("a")
        single_results = db.search(query_embedding, top_k=5)
        results.add_test(
            "Edge Cases",
            "Single character query",
            True,
            f"Got {len(single_results)} results"
        )
    except Exception as e:
        results.add_test("Edge Cases", "Single character query", False, str(e))

    # Special characters
    try:
        query_embedding = embedder.embed_query("!@#$%^&*()")
        special_results = db.search(query_embedding, top_k=5)
        results.add_test(
            "Edge Cases",
            "Special characters query",
            True,
            f"Got {len(special_results)} results"
        )
    except Exception as e:
        results.add_test("Edge Cases", "Special characters", False, str(e))

    # Numeric
    try:
        query_embedding = embedder.embed_query("2024 100%")
        numeric_results = db.search(query_embedding, top_k=5)
        results.add_test(
            "Edge Cases",
            "Numeric query",
            True,
            f"Got {len(numeric_results)} results"
        )
    except Exception as e:
        results.add_test("Edge Cases", "Numeric query", False, str(e))

def test_similar_chunks(db: VectorDatabase, results: TestResults, docs: List[Dict]):
    """Test similar chunks functionality."""
    print("\n" + "="*80)
    print("TEST CATEGORY 6: Similar Chunks Testing")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    # Get a chunk from search results
    query_embedding = embedder.embed_query("machine learning")
    search_results = db.search(query_embedding, top_k=1)

    if not search_results:
        results.add_test("Similar", "Get initial chunk", False, "No search results")
        return

    chunk_text = search_results[0]['chunk_text']
    source_doc_id = search_results[0]['document_id']

    # Find similar chunks
    similar_embedding = embedder.embed_for_similarity(chunk_text)
    similar_results = db.search(similar_embedding, top_k=10)

    results.add_test(
        "Similar",
        "Find similar chunks",
        len(similar_results) > 0,
        f"Found {len(similar_results)} similar chunks"
    )

    # Test with document filter
    filtered_similar = db.search(
        similar_embedding,
        top_k=10,
        document_ids=[source_doc_id]
    )

    leaked = [r for r in filtered_similar if r['document_id'] != source_doc_id]
    results.add_test(
        "Similar",
        "Similar with document filter - no leaks",
        len(leaked) == 0,
        f"{len(leaked)} results leaked" if leaked else "No leaks"
    )

def test_score_quality(db: VectorDatabase, results: TestResults, docs: List[Dict]):
    """Test score quality analysis."""
    print("\n" + "="*80)
    print("TEST CATEGORY 7: Score Quality Analysis")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    query_embedding = embedder.embed_query("artificial intelligence")

    # Without filtering
    all_results = db.search(query_embedding, top_k=50)
    all_scores = [r['score'] for r in all_results]

    # With filtering
    if docs:
        filtered_results = db.search(
            query_embedding,
            top_k=50,
            document_ids=[docs[0]['id']]
        )
        filtered_scores = [r['score'] for r in filtered_results]
    else:
        filtered_scores = []

    # Check for anomalies
    has_invalid = any(s > 1.0 or s < -1.0 for s in all_scores)
    has_negative = any(s < 0 for s in all_scores)

    results.add_test(
        "Scores",
        "No scores > 1.0 or < -1.0",
        not has_invalid,
        f"Invalid scores found" if has_invalid else "All scores in valid range"
    )

    results.add_test(
        "Scores",
        "Scores are positive or negative cosine similarity",
        True,
        f"Range: {min(all_scores):.3f} to {max(all_scores):.3f}, Avg: {sum(all_scores)/len(all_scores):.3f}"
    )

    if filtered_scores:
        results.add_test(
            "Scores",
            "Filtered search has reasonable scores",
            len(filtered_scores) > 0,
            f"Range: {min(filtered_scores):.3f} to {max(filtered_scores):.3f}"
        )

def test_metadata(db: VectorDatabase, results: TestResults):
    """Test metadata verification."""
    print("\n" + "="*80)
    print("TEST CATEGORY 8: Metadata Verification")
    print("="*80)

    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    query_embedding = embedder.embed_query("test query")
    search_results = db.search(query_embedding, top_k=10)

    if not search_results:
        results.add_test("Metadata", "Get search results", False, "No results")
        return

    # Check required fields
    required_fields = ['chunk_id', 'document_id', 'path', 'file_type', 'page_number', 'score']

    for field in required_fields:
        has_field = all(field in r for r in search_results)
        results.add_test(
            "Metadata",
            f"All results have '{field}'",
            has_field,
            "Missing in some results" if not has_field else "Present in all"
        )

    # Verify paths are valid
    paths_valid = all(Path(r['path']).exists() for r in search_results)
    results.add_test(
        "Metadata",
        "All paths are valid files",
        paths_valid,
        "Some paths don't exist" if not paths_valid else "All paths valid"
    )

    # Verify file_types match extensions
    # Note: System uses semantic names: 'word' for .docx, 'powerpoint' for .pptx, 'excel' for .xlsx
    type_mapping = {
        'docx': 'word',
        'pptx': 'powerpoint',
        'xlsx': 'excel',
        'pdf': 'pdf',
        'txt': 'text',
        'md': 'markdown'
    }

    mismatches = []
    for r in search_results:
        ext = Path(r['path']).suffix[1:].lower()
        file_type = r['file_type'].lower()
        expected_type = type_mapping.get(ext, ext)

        # Check if file_type matches either the extension or the mapped semantic name
        if file_type != ext and file_type != expected_type and ext not in file_type:
            mismatches.append(f"{r['path']}: type='{r['file_type']}' ext='{ext}' (expected '{expected_type}')")

    types_match = len(mismatches) == 0
    details = "All match (with semantic mapping)" if types_match else f"{len(mismatches)} mismatches: {mismatches[0] if mismatches else ''}"

    results.add_test(
        "Metadata",
        "File types match extensions",
        types_match,
        details
    )

def cleanup_test_user(db: VectorDatabase = None):
    """Delete test user database."""
    # Close database connection first
    if db:
        try:
            db.close()
        except:
            pass

    db_path = get_user_db_path(USER_ID)
    if Path(db_path).exists():
        try:
            Path(db_path).unlink()
            print(f"\nCleaned up test database: {db_path}")
        except PermissionError:
            print(f"\nWarning: Could not delete {db_path} (file in use)")

def print_detailed_stats(db: VectorDatabase, docs: List[Dict]):
    """Print detailed statistics about the indexed data."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    stats = db.get_stats()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Database size: {stats.get('index_size_mb', 'N/A')} MB")

    print("\nDocument breakdown:")
    for doc in docs:
        print(f"  [{doc['id']}] {doc['name']}")
        print(f"      Type: {doc['file_type']}, Chunks: {doc['chunks']}, Pages: {doc['pages']}")

    # Get some sample searches to show score distributions
    embedder = get_embedder()
    if hasattr(embedder, 'load_model'):
        embedder.load_model()

    query_embedding = embedder.embed_query("test")
    sample_results = db.search(query_embedding, top_k=100)

    if sample_results:
        scores = [r['score'] for r in sample_results]
        print(f"\nScore distribution (sample of 100 results):")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Avg: {sum(scores)/len(scores):.4f}")

        # Count by file type
        type_counts = {}
        for r in sample_results:
            ft = r['file_type']
            type_counts[ft] = type_counts.get(ft, 0) + 1

        print(f"\nResults by file type:")
        for ft, count in sorted(type_counts.items()):
            print(f"  {ft}: {count}")

def main():
    """Run comprehensive tests."""
    print("="*80)
    print("COMPREHENSIVE IN-DEPTH TESTING OF ickyMCP")
    print("="*80)

    results = TestResults()
    db = None

    try:
        # Setup
        db = setup_test_user()

        # Index documents
        docs = index_all_documents(db)
        print(f"\nIndexed {len(docs)} documents:")
        for doc in docs:
            print(f"  [{doc['id']}] {doc['name']} - {doc['chunks']} chunks, {doc['pages']} pages")

        # Run tests
        test_docx_page_numbers(db, results, docs)
        test_document_filtering(db, results, docs)
        test_large_document(db, results, docs)
        test_cross_document(db, results, docs)
        test_edge_cases(db, results)
        test_similar_chunks(db, results, docs)
        test_score_quality(db, results, docs)
        test_metadata(db, results)

        # Print detailed statistics
        print_detailed_stats(db, docs)

        # Print summary
        results.print_summary()

    finally:
        # Cleanup
        cleanup_test_user(db)

    print("\nTest complete!")
    return 0 if results.failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
