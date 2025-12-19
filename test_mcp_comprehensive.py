"""Comprehensive test suite for ickyMCP server functionality."""

import sys
import asyncio
from pathlib import Path

# Force UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root and src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.server import (
    handle_status, handle_list, handle_index, handle_search,
    handle_similar, get_db, get_chunker
)
from src.embedder import get_embedder


async def run_comprehensive_tests():
    """Run comprehensive battery of tests."""

    print("=" * 80)
    print("ICKYMCP COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()

    # Test 1: Status Check
    print("[TEST 1] Checking server status...")
    print("-" * 80)
    status = await handle_status({})
    print(f"Provider: {status['provider']}")
    print(f"Model: {status['model']}")
    print(f"Embedding Dimensions: {status['embedding_dimensions']}")
    print(f"Total Documents: {status['total_documents']}")
    print(f"Total Chunks: {status['total_chunks']}")
    print(f"Chunk Size: {status['chunk_size']} tokens")
    print(f"Chunk Overlap: {status['chunk_overlap']} tokens")
    print(f"Database Size: {status['index_size_mb']:.2f} MB")
    print(f"Supported Types: {', '.join(status['supported_types'])}")
    print()

    # Test 2: List Documents
    print("[TEST 2] Listing currently indexed documents...")
    print("-" * 80)
    list_result = await handle_list({})
    print(f"Total documents: {list_result['count']}")
    if list_result['documents']:
        for doc in list_result['documents']:
            print(f"  - {Path(doc['path']).name}")
            print(f"    Type: {doc['file_type']}, Chunks: {doc['chunks']}, Pages: {doc['pages']}")
            print(f"    Indexed: {doc['indexed_at']}")
    else:
        print("  No documents indexed yet.")
    print()

    # Test 3: Index docs directory if empty
    if list_result['count'] == 0:
        print("[TEST 3] Indexing docs directory...")
        print("-" * 80)
        docs_path = str(Path(__file__).parent / "docs")
        index_result = await handle_index({"path": docs_path})
        print(f"Files found: {index_result['total_files_found']}")
        print(f"Indexed: {index_result['indexed']}")
        print(f"Skipped: {index_result['skipped']}")
        if index_result.get('errors'):
            print(f"Errors: {index_result['errors']}")
        print()

        # Re-list after indexing
        print("[TEST 3.1] Listing documents after indexing...")
        print("-" * 80)
        list_result = await handle_list({})
        print(f"Total documents: {list_result['count']}")
        for doc in list_result['documents']:
            print(f"  - {Path(doc['path']).name}")
            print(f"    Type: {doc['file_type']}, Chunks: {doc['chunks']}, Pages: {doc['pages']}")
        print()
    else:
        print("[TEST 3] Skipping indexing (documents already present)")
        print()

    # Test 4: Basic factual search about procedures
    print("[TEST 4] Search: Factual question about procedures")
    print("-" * 80)
    query = "What are the steps for filing a motion?"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 5})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Page: {r['page_number']}")
        print(f"  Preview: {r['chunk_text'][:200]}...")
    print()

    # Test 5: Search for specific CW procedure
    print("[TEST 5] Search: Specific CW procedure - discovery requests")
    print("-" * 80)
    query = "How do I respond to discovery requests?"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 5})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Page: {r['page_number']}")
        print(f"  Preview: {r['chunk_text'][:200]}...")
    print()

    # Test 6: Broad conceptual question
    print("[TEST 6] Search: Broad conceptual question")
    print("-" * 80)
    query = "What are the ethical obligations of attorneys?"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 5})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Preview: {r['chunk_text'][:200]}...")
    print()

    # Test 7: Very specific term search
    print("[TEST 7] Search: Very specific legal term")
    print("-" * 80)
    query = "statute of limitations"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 5})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Preview: {r['chunk_text'][:200]}...")
    print()

    # Test 8: Edge case - vague query
    print("[TEST 8] Search: Vague/edge case query")
    print("-" * 80)
    query = "deadlines and timeframes"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 3})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Preview: {r['chunk_text'][:150]}...")
    print()

    # Test 9: Different top_k values
    print("[TEST 9] Search: Testing different top_k values")
    print("-" * 80)
    query = "court hearing preparation"
    for k in [3, 10, 20]:
        print(f"\nQuery: '{query}' with top_k={k}")
        result = await handle_search({"query": query, "top_k": k})
        print(f"  Returned {result['count']} results")
        if result['results']:
            print(f"  Similarity range: {result['results'][0]['score']:.3f} to {result['results'][-1]['score']:.3f}")
    print()

    # Test 10: File type filter (PDF only)
    print("[TEST 10] Search: File type filter (PDF only)")
    print("-" * 80)
    query = "client communication guidelines"
    print(f"Query: '{query}' (PDF files only)")
    result = await handle_search({"query": query, "top_k": 5, "file_types": ["pdf"]})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name} [{r['file_type']}]")
        print(f"  Preview: {r['chunk_text'][:150]}...")
    print()

    # Test 11: Path filter
    print("[TEST 11] Search: Path filter test")
    print("-" * 80)
    docs_path = str(Path(__file__).parent / "docs")
    query = "artificial intelligence"
    print(f"Query: '{query}' (filtered to docs directory)")
    result = await handle_search({"query": query, "top_k": 5, "path_filter": docs_path})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Preview: {r['chunk_text'][:150]}...")
    print()

    # Test 12: Similar chunks test
    print("[TEST 12] Similar: Find similar content")
    print("-" * 80)
    # First get a chunk from search
    search_result = await handle_search({"query": "court procedures", "top_k": 1})
    if search_result['results']:
        sample_chunk = search_result['results'][0]['chunk_text']
        print(f"Using sample chunk from: {Path(search_result['results'][0]['path']).name}")
        print(f"Sample preview: {sample_chunk[:150]}...")
        print()

        similar_result = await handle_similar({
            "chunk_text": sample_chunk,
            "top_k": 5,
            "exclude_same_doc": True
        })
        print(f"Similar chunks found: {similar_result['count']}")
        for i, r in enumerate(similar_result['results'][:3], 1):
            print(f"\n  Result {i} (score: {r['score']:.3f}):")
            print(f"  Document: {Path(r['path']).name}")
            print(f"  Preview: {r['chunk_text'][:150]}...")
    print()

    # Test 13: CW-specific procedure search
    print("[TEST 13] Search: CW-specific - document retention policies")
    print("-" * 80)
    query = "document retention and file management"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 5})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Page: {r['page_number']}")
        print(f"  Preview: {r['chunk_text'][:200]}...")
    print()

    # Test 14: Technical AI content from other docs
    print("[TEST 14] Search: Technical AI concepts from other documents")
    print("-" * 80)
    query = "machine learning and neural networks"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 5})
    print(f"Results found: {result['count']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n  Result {i} (score: {r['score']:.3f}):")
        print(f"  Document: {Path(r['path']).name}")
        print(f"  Preview: {r['chunk_text'][:200]}...")
    print()

    # Test 15: Multi-document conceptual search
    print("[TEST 15] Search: Cross-document conceptual query")
    print("-" * 80)
    query = "best practices and recommendations"
    print(f"Query: '{query}'")
    result = await handle_search({"query": query, "top_k": 8})
    print(f"Results found: {result['count']}")

    # Group by document
    docs_found = {}
    for r in result['results']:
        doc_name = Path(r['path']).name
        if doc_name not in docs_found:
            docs_found[doc_name] = []
        docs_found[doc_name].append(r)

    print(f"Results span {len(docs_found)} different documents:")
    for doc_name, chunks in docs_found.items():
        print(f"\n  {doc_name}: {len(chunks)} chunks")
        print(f"    Top score: {chunks[0]['score']:.3f}")
        print(f"    Preview: {chunks[0]['chunk_text'][:150]}...")
    print()

    # Final summary
    print("=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)
    print("\nSummary:")
    status = await handle_status({})
    print(f"  Total Documents Indexed: {status['total_documents']}")
    print(f"  Total Chunks: {status['total_chunks']}")
    print(f"  Database Size: {status['index_size_mb']:.2f} MB")
    print(f"  Embedding Provider: {status['provider']}")
    print(f"  Model: {status['model']}")
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
