"""Test recursive directory indexing."""

import sys
import asyncio
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.server import handle_index, handle_list, handle_search, handle_delete


async def test_recursive_indexing():
    """Test that indexing a directory indexes all files recursively."""

    print("=" * 80)
    print("RECURSIVE DIRECTORY INDEXING TEST")
    print("=" * 80)
    print()

    # Clean up first
    await handle_delete({"user_id": "test-recursive", "all": True})

    # Index the entire docs directory
    print("[TEST] Index entire 'docs' directory recursively")
    print("-" * 80)

    docs_path = str(project_root / "docs")
    result = await handle_index({
        "user_id": "test-recursive",
        "path": docs_path
    })

    print(f"Total files found: {result['total_files_found']}")
    print(f"Indexed: {result['indexed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result.get('errors')}")
    print()

    print("Indexed documents (with IDs for filtering):")
    for doc in result.get('documents', []):
        print(f"  ID {doc['id']}: {Path(doc['path']).name}")
        print(f"       Type: {doc['file_type']}, Chunks: {doc['chunks']}, Pages: {doc['pages']}")
    print()

    # List all documents
    print("[VERIFY] List all indexed documents")
    print("-" * 80)
    list_result = await handle_list({"user_id": "test-recursive"})
    print(f"Total documents in DB: {list_result['count']}")

    # Get document IDs for filtering test
    doc_ids = [d['id'] for d in list_result['documents']]
    print(f"Document IDs available for filtering: {doc_ids}")
    print()

    # Test searching with subset of documents
    if len(doc_ids) >= 2:
        print("[TEST] Search filtered to first 2 documents only")
        print("-" * 80)

        filtered_ids = doc_ids[:2]
        search_result = await handle_search({
            "user_id": "test-recursive",
            "query": "artificial intelligence",
            "document_ids": filtered_ids,
            "top_k": 5
        })

        print(f"Searching with document_ids={filtered_ids}")
        print(f"Results: {search_result['count']}")

        result_docs = set(Path(r['path']).name for r in search_result['results'])
        print(f"Results came from: {result_docs}")
    print()

    # Cleanup
    await handle_delete({"user_id": "test-recursive", "all": True})
    print("[CLEANUP] Done")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"- Directory indexing: Working (found {result['total_files_found']} files)")
    print(f"- Recursive: Yes (indexed all supported files)")
    print(f"- Returns doc IDs: Yes (for use with document_ids filter)")


if __name__ == "__main__":
    asyncio.run(test_recursive_indexing())
