"""Additional targeted tests for ickyMCP - focus on CW document and edge cases."""

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

from src.server import handle_search, handle_status, handle_list
from src.database import VectorDatabase
from src.config import DB_PATH


async def run_additional_tests():
    """Run additional targeted tests."""

    print("=" * 80)
    print("ADDITIONAL ICKYMCP TESTS - CW DOCUMENT FOCUS")
    print("=" * 80)
    print()

    # Test path filter issue
    print("[DIAGNOSTIC] Checking actual file paths in database...")
    print("-" * 80)
    db = VectorDatabase(DB_PATH)
    db.connect()
    docs = db.list_documents()
    for doc in docs:
        print(f"  Path: {doc['path']}")
    print()

    # CW-specific procedural tests
    cw_queries = [
        ("What is the process for foster parent certification?", 5),
        ("How do I handle a child abuse investigation?", 5),
        ("What are the requirements for permanency planning?", 5),
        ("Explain the foster care placement procedures", 5),
        ("What documentation is required for court hearings?", 5),
        ("How should caseworkers handle emergency removals?", 5),
        ("What are the guidelines for home visits?", 5),
        ("Describe the adoption finalization process", 5),
    ]

    for i, (query, top_k) in enumerate(cw_queries, 1):
        print(f"[CW TEST {i}] Query: '{query}'")
        print("-" * 80)
        result = await handle_search({"query": query, "top_k": top_k})
        print(f"Results: {result['count']}")

        if result['results']:
            top_result = result['results'][0]
            print(f"\nTop Match (score: {top_result['score']:.3f}):")
            print(f"  Document: {Path(top_result['path']).name}")
            print(f"  Page: {top_result['page_number']}")
            print(f"  Chunk Index: {top_result['chunk_index']}")
            print(f"  Token Count: {top_result['token_count']}")
            print(f"  Preview: {top_result['chunk_text'][:300]}...")

            # Show score distribution
            scores = [r['score'] for r in result['results']]
            print(f"\nScore distribution: {scores[0]:.3f} (best) to {scores[-1]:.3f} (worst)")
        print()

    # Test with exact path for path_filter
    print("[PATH FILTER TEST] Using exact document path")
    print("-" * 80)
    if docs:
        cw_doc = [d for d in docs if 'cw-procedure' in d['path'].lower()]
        if cw_doc:
            cw_path = cw_doc[0]['path']
            print(f"CW Document path: {cw_path}")

            # Get directory portion
            cw_dir = str(Path(cw_path).parent)
            print(f"Directory portion: {cw_dir}")

            query = "foster care placement"
            print(f"\nQuery: '{query}' with path_filter='{cw_dir}'")
            result = await handle_search({
                "query": query,
                "top_k": 5,
                "path_filter": cw_dir
            })
            print(f"Results found: {result['count']}")
            if result['results']:
                for r in result['results'][:3]:
                    print(f"  - {Path(r['path']).name} (score: {r['score']:.3f})")
    print()

    # Test edge cases
    print("[EDGE CASE 1] Very long query")
    print("-" * 80)
    long_query = "What are the specific procedures, requirements, guidelines, and best practices that caseworkers must follow when conducting home visits for foster care placements including documentation requirements"
    result = await handle_search({"query": long_query, "top_k": 3})
    print(f"Results: {result['count']}")
    if result['results']:
        print(f"Top score: {result['results'][0]['score']:.3f}")
        print(f"Document: {Path(result['results'][0]['path']).name}")
    print()

    print("[EDGE CASE 2] Single word query")
    print("-" * 80)
    result = await handle_search({"query": "adoption", "top_k": 5})
    print(f"Results: {result['count']}")
    if result['results']:
        scores = [r['score'] for r in result['results']]
        print(f"Score range: {scores[0]:.3f} to {scores[-1]:.3f}")
        doc_counts = {}
        for r in result['results']:
            doc = Path(r['path']).name
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        print(f"Document distribution: {doc_counts}")
    print()

    print("[EDGE CASE 3] Query with special characters")
    print("-" * 80)
    result = await handle_search({"query": "parent's rights & responsibilities", "top_k": 5})
    print(f"Results: {result['count']}")
    if result['results']:
        print(f"Top score: {result['results'][0]['score']:.3f}")
    print()

    print("[EDGE CASE 4] Very specific page number query")
    print("-" * 80)
    result = await handle_search({"query": "What information is on page 1000?", "top_k": 5})
    print(f"Results: {result['count']}")
    if result['results']:
        for r in result['results'][:3]:
            print(f"  Page {r['page_number']}: score {r['score']:.3f}")
    print()

    # Performance test - retrieve many results
    print("[PERFORMANCE TEST] Retrieving 50 results")
    print("-" * 80)
    import time
    start = time.time()
    result = await handle_search({"query": "child welfare services", "top_k": 50})
    elapsed = time.time() - start
    print(f"Retrieved {result['count']} results in {elapsed:.3f} seconds")
    if result['results']:
        unique_docs = set(Path(r['path']).name for r in result['results'])
        print(f"Results span {len(unique_docs)} different documents")
        print(f"Score range: {result['results'][0]['score']:.3f} to {result['results'][-1]['score']:.3f}")
    print()

    # Test AI document searches
    print("[AI DOCUMENT TEST 1] Technical AI query")
    print("-" * 80)
    result = await handle_search({"query": "transformer architecture and attention mechanisms", "top_k": 5})
    print(f"Results: {result['count']}")
    if result['results']:
        for i, r in enumerate(result['results'], 1):
            print(f"  {i}. {Path(r['path']).name} (score: {r['score']:.3f})")
    print()

    print("[AI DOCUMENT TEST 2] Reinforcement learning concepts")
    print("-" * 80)
    result = await handle_search({"query": "reward modeling and policy optimization", "top_k": 5})
    print(f"Results: {result['count']}")
    if result['results']:
        for i, r in enumerate(result['results'], 1):
            print(f"  {i}. {Path(r['path']).name} (score: {r['score']:.3f})")
    print()

    print("[AI DOCUMENT TEST 3] Filter to only DOCX files")
    print("-" * 80)
    result = await handle_search({
        "query": "artificial intelligence trends",
        "top_k": 5,
        "file_types": ["word"]
    })
    print(f"Results: {result['count']}")
    if result['results']:
        for i, r in enumerate(result['results'], 1):
            print(f"  {i}. {Path(r['path']).name} [{r['file_type']}] (score: {r['score']:.3f})")
    print()

    # Multi-document conceptual tests
    print("[MULTI-DOC TEST 1] Broad technology query")
    print("-" * 80)
    result = await handle_search({"query": "data processing and analysis techniques", "top_k": 10})
    print(f"Results: {result['count']}")
    doc_breakdown = {}
    for r in result['results']:
        doc = Path(r['path']).name
        if doc not in doc_breakdown:
            doc_breakdown[doc] = {'count': 0, 'best_score': r['score']}
        doc_breakdown[doc]['count'] += 1

    for doc, info in sorted(doc_breakdown.items(), key=lambda x: x[1]['best_score'], reverse=True):
        print(f"  {doc}: {info['count']} results, best score: {info['best_score']:.3f}")
    print()

    print("=" * 80)
    print("ADDITIONAL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_additional_tests())
