import logging
from .models import Document, DocumentChunk
from .services import retrieve_relevant_chunks, retrieve_relevant_chunks_simple, retrieve_relevant_chunks_bm25, retrieve_relevant_chunks_hybrid

logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    {
        "question": "who is mahyar",
        "expected_keywords": ["mahyar", "computer engineering", "sharif"],
        "expected_source_doc": None,
    },
    {
        "question": "what programming languages",
        "expected_keywords": ["python", "java", "c"],
        "expected_source_doc": None,
    },
    {
        "question": "what projects",
        "expected_keywords": ["sgit", "gwent", "llm"],
        "expected_source_doc": None,
    },
]


def evaluate_retrieval(search_method="hybrid", document_ids=None):
    results = []

    for test in EVAL_QUESTIONS:
        question = test["question"]
        expected_keywords = [k.lower() for k in test["expected_keywords"]]

        chunks = retrieve_relevant_chunks(
            question=question,
            search_method=search_method,
            limit=3,
            document_ids=document_ids,
        )

        retrieved_text = " ".join([c.content.lower() for c in chunks])

        keyword_hits = sum(1 for kw in expected_keywords if kw in retrieved_text)
        keyword_recall = keyword_hits / len(expected_keywords) if expected_keywords else 0

        has_relevant = keyword_recall > 0

        results.append({
            "question": question,
            "search_method": search_method,
            "chunks_retrieved": len(chunks),
            "keyword_hits": keyword_hits,
            "keyword_recall": round(keyword_recall, 2),
            "has_relevant": has_relevant,
            "sources": [
                {
                    "document": c.document.title,
                    "chunk_index": c.chunk_index,
                    "preview": c.content[:100],
                }
                for c in chunks
            ],
        })

    total_recall = sum(r["keyword_recall"] for r in results) / len(results) if results else 0
    success_rate = sum(1 for r in results if r["has_relevant"]) / len(results) if results else 0

    return {
        "results": results,
        "summary": {
            "total_questions": len(results),
            "avg_keyword_recall": round(total_recall, 2),
            "success_rate": round(success_rate, 2),
            "search_method": search_method,
        },
    }


def compare_search_methods(document_ids=None):
    methods = ["simple", "bm25", "hybrid"]
    comparison = {}

    for method in methods:
        result = evaluate_retrieval(search_method=method, document_ids=document_ids)
        comparison[method] = result["summary"]

    return comparison
