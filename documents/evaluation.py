import logging
import json
from datetime import datetime
from pathlib import Path

from .models import Document, DocumentChunk
from .rag.retrieval_router import retrieve
from .rag.answer_service import generate_answer

logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    {
        "id": "who_is_mahyar",
        "question": "who is mahyar",
        "expected_keywords": ["mahyar", "computer engineering", "sharif"],
        "category": "person",
        "difficulty": "easy",
    },
    {
        "id": "programming_languages",
        "question": "what programming languages",
        "expected_keywords": ["python", "java", "c"],
        "category": "skills",
        "difficulty": "easy",
    },
    {
        "id": "projects",
        "question": "what projects",
        "expected_keywords": ["sgit", "gwent", "llm"],
        "category": "projects",
        "difficulty": "easy",
    },
    {
        "id": "education",
        "question": "where does mahyar study",
        "expected_keywords": ["sharif", "university", "tehran"],
        "category": "education",
        "difficulty": "easy",
    },
    {
        "id": "skills",
        "question": "what are mahyar's technical skills",
        "expected_keywords": ["python", "django", "docker", "langchain"],
        "category": "skills",
        "difficulty": "medium",
    },
    {
        "id": "rag_project",
        "question": "tell me about the LLM Document Q&A project",
        "expected_keywords": ["rag", "django", "bm25", "openrouter"],
        "category": "projects",
        "difficulty": "medium",
    },
    {
        "id": "sgit_project",
        "question": "what is sgit",
        "expected_keywords": ["git", "version control", "c", "linux"],
        "category": "projects",
        "difficulty": "medium",
    },
    {
        "id": "gwent_project",
        "question": "what is gwent online",
        "expected_keywords": ["card game", "java", "multiplayer", "tcp"],
        "category": "projects",
        "difficulty": "medium",
    },
    {
        "id": "contact_info",
        "question": "how to contact mahyar",
        "expected_keywords": ["email", "gmail", "phone"],
        "category": "contact",
        "difficulty": "easy",
    },
    {
        "id": "coursework",
        "question": "what courses has mahyar taken",
        "expected_keywords": ["programming", "algebra", "algorithms", "architecture"],
        "category": "education",
        "difficulty": "medium",
    },
]


def evaluate_retrieval(search_method="hybrid", document_ids=None):
    results = []

    for test in EVAL_QUESTIONS:
        question = test["question"]
        expected_keywords = [k.lower() for k in test["expected_keywords"]]

        chunks = retrieve(
            query=question,
            search_method=search_method,
            limit=3,
            document_ids=document_ids,
        )

        retrieved_text = " ".join([c.content.lower() for c in chunks])

        keyword_hits = sum(1 for kw in expected_keywords if kw in retrieved_text)
        keyword_recall = keyword_hits / len(expected_keywords) if expected_keywords else 0

        has_relevant = keyword_recall > 0

        results.append({
            "id": test["id"],
            "question": question,
            "category": test["category"],
            "difficulty": test["difficulty"],
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

    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "success": 0}
        by_category[cat]["total"] += 1
        if r["has_relevant"]:
            by_category[cat]["success"] += 1

    by_difficulty = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "success": 0}
        by_difficulty[diff]["total"] += 1
        if r["has_relevant"]:
            by_difficulty[diff]["success"] += 1

    return {
        "results": results,
        "summary": {
            "total_questions": len(results),
            "avg_keyword_recall": round(total_recall, 2),
            "success_rate": round(success_rate, 2),
            "search_method": search_method,
            "by_category": {
                cat: {"success_rate": round(v["success"] / v["total"], 2)}
                for cat, v in by_category.items()
            },
            "by_difficulty": {
                diff: {"success_rate": round(v["success"] / v["total"], 2)}
                for diff, v in by_difficulty.items()
            },
        },
    }


def evaluate_answer_quality(search_method="hybrid", document_ids=None):
    results = []

    for test in EVAL_QUESTIONS[:5]:
        question = test["question"]
        expected_keywords = [k.lower() for k in test["expected_keywords"]]

        response = generate_answer(
            question=question,
            search_method=search_method,
            document_ids=document_ids,
        )

        answer_lower = response.get("answer", "").lower()
        keyword_hits = sum(1 for kw in expected_keywords if kw in answer_lower)
        keyword_recall = keyword_hits / len(expected_keywords) if expected_keywords else 0

        results.append({
            "id": test["id"],
            "question": question,
            "answer_preview": response.get("answer", "")[:200],
            "keyword_recall": round(keyword_recall, 2),
            "has_citations": len(response.get("citations", [])) > 0,
            "verification_passed": response.get("verification", {}).get("is_valid", False),
        })

    return {
        "results": results,
        "summary": {
            "total_questions": len(results),
            "avg_keyword_recall": round(
                sum(r["keyword_recall"] for r in results) / len(results), 2
            ) if results else 0,
            "citation_rate": round(
                sum(1 for r in results if r["has_citations"]) / len(results), 2
            ) if results else 0,
            "verification_rate": round(
                sum(1 for r in results if r["verification_passed"]) / len(results), 2
            ) if results else 0,
        },
    }


def compare_search_methods(document_ids=None):
    methods = ["simple", "bm25", "hybrid"]
    comparison = {}

    for method in methods:
        result = evaluate_retrieval(search_method=method, document_ids=document_ids)
        comparison[method] = result["summary"]

    return comparison


def run_regression_suite(document_ids=None, output_dir=None):
    logger.info("Running regression evaluation suite")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "retrieval": {},
        "answer_quality": {},
    }

    for method in ["bm25", "vector", "hybrid"]:
        results["retrieval"][method] = evaluate_retrieval(
            search_method=method,
            document_ids=document_ids,
        )["summary"]

    results["answer_quality"] = evaluate_answer_quality(
        search_method="hybrid",
        document_ids=document_ids,
    )["summary"]

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Regression results saved to {output_path / filename}")

    return results