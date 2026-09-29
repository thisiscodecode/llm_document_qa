"""Corpus-specific, labeled retrieval evaluation.

Each JSONL record needs a question and the IDs of chunks that answer it::

    {"question": "What is the warranty?", "relevant_chunk_ids": [12, 13]}

An optional ``document_ids`` array scopes that case to particular documents.
No external model is needed to score retrieval quality.
"""

import json
import logging
import os
from pathlib import Path
from time import perf_counter

from .models import Document, DocumentChunk
from .rag.retrieval_router import retrieve

logger = logging.getLogger(__name__)

SEARCH_METHODS = ("simple", "bm25", "vector", "hybrid")


def _positive_ids(value, field, line_number):
    if not isinstance(value, list) or not value:
        raise ValueError(f"Line {line_number}: {field} must be a non-empty array")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"Line {line_number}: {field} must contain positive integer IDs")
    return sorted(set(value))


def load_evaluation_cases(dataset_path=None):
    """Read and validate a JSONL set of labeled questions."""
    dataset_path = dataset_path or os.getenv("RAG_EVAL_DATASET")
    if not dataset_path:
        raise ValueError("Set RAG_EVAL_DATASET to a labeled JSONL file")

    cases = []
    with Path(dataset_path).open(encoding="utf-8") as dataset:
        for line_number, raw_line in enumerate(dataset, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number}: question must be non-empty text")
            relevant_ids = _positive_ids(
                item.get("relevant_chunk_ids"), "relevant_chunk_ids", line_number
            )
            case = {
                "question": question.strip(),
                "relevant_chunk_ids": relevant_ids,
            }
            if "document_ids" in item:
                case["document_ids"] = _positive_ids(
                    item["document_ids"], "document_ids", line_number
                )
            cases.append(case)

    if not cases:
        raise ValueError("Evaluation dataset contains no labeled questions")
    return cases


def _summary(results, method, k):
    count = len(results)
    return {
        "search_method": method,
        "total_questions": count,
        "k": k,
        "hit_rate_at_k": round(sum(row["hit_at_k"] for row in results) / count, 4),
        "recall_at_k": round(sum(row["recall_at_k"] for row in results) / count, 4),
        "precision_at_k": round(sum(row["precision_at_k"] for row in results) / count, 4),
        "mrr_at_k": round(sum(row["reciprocal_rank"] for row in results) / count, 4),
        "mean_latency_ms": round(sum(row["latency_ms"] for row in results) / count, 1),
    }


def evaluate_retrieval(
    search_method="hybrid", document_ids=None, owner=None, dataset_path=None,
    k=5, cases=None,
):
    if search_method not in SEARCH_METHODS:
        raise ValueError(f"Unknown search method: {search_method}")
    if type(k) is not int or k < 1 or k > 100:
        raise ValueError("k must be an integer from 1 to 100")
    if cases is None:
        cases = load_evaluation_cases(dataset_path)
    if not cases:
        raise ValueError("Evaluation requires at least one labeled question")

    ready_documents = Document.objects.filter(owner=owner, status="ready")
    accessible_ids = set(ready_documents.values_list("id", flat=True))
    selected_ids = set(document_ids) if document_ids is not None else accessible_ids
    if not selected_ids <= accessible_ids:
        raise ValueError("Selected documents must be ready and accessible to the evaluator")

    relevant_ids = {
        chunk_id for case in cases for chunk_id in case["relevant_chunk_ids"]
    }
    labeled_chunks = dict(DocumentChunk.objects.filter(
        id__in=relevant_ids, document_id__in=accessible_ids
    ).values_list("id", "document_id"))
    if missing := relevant_ids - labeled_chunks.keys():
        raise ValueError(
            "Evaluation labels reference missing, unready, or inaccessible chunks: "
            + ", ".join(map(str, sorted(missing)))
        )

    results = []
    for case in cases:
        case_labels = set(case["relevant_chunk_ids"])
        label_document_ids = {labeled_chunks[chunk_id] for chunk_id in case_labels}
        declared_ids = set(case.get("document_ids", accessible_ids))
        if not label_document_ids <= declared_ids:
            raise ValueError(
                f"Evaluation case {case['question']!r} excludes one of its relevant chunks"
            )
        # A partly excluded ground-truth set would make recall incomparable.
        if not label_document_ids <= selected_ids:
            continue
        case_ids = sorted(declared_ids & selected_ids)
        if not case_ids:
            continue

        started = perf_counter()
        chunks = retrieve(
            query=case["question"], search_method=search_method, limit=k,
            document_ids=case_ids, owner=owner,
        )
        latency_ms = (perf_counter() - started) * 1000
        ranked_ids = [chunk.id for chunk in chunks]
        relevant = case_labels
        hits = [index + 1 for index, chunk_id in enumerate(ranked_ids) if chunk_id in relevant]
        results.append({
            "question": case["question"],
            "retrieved_chunk_ids": ranked_ids,
            "relevant_chunk_ids": sorted(relevant),
            "hit_at_k": int(bool(hits)),
            "recall_at_k": len(hits) / len(relevant),
            "precision_at_k": len(hits) / k,
            "reciprocal_rank": 1 / hits[0] if hits else 0.0,
            "latency_ms": round(latency_ms, 1),
        })

    if not results:
        raise ValueError("No evaluation cases apply to the selected documents")
    return {
        "results": results,
        "summary": _summary(results, search_method, k),
        "skipped_cases": len(cases) - len(results),
    }


def compare_search_methods(document_ids=None, owner=None, dataset_path=None, k=5):
    cases = load_evaluation_cases(dataset_path)
    return {
        method: evaluate_retrieval(
            search_method=method, document_ids=document_ids, owner=owner,
            k=k, cases=cases,
        )["summary"]
        for method in SEARCH_METHODS
    }


def run_regression_suite(document_ids=None, output_dir=None, owner=None,
                         dataset_path=None, k=5):
    """Run the same labeled cases against every retrieval strategy."""
    from datetime import datetime, timezone

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval": compare_search_methods(
            document_ids=document_ids, owner=owner, dataset_path=dataset_path, k=k
        ),
    }
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"regression_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
        (output_path / filename).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Retrieval evaluation saved to %s", output_path / filename)
    return results
