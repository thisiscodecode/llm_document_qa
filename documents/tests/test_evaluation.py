import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from documents.evaluation import evaluate_retrieval, load_evaluation_cases
from documents.models import Document, DocumentChunk


class RetrievalEvaluationTests(TestCase):
    def setUp(self):
        self.document = Document.objects.create(
            title="A", file="documents/a.pdf", status="ready"
        )
        self.chunks = [DocumentChunk.objects.create(
            document=self.document, content=f"chunk {index}", chunk_index=index
        ) for index in range(4)]

    def test_labeled_metrics_use_relevant_chunk_ids_and_rank(self):
        cases = [
            {"question": "first", "relevant_chunk_ids": [self.chunks[1].id, self.chunks[2].id]},
            {"question": "second", "relevant_chunk_ids": [self.chunks[3].id]},
        ]
        with patch("documents.evaluation.retrieve") as retrieve:
            retrieve.side_effect = [
                [SimpleNamespace(id=self.chunks[0].id), SimpleNamespace(id=self.chunks[1].id)],
                [SimpleNamespace(id=self.chunks[3].id), SimpleNamespace(id=self.chunks[0].id)],
            ]
            report = evaluate_retrieval(
                search_method="hybrid", cases=cases, document_ids=[self.document.id], k=2
            )

        summary = report["summary"]
        self.assertEqual(summary["total_questions"], 2)
        self.assertEqual(summary["hit_rate_at_k"], 1.0)
        self.assertEqual(summary["recall_at_k"], 0.75)
        self.assertEqual(summary["precision_at_k"], 0.5)
        self.assertEqual(summary["mrr_at_k"], 0.75)
        self.assertEqual(retrieve.call_args.kwargs["document_ids"], [self.document.id])

    def test_partly_selected_ground_truth_case_is_skipped(self):
        other = Document.objects.create(
            title="B", file="documents/b.pdf", status="ready"
        )
        other_chunk = DocumentChunk.objects.create(
            document=other, content="another", chunk_index=0
        )
        cases = [
            {"question": "both", "relevant_chunk_ids": [self.chunks[0].id, other_chunk.id]},
            {"question": "only A", "relevant_chunk_ids": [self.chunks[1].id]},
        ]
        with patch("documents.evaluation.retrieve", return_value=[self.chunks[1]]) as retrieve:
            report = evaluate_retrieval(cases=cases, document_ids=[self.document.id])
        self.assertEqual(report["summary"]["total_questions"], 1)
        self.assertEqual(report["skipped_cases"], 1)
        self.assertEqual(retrieve.call_count, 1)

    def test_stale_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing, unready, or inaccessible"):
            evaluate_retrieval(cases=[{
                "question": "stale", "relevant_chunk_ids": [self.chunks[-1].id + 1000]
            }])

    def test_case_document_scope_must_include_its_labels(self):
        other = Document.objects.create(title="B", file="documents/b.pdf", status="ready")
        with self.assertRaisesRegex(ValueError, "excludes one of its relevant chunks"):
            evaluate_retrieval(cases=[{
                "question": "wrong scope", "relevant_chunk_ids": [self.chunks[0].id],
                "document_ids": [other.id],
            }])


class EvaluationDatasetTests(SimpleTestCase):

    def test_jsonl_requires_valid_labels(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({
                "question": "sample", "relevant_chunk_ids": [3, 3, 1],
                "document_ids": [9],
            }) + "\n", encoding="utf-8")
            self.assertEqual(load_evaluation_cases(path)[0]["relevant_chunk_ids"], [1, 3])

            path.write_text(json.dumps({
                "question": "sample", "relevant_chunk_ids": [True],
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_evaluation_cases(path)
