"""Run a labeled retrieval benchmark against the current corpus."""

import json

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from documents.evaluation import (
    SEARCH_METHODS,
    compare_search_methods,
    evaluate_retrieval,
)


class Command(BaseCommand):
    help = "Evaluate retrieval from a labeled JSONL dataset (no LLM calls)"

    def add_arguments(self, parser):
        parser.add_argument("--dataset", required=True, help="Labeled JSONL file")
        parser.add_argument("--method", choices=SEARCH_METHODS,
                            help="Evaluate one method; default compares all")
        parser.add_argument("--k", type=int, default=5, help="Top-k cutoff (1-100)")
        parser.add_argument("--username", help="Evaluate a user's accessible corpus")
        parser.add_argument("--document-id", type=int, action="append", dest="document_ids",
                            help="Restrict to this document ID; can be repeated")

    def handle(self, *args, **options):
        owner = None
        if options["username"]:
            try:
                owner = get_user_model().objects.get(username=options["username"])
            except get_user_model().DoesNotExist as exc:
                raise CommandError("Username not found") from exc

        try:
            kwargs = {
                "document_ids": options["document_ids"],
                "owner": owner,
                "dataset_path": options["dataset"],
                "k": options["k"],
            }
            if options["method"]:
                report = evaluate_retrieval(search_method=options["method"], **kwargs)
            else:
                report = compare_search_methods(**kwargs)
        except (ValueError, OSError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(json.dumps(report, ensure_ascii=False, indent=2))
