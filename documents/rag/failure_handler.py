import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    NO_CHUNKS_RETRIEVED = "no_chunks_retrieved"
    NO_CONTEXT = "no_context"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_ERROR = "llm_error"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_FAILED_LOW_SUPPORT = "verification_low_support"
    VERIFICATION_FAILED_NO_CONTEXT = "verification_no_context"
    CITATION_MAPPING_FAILED = "citation_mapping_failed"
    EMBEDDING_FAILED = "embedding_failed"
    INDEXING_FAILED = "indexing_failed"
    PROCESSING_FAILED = "processing_failed"


@dataclass
class FailureResult:
    mode: FailureMode
    message: str
    stage: str
    retryable: bool = False
    fallback_action: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


FAILURE_RESPONSES = {
    FailureMode.NO_CHUNKS_RETRIEVED: {
        "message": "I could not find any relevant information in the uploaded documents.",
        "retryable": False,
        "fallback": None,
    },
    FailureMode.NO_CONTEXT: {
        "message": "I could not find relevant information in the uploaded documents.",
        "retryable": False,
        "fallback": None,
    },
    FailureMode.LLM_UNAVAILABLE: {
        "message": "The language model is currently unavailable. Please try again later.",
        "retryable": True,
        "fallback": "retry_with_backoff",
    },
    FailureMode.LLM_ERROR: {
        "message": "An error occurred while generating the answer. Please try again.",
        "retryable": True,
        "fallback": "retry_once",
    },
    FailureMode.VERIFICATION_FAILED: {
        "message": "I cannot provide a reliable answer based on the available information.",
        "retryable": False,
        "fallback": "refuse_answer",
    },
    FailureMode.VERIFICATION_FAILED_LOW_SUPPORT: {
        "message": "I cannot provide a reliable answer based on the available information.",
        "retryable": False,
        "fallback": "refuse_answer",
    },
    FailureMode.VERIFICATION_FAILED_NO_CONTEXT: {
        "message": "The answer is not supported by the available context.",
        "retryable": False,
        "fallback": "refuse_answer",
    },
    FailureMode.CITATION_MAPPING_FAILED: {
        "message": "I found some information but cannot properly cite the sources.",
        "retryable": False,
        "fallback": "return_without_citations",
    },
    FailureMode.EMBEDDING_FAILED: {
        "message": "Failed to generate embeddings for the query.",
        "retryable": True,
        "fallback": "use_bm25_fallback",
    },
    FailureMode.INDEXING_FAILED: {
        "message": "Failed to index document chunks.",
        "retryable": True,
        "fallback": "mark_indexing_failed",
    },
    FailureMode.PROCESSING_FAILED: {
        "message": "Document processing failed.",
        "retryable": True,
        "fallback": "mark_processing_failed",
    },
}


def handle_no_chunks_retrieved(query: str, document_ids: list = None) -> FailureResult:
    logger.warning(f"No chunks retrieved for query: {query[:50]}...")
    return FailureResult(
        mode=FailureMode.NO_CHUNKS_RETRIEVED,
        message=FAILURE_RESPONSES[FailureMode.NO_CHUNKS_RETRIEVED]["message"],
        stage="retrieve",
        retryable=False,
    )


def handle_no_context(chunks: list) -> FailureResult:
    logger.warning(f"No context built from {len(chunks)} chunks")
    return FailureResult(
        mode=FailureMode.NO_CONTEXT,
        message=FAILURE_RESPONSES[FailureMode.NO_CONTEXT]["message"],
        stage="build_context",
        retryable=False,
    )


def handle_llm_unavailable() -> FailureResult:
    logger.error("LLM client unavailable")
    return FailureResult(
        mode=FailureMode.LLM_UNAVAILABLE,
        message=FAILURE_RESPONSES[FailureMode.LLM_UNAVAILABLE]["message"],
        stage="generate",
        retryable=True,
        fallback_action="retry_with_backoff",
    )


def handle_llm_error(error: Exception) -> FailureResult:
    logger.error(f"LLM invocation failed: {error}")
    return FailureResult(
        mode=FailureMode.LLM_ERROR,
        message=FAILURE_RESPONSES[FailureMode.LLM_ERROR]["message"],
        stage="generate",
        retryable=True,
        fallback_action="retry_once",
        metadata={"error": str(error)},
    )


def handle_verification_failed(verification: dict) -> FailureResult:
    reason = verification.get("reason", "unknown")

    if reason == "low_support":
        mode = FailureMode.VERIFICATION_FAILED_LOW_SUPPORT
    else:
        mode = FailureMode.VERIFICATION_FAILED_NO_CONTEXT

    logger.warning(f"Answer verification failed: {reason}")
    return FailureResult(
        mode=mode,
        message=FAILURE_RESPONSES[mode]["message"],
        stage="verify",
        retryable=False,
        fallback_action="refuse_answer",
        metadata={"verification_reason": reason},
    )


def handle_citation_failed(answer: str, sources: list) -> FailureResult:
    logger.warning("Citation mapping failed despite having sources")
    return FailureResult(
        mode=FailureMode.CITATION_MAPPING_FAILED,
        message=FAILURE_RESPONSES[FailureMode.CITATION_MAPPING_FAILED]["message"],
        stage="attach_citations",
        retryable=False,
        fallback_action="return_without_citations",
    )


def handle_embedding_error(error: Exception) -> FailureResult:
    logger.error(f"Embedding generation failed: {error}")
    return FailureResult(
        mode=FailureMode.EMBEDDING_FAILED,
        message=FAILURE_RESPONSES[FailureMode.EMBEDDING_FAILED]["message"],
        stage="embedding",
        retryable=True,
        fallback_action="use_bm25_fallback",
        metadata={"error": str(error)},
    )


def handle_indexing_error(document_id: int, error: Exception) -> FailureResult:
    logger.error(f"Indexing failed for document {document_id}: {error}")
    return FailureResult(
        mode=FailureMode.INDEXING_FAILED,
        message=FAILURE_RESPONSES[FailureMode.INDEXING_FAILED]["message"],
        stage="indexing",
        retryable=True,
        fallback_action="mark_indexing_failed",
        metadata={"document_id": document_id, "error": str(error)},
    )


def handle_processing_error(document_id: int, error: Exception) -> FailureResult:
    logger.error(f"Processing failed for document {document_id}: {error}")
    return FailureResult(
        mode=FailureMode.PROCESSING_FAILED,
        message=FAILURE_RESPONSES[FailureMode.PROCESSING_FAILED]["message"],
        stage="processing",
        retryable=True,
        fallback_action="mark_processing_failed",
        metadata={"document_id": document_id, "error": str(error)},
    )


def build_failure_response(failure: FailureResult) -> dict:
    return {
        "answer": failure.message,
        "context": "",
        "sources": [],
        "citations": [],
        "verification": {"is_valid": False, "reason": failure.mode.value},
        "failure": {
            "mode": failure.mode.value,
            "stage": failure.stage,
            "retryable": failure.retryable,
            "fallback": failure.fallback_action,
        },
    }