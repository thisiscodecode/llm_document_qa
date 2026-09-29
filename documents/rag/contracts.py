from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum


@dataclass
class QueryAnalysisResult:
    original: str
    rewritten: str
    language: str
    intent: str
    filters: dict = field(default_factory=dict)


@dataclass
class RetrievalConstraints:
    document_ids: Optional[List[int]] = None
    owner: Any = None
    filters: dict = field(default_factory=dict)
    search_method: str = "hybrid"
    limit: int = 5


@dataclass
class RetrievedChunk:
    id: int
    content: str
    document_id: int
    document_title: str
    chunk_index: int
    page_number: int
    retrieval_score: float = 0.0


@dataclass
class RerankedChunk:
    id: int
    content: str
    document_id: int
    document_title: str
    chunk_index: int
    page_number: int
    rerank_score: float = 0.0
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    position_score: float = 0.0
    retrieval_score: float = 0.0
    rank_delta: int = 0


@dataclass
class BuiltContext:
    context_text: str
    chunk_count: int
    total_tokens: int
    sources: List[dict] = field(default_factory=list)


@dataclass
class PromptPayload:
    prompt_text: str
    question: str
    context: str
    sources_text: str
    language: str = "en"


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class VerificationResult:
    is_valid: bool
    reason: str
    confidence: float = 0.0
    suggestion: Optional[str] = None


@dataclass
class Citation:
    source_number: int
    document_title: str
    chunk_index: int
    page_number: int
    claim_text: str = ""


@dataclass
class CitationMappedAnswer:
    answer: str
    citations: List[Citation]
    citation_count: int = 0
    has_all_citations: bool = True


@dataclass
class IndexingResult:
    success: bool
    document_id: int
    chunks_indexed: int = 0
    index_type: str = "chroma"
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    success: bool
    document_id: int
    chunks_created: int = 0
    pages_extracted: int = 0
    error: Optional[str] = None


@dataclass
class FailureResult:
    mode: str
    message: str
    stage: str
    retryable: bool = False
    fallback_action: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    success: bool
    answer: str
    context: str
    search_method: str
    sources: List[dict]
    citations: List[Citation]
    verification: VerificationResult
    query_analysis: QueryAnalysisResult
    failure: Optional[FailureResult] = None
    request_id: Optional[str] = None
    cache_hit: bool = False
    web_search_used: bool = False
