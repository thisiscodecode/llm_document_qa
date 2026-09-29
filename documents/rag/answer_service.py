import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from .query_analyzer import analyze_query
from .retrieval_router import retrieve
from .reranker import rerank
from .context_builder import build_context, build_sources
from .prompt_builder import build_grounded_prompt
from .answer_verifier import verify_answer
from .citation_builder import format_sources, map_claims_to_sources
from .llm_client import invoke_llm
from .observability import get_tracer, get_metrics
from .failure_handler import FailureMode, FAILURE_RESPONSES as FAILURE_MESSAGES

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    question: str
    search_method: str = "hybrid"
    document_ids: Optional[list] = None
    owner: object = None
    request_id: str = ""
    
    query_analysis: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    reranked_chunks: list = field(default_factory=list)
    context: str = ""
    sources: list = field(default_factory=list)
    sources_text: str = ""
    prompt: str = ""
    llm_response: str = ""
    answer: str = ""
    verification: dict = field(default_factory=dict)
    citations: list = field(default_factory=list)
    cache_hit: bool = False
    web_search_used: bool = False
    
    error: Optional[str] = None
    failure_mode: Optional[FailureMode] = None
    stage: str = "init"


def _get_failure_message(mode: FailureMode) -> str:
    entry = FAILURE_MESSAGES.get(mode, {})
    if isinstance(entry, dict):
        return entry.get("message", str(mode.value))
    return str(entry)


class RAGPipeline:
    def __init__(self):
        self.stages = [
            ("analyze_query", self._analyze_query),
            ("check_cache", self._check_cache),
            ("retrieve", self._retrieve),
            ("web_search_fallback", self._web_search_fallback),
            ("rerank", self._rerank),
            ("build_context", self._build_context),
            ("build_prompt", self._build_prompt),
            ("generate", self._generate),
            ("verify", self._verify),
            ("attach_citations", self._attach_citations),
            ("cache_result", self._cache_result),
        ]
    
    def run(self, question: str, search_method: str = "hybrid", 
            document_ids: list = None, owner=None, request_id: str = None) -> dict:
        tracer = get_tracer()
        metrics = get_metrics()
        
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        root_span = tracer.start_span("rag_pipeline", {
            "question_length": len(question),
            "search_method": search_method,
            "request_id": request_id,
        })
        metrics.increment("pipeline_requests")
        
        state = PipelineState(
            question=question,
            search_method=search_method,
            document_ids=document_ids,
            owner=owner,
            request_id=request_id,
        )
        
        for stage_name, stage_fn in self.stages:
            state.stage = stage_name
            
            stage_span = tracer.start_span(stage_name)
            
            try:
                state = stage_fn(state)
                stage_span.add_metadata("success", True)
            except Exception as e:
                logger.error(f"[{request_id}] Pipeline failed at {stage_name}: {e}")
                state.error = str(e)
                state.failure_mode = FailureMode.LLM_ERROR
                stage_span.add_metadata("error", str(e))
                stage_span.add_metadata("success", False)
                tracer.finish_span(stage_span)
                tracer.finish_span(root_span)
                metrics.increment("pipeline_errors")
                return self._build_error_response(state)
            finally:
                tracer.finish_span(stage_span)
            
            if state.error and stage_name not in ("check_cache", "web_search_fallback"):
                tracer.finish_span(root_span)
                metrics.increment(f"pipeline_failure_{state.failure_mode.value if state.failure_mode else 'unknown'}")
                return self._build_error_response(state)
        
        tracer.finish_span(root_span)
        metrics.increment("pipeline_success")
        
        return self._build_success_response(state)
    
    def _analyze_query(self, state: PipelineState) -> PipelineState:
        state.query_analysis = analyze_query(state.question)
        logger.info(f"[{state.request_id}] Query analysis: lang={state.query_analysis['language']}, "
                    f"intent={state.query_analysis['intent']}")
        return state
    
    def _check_cache(self, state: PipelineState) -> PipelineState:
        from .cache import get_cached_query
        from .feature_flags import is_enabled
        
        if not is_enabled('advanced_caching'):
            return state
        
        doc_ids = tuple(state.document_ids or [])
        cached = get_cached_query(state.question, state.search_method, doc_ids)
        if cached is not None:
            state.answer = cached.get('answer', '')
            state.context = cached.get('context', '')
            state.sources = cached.get('sources', [])
            state.citations = cached.get('citations', [])
            state.verification = cached.get('verification', {'is_valid': True, 'reason': 'cached'})
            state.query_analysis = cached.get('query_analysis', state.query_analysis)
            state.cache_hit = True
            state.error = "cached"
            logger.info(f"[{state.request_id}] Cache hit for query")
        return state
    
    def _retrieve(self, state: PipelineState) -> PipelineState:
        if state.cache_hit:
            return state
        
        state.chunks = retrieve(
            query=state.query_analysis.get('rewritten', state.question),
            search_method=state.search_method,
            limit=10,
            document_ids=state.document_ids,
            owner=state.owner,
        )
        
        if not state.chunks:
            logger.warning(f"[{state.request_id}] No chunks retrieved")
        
        logger.info(f"[{state.request_id}] Retrieved {len(state.chunks)} chunks")
        return state
    
    def _web_search_fallback(self, state: PipelineState) -> PipelineState:
        if state.cache_hit or state.chunks:
            return state
        
        from .feature_flags import is_enabled
        if not is_enabled('web_search_fallback'):
            state.error = FailureMode.NO_CHUNKS_RETRIEVED.value
            state.failure_mode = FailureMode.NO_CHUNKS_RETRIEVED
            state.answer = _get_failure_message(FailureMode.NO_CHUNKS_RETRIEVED)
            return state
        
        from .web_search import web_search, format_web_results, build_web_search_prompt
        metrics = get_metrics()
        results = web_search(state.question)
        if results:
            metrics.increment("web_search_hits")
            web_text = format_web_results(results)
            state.prompt = build_web_search_prompt(state.question, web_text)
            state.web_search_used = True
            state.context = web_text
            state.sources = [{'index': i + 1, 'document': r['title'], 'chunk_index': 0,
                             'page': 1, 'score': 0, 'preview': r['snippet'][:200]}
                            for i, r in enumerate(results)]
            logger.info(f"[{state.request_id}] Web search returned {len(results)} results")
        else:
            metrics.increment("web_search_misses")
            state.error = FailureMode.NO_CHUNKS_RETRIEVED.value
            state.failure_mode = FailureMode.NO_CHUNKS_RETRIEVED
            state.answer = _get_failure_message(FailureMode.NO_CHUNKS_RETRIEVED)
        return state
    
    def _rerank(self, state: PipelineState) -> PipelineState:
        if state.cache_hit or state.web_search_used:
            return state
        if not state.chunks:
            return state
        
        state.reranked_chunks = rerank(
            state.question, 
            state.chunks, 
            limit=5
        )
        logger.info(f"[{state.request_id}] Reranked to {len(state.reranked_chunks)} chunks")
        return state
    
    def _build_context(self, state: PipelineState) -> PipelineState:
        if state.cache_hit or state.web_search_used:
            return state
        
        state.context = build_context(state.reranked_chunks)
        state.sources = build_sources(state.reranked_chunks)
        
        if not state.context:
            logger.warning(f"[{state.request_id}] No context built from chunks")
            state.error = FailureMode.NO_CONTEXT.value
            state.failure_mode = FailureMode.NO_CONTEXT
            state.answer = _get_failure_message(FailureMode.NO_CONTEXT)
        
        return state
    
    def _build_prompt(self, state: PipelineState) -> PipelineState:
        if state.cache_hit:
            return state
        if state.web_search_used and state.prompt:
            return state
        
        state.sources_text = format_sources(state.sources)
        state.prompt = build_grounded_prompt(
            state.question, 
            state.context, 
            state.sources_text
        )
        return state
    
    def _generate(self, state: PipelineState) -> PipelineState:
        if state.cache_hit:
            return state
        
        content, error = invoke_llm(state.prompt)
        if content is None:
            logger.error(f"[{state.request_id}] LLM invocation failed: {error}")
            state.error = error or "llm_error"
            state.failure_mode = FailureMode.LLM_UNAVAILABLE if error == "llm_unavailable" else FailureMode.LLM_ERROR
            state.answer = _get_failure_message(state.failure_mode)
            return state
        
        state.llm_response = content
        state.answer = content
        return state
    
    def _verify(self, state: PipelineState) -> PipelineState:
        if state.cache_hit or state.web_search_used:
            return state
        
        state.verification = verify_answer(state.answer, state.context)
        
        if not state.verification['is_valid']:
            logger.warning(f"[{state.request_id}] Answer verification failed: {state.verification['reason']}")
            
            if state.verification['reason'] == 'low_support':
                state.error = FailureMode.VERIFICATION_FAILED.value
                state.failure_mode = FailureMode.VERIFICATION_FAILED
                state.answer = _get_failure_message(FailureMode.VERIFICATION_FAILED)
        
        return state
    
    def _attach_citations(self, state: PipelineState) -> PipelineState:
        if state.cache_hit:
            return state
        
        state.citations = map_claims_to_sources(state.answer, state.sources)
        
        if state.sources and not state.citations:
            logger.warning(f"[{state.request_id}] No citations mapped despite having sources")
        
        logger.info(f"[{state.request_id}] Attached {len(state.citations)} citations")
        return state
    
    def _cache_result(self, state: PipelineState) -> PipelineState:
        if state.cache_hit or state.error:
            return state
        
        from .cache import set_cached_query
        from .feature_flags import is_enabled
        
        if is_enabled('advanced_caching'):
            doc_ids = tuple(state.document_ids or [])
            set_cached_query(state.question, state.search_method, doc_ids, {
                'answer': state.answer,
                'context': state.context,
                'sources': state.sources,
                'citations': state.citations,
                'verification': state.verification,
                'query_analysis': state.query_analysis,
            })
        
        return state
    
    def _build_success_response(self, state: PipelineState) -> dict:
        return {
            "answer": state.answer,
            "context": state.context,
            "search_method": state.search_method,
            "sources": state.sources,
            "citations": state.citations,
            "verification": state.verification,
            "query_analysis": state.query_analysis,
            "request_id": state.request_id,
            "cache_hit": state.cache_hit,
            "web_search_used": state.web_search_used,
        }
    
    def _build_error_response(self, state: PipelineState) -> dict:
        return {
            "answer": state.answer or "An error occurred processing your request.",
            "context": state.context,
            "search_method": state.search_method,
            "sources": state.sources,
            "citations": [],
            "verification": {"is_valid": False, "reason": state.failure_mode.value if state.failure_mode else state.error},
            "query_analysis": state.query_analysis,
            "error": state.error,
            "failure_mode": state.failure_mode.value if state.failure_mode else None,
            "request_id": state.request_id,
            "cache_hit": state.cache_hit,
            "web_search_used": state.web_search_used,
        }


_pipeline = RAGPipeline()


def generate_answer(question, search_method="hybrid", document_ids=None, owner=None, request_id=None):
    return _pipeline.run(
        question=question,
        search_method=search_method,
        document_ids=document_ids,
        owner=owner,
        request_id=request_id,
    )
