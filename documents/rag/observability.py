import time
import logging
from functools import wraps
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    parent: Optional['TraceSpan'] = None
    children: list = field(default_factory=list)
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def finish(self):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self
    
    def add_metadata(self, key, value):
        self.metadata[key] = value
        return self
    
    def to_dict(self):
        return {
            'name': self.name,
            'duration_ms': round(self.duration_ms, 2),
            'metadata': self.metadata,
            'children': [c.to_dict() for c in self.children],
        }


class Tracer:
    def __init__(self):
        self._spans = []
        self._current_span = None
    
    def start_span(self, name, metadata=None):
        span = TraceSpan(name=name, parent=self._current_span)
        span.start()
        if metadata:
            span.metadata = metadata
        
        if self._current_span:
            self._current_span.children.append(span)
        
        self._current_span = span
        self._spans.append(span)
        return span
    
    def finish_span(self, span):
        span.finish()
        if span.parent:
            self._current_span = span.parent
        else:
            self._current_span = None
        return span
    
    def get_root_spans(self):
        return [s for s in self._spans if s.parent is None]
    
    def get_summary(self):
        root_spans = self.get_root_spans()
        if not root_spans:
            return {}
        
        total_duration = sum(s.duration_ms for s in root_spans)
        return {
            'total_duration_ms': round(total_duration, 2),
            'spans': [s.to_dict() for s in root_spans],
        }


_tracer = Tracer()


def get_tracer():
    return _tracer


def trace(name, metadata=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span = _tracer.start_span(name, metadata)
            try:
                result = func(*args, **kwargs)
                span.add_metadata('success', True)
                return result
            except Exception as e:
                span.add_metadata('error', str(e))
                span.add_metadata('success', False)
                raise
            finally:
                _tracer.finish_span(span)
        return wrapper
    return decorator


class Metrics:
    def __init__(self):
        self._counters = {}
        self._timers = {}
    
    def increment(self, name, value=1):
        self._counters[name] = self._counters.get(name, 0) + value
    
    def record_timing(self, name, duration_ms):
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(duration_ms)
    
    def get_counter(self, name):
        return self._counters.get(name, 0)
    
    def get_timing_stats(self, name):
        timings = self._timers.get(name, [])
        if not timings:
            return None
        
        return {
            'count': len(timings),
            'min_ms': round(min(timings), 2),
            'max_ms': round(max(timings), 2),
            'avg_ms': round(sum(timings) / len(timings), 2),
        }
    
    def get_summary(self):
        return {
            'counters': self._counters,
            'timers': {
                name: self.get_timing_stats(name)
                for name in self._timers
            },
        }


_metrics = Metrics()


def get_metrics():
    return _metrics