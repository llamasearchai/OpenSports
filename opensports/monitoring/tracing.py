"""
Distributed Tracing System

Advanced distributed tracing using OpenTelemetry for comprehensive
observability across the OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
import json
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opensports.core.config import settings
from opensports.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpanContext:
    """Context information for a span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    tags: Dict[str, Any] = None
    start_time: float = 0
    end_time: Optional[float] = None


class TracingManager:
    """
    Advanced distributed tracing manager.
    
    Features:
    - OpenTelemetry integration
    - Jaeger exporter for trace visualization
    - Custom span creation and management
    - Automatic instrumentation for common libraries
    - Performance monitoring and analysis
    """
    
    def __init__(self):
        self.tracer_provider = None
        self.tracer = None
        self.active_spans = {}
        self.span_processors = []
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the tracing system."""
        logger.info("Initializing distributed tracing")
        
        # Set up tracer provider
        self.tracer_provider = TracerProvider()
        trace.set_tracer_provider(self.tracer_provider)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=getattr(settings, 'JAEGER_HOST', 'localhost'),
            agent_port=getattr(settings, 'JAEGER_PORT', 6831),
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        self.span_processors.append(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Set up propagation
        set_global_textmap(B3MultiFormat())
        
        # Auto-instrument common libraries
        self._setup_auto_instrumentation()
        
        self.is_initialized = True
        logger.info("Distributed tracing initialized successfully")
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        try:
            # Instrument FastAPI
            FastAPIInstrumentor.instrument()
            
            # Instrument requests
            RequestsInstrumentor().instrument()
            
            # Instrument Redis
            RedisInstrumentor().instrument()
            
            logger.info("Auto-instrumentation configured")
        except Exception as e:
            logger.warning(f"Some auto-instrumentation failed: {e}")
    
    @contextmanager
    def create_span(
        self,
        operation_name: str,
        tags: Dict[str, Any] = None,
        parent_span: Optional[trace.Span] = None
    ):
        """
        Create a new span with context management.
        
        Args:
            operation_name: Name of the operation
            tags: Additional tags for the span
            parent_span: Parent span for nested operations
            
        Yields:
            The created span
        """
        if not self.is_initialized:
            self.initialize()
        
        # Create span
        span = self.tracer.start_span(operation_name, parent=parent_span)
        
        # Add tags
        if tags:
            for key, value in tags.items():
                span.set_attribute(key, str(value))
        
        # Add default attributes
        span.set_attribute("service.name", "opensports")
        span.set_attribute("service.version", "1.0.0")
        span.set_attribute("timestamp", datetime.now().isoformat())
        
        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()
    
    @asynccontextmanager
    async def create_async_span(
        self,
        operation_name: str,
        tags: Dict[str, Any] = None,
        parent_span: Optional[trace.Span] = None
    ):
        """
        Create a new async span with context management.
        
        Args:
            operation_name: Name of the operation
            tags: Additional tags for the span
            parent_span: Parent span for nested operations
            
        Yields:
            The created span
        """
        if not self.is_initialized:
            self.initialize()
        
        # Create span
        span = self.tracer.start_span(operation_name, parent=parent_span)
        
        # Add tags
        if tags:
            for key, value in tags.items():
                span.set_attribute(key, str(value))
        
        # Add default attributes
        span.set_attribute("service.name", "opensports")
        span.set_attribute("service.version", "1.0.0")
        span.set_attribute("timestamp", datetime.now().isoformat())
        
        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()
    
    def trace_function(
        self,
        operation_name: Optional[str] = None,
        tags: Dict[str, Any] = None
    ):
        """
        Decorator to automatically trace function execution.
        
        Args:
            operation_name: Custom operation name (defaults to function name)
            tags: Additional tags for the span
        """
        def decorator(func):
            func_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.create_async_span(func_name, tags) as span:
                        # Add function metadata
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        span.set_attribute("function.args_count", len(args))
                        span.set_attribute("function.kwargs_count", len(kwargs))
                        
                        start_time = time.time()
                        try:
                            result = await func(*args, **kwargs)
                            span.set_attribute("function.success", True)
                            return result
                        except Exception as e:
                            span.set_attribute("function.success", False)
                            span.set_attribute("function.error", str(e))
                            raise
                        finally:
                            duration = time.time() - start_time
                            span.set_attribute("function.duration_seconds", duration)
                
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    with self.create_span(func_name, tags) as span:
                        # Add function metadata
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        span.set_attribute("function.args_count", len(args))
                        span.set_attribute("function.kwargs_count", len(kwargs))
                        
                        start_time = time.time()
                        try:
                            result = func(*args, **kwargs)
                            span.set_attribute("function.success", True)
                            return result
                        except Exception as e:
                            span.set_attribute("function.success", False)
                            span.set_attribute("function.error", str(e))
                            raise
                        finally:
                            duration = time.time() - start_time
                            span.set_attribute("function.duration_seconds", duration)
                
                return sync_wrapper
        
        return decorator
    
    def trace_database_operation(
        self,
        operation: str,
        table: str = None,
        query: str = None
    ):
        """
        Create a span for database operations.
        
        Args:
            operation: Database operation (SELECT, INSERT, UPDATE, DELETE)
            table: Table name
            query: SQL query (will be sanitized)
        """
        tags = {
            "db.operation": operation,
            "db.type": "sqlite",
            "component": "database"
        }
        
        if table:
            tags["db.table"] = table
        
        if query:
            # Sanitize query (remove sensitive data)
            sanitized_query = self._sanitize_query(query)
            tags["db.statement"] = sanitized_query
        
        return self.create_span(f"db.{operation.lower()}", tags)
    
    def trace_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int = None,
        user_id: str = None
    ):
        """
        Create a span for API requests.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            user_id: User identifier
        """
        tags = {
            "http.method": method,
            "http.url": endpoint,
            "component": "api"
        }
        
        if status_code:
            tags["http.status_code"] = status_code
        
        if user_id:
            tags["user.id"] = user_id
        
        return self.create_span(f"http.{method.lower()}", tags)
    
    def trace_ml_operation(
        self,
        operation: str,
        model_name: str = None,
        dataset_size: int = None
    ):
        """
        Create a span for machine learning operations.
        
        Args:
            operation: ML operation (train, predict, evaluate)
            model_name: Name of the model
            dataset_size: Size of the dataset
        """
        tags = {
            "ml.operation": operation,
            "component": "ml"
        }
        
        if model_name:
            tags["ml.model"] = model_name
        
        if dataset_size:
            tags["ml.dataset_size"] = dataset_size
        
        return self.create_span(f"ml.{operation}", tags)
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize SQL query to remove sensitive information."""
        # Simple sanitization - in production, use more sophisticated methods
        sanitized = query
        
        # Remove potential sensitive values
        import re
        sanitized = re.sub(r"'[^']*'", "'?'", sanitized)
        sanitized = re.sub(r'"[^"]*"', '"?"', sanitized)
        sanitized = re.sub(r'\b\d+\b', '?', sanitized)
        
        return sanitized
    
    def get_current_span(self) -> Optional[trace.Span]:
        """Get the currently active span."""
        return trace.get_current_span()
    
    def add_span_event(self, span: trace.Span, name: str, attributes: Dict[str, Any] = None):
        """Add an event to a span."""
        span.add_event(name, attributes or {})
    
    def set_span_tag(self, span: trace.Span, key: str, value: Any):
        """Set a tag on a span."""
        span.set_attribute(key, str(value))
    
    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation."""
        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()
            return {
                "trace_id": format(span_context.trace_id, '032x'),
                "span_id": format(span_context.span_id, '016x'),
                "trace_flags": format(span_context.trace_flags, '02x')
            }
        return {}
    
    def shutdown(self):
        """Shutdown the tracing system."""
        logger.info("Shutting down tracing system")
        
        for processor in self.span_processors:
            processor.shutdown()
        
        self.is_initialized = False


class SpanManager:
    """
    High-level span management utilities.
    """
    
    def __init__(self, tracing_manager: TracingManager):
        self.tracing_manager = tracing_manager
        self.span_stack = []
    
    async def trace_game_analysis(
        self,
        game_id: str,
        analysis_type: str,
        sport: str
    ):
        """Trace a game analysis operation."""
        tags = {
            "game.id": game_id,
            "game.sport": sport,
            "analysis.type": analysis_type,
            "operation": "game_analysis"
        }
        
        return self.tracing_manager.create_async_span(
            "game.analysis",
            tags
        )
    
    async def trace_prediction(
        self,
        prediction_type: str,
        model_name: str,
        sport: str
    ):
        """Trace a prediction operation."""
        tags = {
            "prediction.type": prediction_type,
            "prediction.model": model_name,
            "prediction.sport": sport,
            "operation": "prediction"
        }
        
        return self.tracing_manager.create_async_span(
            "prediction.generate",
            tags
        )
    
    async def trace_data_ingestion(
        self,
        source: str,
        sport: str,
        record_count: int
    ):
        """Trace a data ingestion operation."""
        tags = {
            "ingestion.source": source,
            "ingestion.sport": sport,
            "ingestion.record_count": record_count,
            "operation": "data_ingestion"
        }
        
        return self.tracing_manager.create_async_span(
            "data.ingestion",
            tags
        )
    
    async def trace_model_training(
        self,
        model_name: str,
        sport: str,
        dataset_size: int
    ):
        """Trace a model training operation."""
        tags = {
            "training.model": model_name,
            "training.sport": sport,
            "training.dataset_size": dataset_size,
            "operation": "model_training"
        }
        
        return self.tracing_manager.create_async_span(
            "model.training",
            tags
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from traces."""
        # This would analyze completed spans to extract performance metrics
        # For now, return mock metrics
        return {
            "average_response_time": 0.25,
            "p95_response_time": 0.8,
            "error_rate": 0.02,
            "throughput": 150.0,
            "total_requests": 10000
        }
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a specific trace."""
        # This would query the tracing backend for trace details
        # For now, return mock summary
        return {
            "trace_id": trace_id,
            "duration_ms": 245,
            "span_count": 8,
            "service_count": 3,
            "error_count": 0,
            "root_service": "opensports-api",
            "status": "success"
        } 