"""
Advanced Metrics Collection System

Comprehensive metrics collection for business KPIs, system performance,
and operational monitoring with Prometheus integration.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import redis.asyncio as redis
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import get_database

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: str = ""
    namespace: str = "opensports"


class MetricsCollector:
    """
    Advanced metrics collection system with Prometheus integration.
    
    Features:
    - Multiple metric types (Counter, Gauge, Histogram, Summary)
    - Custom business metrics
    - System performance metrics
    - Real-time aggregation
    - Redis caching for fast access
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.redis_client = None
        self.db = get_database()
        self.collection_interval = 30  # seconds
        self.is_collecting = False
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    async def initialize(self):
        """Initialize metrics collector."""
        logger.info("Initializing metrics collector")
        
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for metrics")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    def _initialize_core_metrics(self):
        """Initialize core system and business metrics."""
        core_metrics = [
            # System metrics
            MetricDefinition(
                name="http_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total HTTP requests",
                labels=["method", "endpoint", "status_code"]
            ),
            MetricDefinition(
                name="http_request_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="HTTP request duration",
                labels=["method", "endpoint"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricDefinition(
                name="database_connections_active",
                metric_type=MetricType.GAUGE,
                description="Active database connections"
            ),
            MetricDefinition(
                name="cache_hit_rate",
                metric_type=MetricType.GAUGE,
                description="Cache hit rate percentage"
            ),
            
            # Business metrics
            MetricDefinition(
                name="games_analyzed_total",
                metric_type=MetricType.COUNTER,
                description="Total games analyzed",
                labels=["sport", "analysis_type"]
            ),
            MetricDefinition(
                name="predictions_made_total",
                metric_type=MetricType.COUNTER,
                description="Total predictions made",
                labels=["sport", "prediction_type"]
            ),
            MetricDefinition(
                name="prediction_accuracy",
                metric_type=MetricType.GAUGE,
                description="Prediction accuracy percentage",
                labels=["sport", "model"]
            ),
            MetricDefinition(
                name="user_sessions_active",
                metric_type=MetricType.GAUGE,
                description="Active user sessions"
            ),
            MetricDefinition(
                name="api_calls_per_minute",
                metric_type=MetricType.GAUGE,
                description="API calls per minute",
                labels=["endpoint"]
            ),
            
            # Data quality metrics
            MetricDefinition(
                name="data_ingestion_rate",
                metric_type=MetricType.GAUGE,
                description="Data ingestion rate (records/second)",
                labels=["source", "sport"]
            ),
            MetricDefinition(
                name="data_quality_score",
                metric_type=MetricType.GAUGE,
                description="Data quality score (0-100)",
                labels=["dataset", "sport"]
            ),
            MetricDefinition(
                name="model_training_duration_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Model training duration",
                labels=["model_type", "sport"],
                buckets=[60, 300, 900, 1800, 3600, 7200]
            ),
        ]
        
        for metric_def in core_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric."""
        metric_name = f"{metric_def.namespace}_{metric_def.name}"
        
        if metric_def.metric_type == MetricType.COUNTER:
            metric = Counter(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.GAUGE:
            metric = Gauge(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.HISTOGRAM:
            metric = Histogram(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                buckets=metric_def.buckets,
                registry=self.registry
            )
        elif metric_def.metric_type == MetricType.SUMMARY:
            metric = Summary(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric_def.metric_type}")
        
        self.metrics[metric_def.name] = {
            'metric': metric,
            'definition': metric_def
        }
        
        logger.info(f"Registered metric: {metric_name}")
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None, value: float = 1):
        """Increment a counter metric."""
        if metric_name not in self.metrics:
            logger.warning(f"Metric not found: {metric_name}")
            return
        
        metric_info = self.metrics[metric_name]
        if metric_info['definition'].metric_type != MetricType.COUNTER:
            logger.warning(f"Metric {metric_name} is not a counter")
            return
        
        if labels:
            metric_info['metric'].labels(**labels).inc(value)
        else:
            metric_info['metric'].inc(value)
    
    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        if metric_name not in self.metrics:
            logger.warning(f"Metric not found: {metric_name}")
            return
        
        metric_info = self.metrics[metric_name]
        if metric_info['definition'].metric_type != MetricType.GAUGE:
            logger.warning(f"Metric {metric_name} is not a gauge")
            return
        
        if labels:
            metric_info['metric'].labels(**labels).set(value)
        else:
            metric_info['metric'].set(value)
    
    def observe_histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value in a histogram metric."""
        if metric_name not in self.metrics:
            logger.warning(f"Metric not found: {metric_name}")
            return
        
        metric_info = self.metrics[metric_name]
        if metric_info['definition'].metric_type != MetricType.HISTOGRAM:
            logger.warning(f"Metric {metric_name} is not a histogram")
            return
        
        if labels:
            metric_info['metric'].labels(**labels).observe(value)
        else:
            metric_info['metric'].observe(value)
    
    def time_function(self, metric_name: str, labels: Dict[str, str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels)
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def start_collection(self):
        """Start automatic metrics collection."""
        self.is_collecting = True
        logger.info("Starting metrics collection")
        
        while self.is_collecting:
            try:
                await self._collect_system_metrics()
                await self._collect_business_metrics()
                await self._store_metrics_snapshot()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    async def stop_collection(self):
        """Stop automatic metrics collection."""
        self.is_collecting = False
        logger.info("Stopping metrics collection")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # Database connections
            # This would query actual database connection pool
            active_connections = 10  # Mock value
            self.set_gauge("database_connections_active", active_connections)
            
            # Cache hit rate
            if self.redis_client:
                info = await self.redis_client.info()
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0
                self.set_gauge("cache_hit_rate", hit_rate)
            
            # Active sessions
            if self.redis_client:
                session_keys = await self.redis_client.keys("session:*")
                active_sessions = len(session_keys)
                self.set_gauge("user_sessions_active", active_sessions)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_business_metrics(self):
        """Collect business-specific metrics."""
        try:
            # Data ingestion rates
            sports = ['nba', 'nfl', 'soccer', 'hockey']
            for sport in sports:
                # Mock data ingestion rate
                ingestion_rate = np.random.normal(100, 20)
                self.set_gauge("data_ingestion_rate", ingestion_rate, {"source": "api", "sport": sport})
                
                # Mock data quality score
                quality_score = np.random.normal(85, 10)
                self.set_gauge("data_quality_score", quality_score, {"dataset": "games", "sport": sport})
                
                # Mock prediction accuracy
                accuracy = np.random.normal(75, 5)
                self.set_gauge("prediction_accuracy", accuracy, {"sport": sport, "model": "ensemble"})
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
    
    async def _store_metrics_snapshot(self):
        """Store current metrics snapshot in Redis."""
        try:
            if not self.redis_client:
                return
            
            timestamp = datetime.now().isoformat()
            metrics_data = {}
            
            # Collect current metric values
            for metric_name, metric_info in self.metrics.items():
                metric = metric_info['metric']
                metric_type = metric_info['definition'].metric_type
                
                if metric_type == MetricType.GAUGE:
                    # For gauges, get current value
                    metrics_data[metric_name] = metric._value._value
                elif metric_type == MetricType.COUNTER:
                    # For counters, get current count
                    metrics_data[metric_name] = metric._value._value
            
            # Store in Redis with timestamp
            await self.redis_client.setex(
                f"metrics:snapshot:{timestamp}",
                3600,  # 1 hour TTL
                str(metrics_data)
            )
            
            # Keep latest snapshot
            await self.redis_client.setex(
                "metrics:latest",
                3600,
                str(metrics_data)
            )
            
        except Exception as e:
            logger.error(f"Error storing metrics snapshot: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            'total_metrics': len(self.metrics),
            'metric_types': {},
            'collection_status': self.is_collecting,
            'collection_interval': self.collection_interval,
            'timestamp': datetime.now().isoformat()
        }
        
        # Count metrics by type
        for metric_info in self.metrics.values():
            metric_type = metric_info['definition'].metric_type.value
            summary['metric_types'][metric_type] = summary['metric_types'].get(metric_type, 0) + 1
        
        return summary
    
    async def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric."""
        if not self.redis_client:
            return []
        
        history = []
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # This would query stored metric snapshots
        # For now, return mock historical data
        timestamps = pd.date_range(start_time, end_time, freq='5min')
        
        for timestamp in timestamps:
            # Mock historical values
            value = np.random.normal(50, 10)
            history.append({
                'timestamp': timestamp.isoformat(),
                'value': float(value)
            })
        
        return history


class BusinessMetrics:
    """
    Business-specific metrics for sports analytics.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_game_analysis(self, sport: str, analysis_type: str):
        """Record a game analysis event."""
        self.collector.increment_counter(
            "games_analyzed_total",
            {"sport": sport, "analysis_type": analysis_type}
        )
    
    def record_prediction(self, sport: str, prediction_type: str):
        """Record a prediction event."""
        self.collector.increment_counter(
            "predictions_made_total",
            {"sport": sport, "prediction_type": prediction_type}
        )
    
    def update_prediction_accuracy(self, sport: str, model: str, accuracy: float):
        """Update prediction accuracy metric."""
        self.collector.set_gauge(
            "prediction_accuracy",
            accuracy,
            {"sport": sport, "model": model}
        )
    
    def record_model_training(self, model_type: str, sport: str, duration: float):
        """Record model training duration."""
        self.collector.observe_histogram(
            "model_training_duration_seconds",
            duration,
            {"model_type": model_type, "sport": sport}
        )
    
    def update_data_quality(self, dataset: str, sport: str, score: float):
        """Update data quality score."""
        self.collector.set_gauge(
            "data_quality_score",
            score,
            {"dataset": dataset, "sport": sport}
        )


class SystemMetrics:
    """
    System performance metrics.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        # Increment request counter
        self.collector.increment_counter(
            "http_requests_total",
            {"method": method, "endpoint": endpoint, "status_code": str(status_code)}
        )
        
        # Record request duration
        self.collector.observe_histogram(
            "http_request_duration_seconds",
            duration,
            {"method": method, "endpoint": endpoint}
        )
    
    def update_api_rate(self, endpoint: str, rate: float):
        """Update API call rate."""
        self.collector.set_gauge(
            "api_calls_per_minute",
            rate,
            {"endpoint": endpoint}
        )
    
    def update_cache_hit_rate(self, rate: float):
        """Update cache hit rate."""
        self.collector.set_gauge("cache_hit_rate", rate)
    
    def update_database_connections(self, count: int):
        """Update active database connections."""
        self.collector.set_gauge("database_connections_active", count) 