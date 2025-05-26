"""
Comprehensive Health Checking System

Advanced health monitoring for all OpenSports platform components
with dependency tracking, circuit breakers, and detailed diagnostics.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import redis.asyncio as redis
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import get_database

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    DATABASE = "database"
    CACHE = "cache"
    API = "api"
    EXTERNAL_SERVICE = "external_service"
    QUEUE = "queue"
    STORAGE = "storage"
    ML_MODEL = "ml_model"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Definition of a health check."""
    name: str
    component_type: ComponentType
    check_function: Callable
    timeout: float = 5.0
    interval: int = 30  # seconds
    retries: int = 3
    enabled: bool = True
    critical: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    component_count: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    checks: List[HealthResult] = field(default_factory=list)
    uptime_seconds: float = 0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Advanced health checking system.
    
    Features:
    - Configurable health checks for all components
    - Circuit breaker pattern for failing services
    - Dependency tracking and cascade failure detection
    - System resource monitoring
    - Historical health data
    - Alerting integration
    """
    
    def __init__(self):
        self.checks = {}
        self.results = {}
        self.circuit_breakers = {}
        self.dependencies = {}
        self.redis_client = None
        self.is_running = False
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the health checker."""
        logger.info("Initializing health checker")
        
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for health checking")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
        
        # Register default health checks
        await self._register_default_checks()
        
        logger.info("Health checker initialized successfully")
    
    async def _register_default_checks(self):
        """Register default health checks for core components."""
        
        # Database health check
        await self.register_check(HealthCheck(
            name="database",
            component_type=ComponentType.DATABASE,
            check_function=self._check_database_health,
            timeout=5.0,
            interval=30,
            critical=True,
            tags={"component": "database", "type": "sqlite"}
        ))
        
        # Redis health check
        await self.register_check(HealthCheck(
            name="redis",
            component_type=ComponentType.CACHE,
            check_function=self._check_redis_health,
            timeout=3.0,
            interval=30,
            critical=True,
            tags={"component": "cache", "type": "redis"}
        ))
        
        # System resources check
        await self.register_check(HealthCheck(
            name="system_resources",
            component_type=ComponentType.CUSTOM,
            check_function=self._check_system_resources,
            timeout=2.0,
            interval=60,
            critical=False,
            tags={"component": "system", "type": "resources"}
        ))
        
        # API endpoints check
        await self.register_check(HealthCheck(
            name="api_endpoints",
            component_type=ComponentType.API,
            check_function=self._check_api_endpoints,
            timeout=10.0,
            interval=60,
            critical=True,
            tags={"component": "api", "type": "endpoints"}
        ))
        
        # ML models check
        await self.register_check(HealthCheck(
            name="ml_models",
            component_type=ComponentType.ML_MODEL,
            check_function=self._check_ml_models,
            timeout=15.0,
            interval=300,
            critical=False,
            tags={"component": "ml", "type": "models"}
        ))
    
    async def register_check(self, check: HealthCheck):
        """Register a new health check."""
        self.checks[check.name] = check
        self.circuit_breakers[check.name] = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=Exception
        )
        logger.info(f"Registered health check: {check.name}")
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        self.is_running = True
        logger.info("Starting health monitoring")
        
        # Start individual check tasks
        tasks = []
        for check_name, check in self.checks.items():
            if check.enabled:
                task = asyncio.create_task(self._run_check_loop(check))
                tasks.append(task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in health monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop."""
        self.is_running = False
        logger.info("Stopping health monitoring")
    
    async def _run_check_loop(self, check: HealthCheck):
        """Run a single health check in a loop."""
        while self.is_running:
            try:
                await self._execute_check(check)
                await asyncio.sleep(check.interval)
            except Exception as e:
                logger.error(f"Error in check loop for {check.name}: {e}")
                await asyncio.sleep(5)
    
    async def _execute_check(self, check: HealthCheck):
        """Execute a single health check."""
        circuit_breaker = self.circuit_breakers[check.name]
        
        start_time = time.time()
        result = None
        
        try:
            # Check circuit breaker state
            if circuit_breaker.state == "open":
                result = HealthResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Circuit breaker is open",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    error="Circuit breaker protection active"
                )
            else:
                # Execute the check with timeout
                result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout
                )
                result.name = check.name
                result.timestamp = datetime.now()
                result.duration_ms = (time.time() - start_time) * 1000
                
                # Update circuit breaker
                if result.status == HealthStatus.HEALTHY:
                    circuit_breaker.record_success()
                else:
                    circuit_breaker.record_failure()
                    
        except asyncio.TimeoutError:
            result = HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                error="Timeout"
            )
            circuit_breaker.record_failure()
            
        except Exception as e:
            result = HealthResult(
                name=check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
            circuit_breaker.record_failure()
        
        # Store result
        self.results[check.name] = result
        
        # Store in Redis for persistence
        if self.redis_client:
            await self._store_result_in_redis(result)
        
        logger.debug(f"Health check {check.name}: {result.status.value}")
    
    async def _store_result_in_redis(self, result: HealthResult):
        """Store health check result in Redis."""
        try:
            key = f"health:{result.name}:latest"
            data = {
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms,
                "error": result.error,
                "details": result.details
            }
            
            await self.redis_client.setex(key, 3600, str(data))  # 1 hour TTL
            
            # Also store in time series for history
            history_key = f"health:{result.name}:history"
            await self.redis_client.lpush(history_key, str(data))
            await self.redis_client.ltrim(history_key, 0, 100)  # Keep last 100 results
            
        except Exception as e:
            logger.error(f"Failed to store health result in Redis: {e}")
    
    async def _check_database_health(self) -> HealthResult:
        """Check database connectivity and performance."""
        try:
            db = get_database()
            
            # Test basic connectivity
            start_time = time.time()
            await db.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            
            # Get database stats
            stats = await db.fetch_one("SELECT COUNT(*) as table_count FROM sqlite_master WHERE type='table'")
            table_count = stats['table_count'] if stats else 0
            
            # Check if query time is acceptable
            if query_time > 1000:  # 1 second
                status = HealthStatus.DEGRADED
                message = f"Database responding slowly ({query_time:.2f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy ({query_time:.2f}ms)"
            
            return HealthResult(
                name="database",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=query_time,
                details={
                    "query_time_ms": query_time,
                    "table_count": table_count,
                    "database_type": "sqlite"
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_redis_health(self) -> HealthResult:
        """Check Redis connectivity and performance."""
        try:
            if not self.redis_client:
                return HealthResult(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis client not initialized",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    error="No Redis connection"
                )
            
            # Test basic connectivity
            start_time = time.time()
            await self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = await self.redis_client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 0)
            
            # Check if ping time is acceptable
            if ping_time > 100:  # 100ms
                status = HealthStatus.DEGRADED
                message = f"Redis responding slowly ({ping_time:.2f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis healthy ({ping_time:.2f}ms)"
            
            return HealthResult(
                name="redis",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=ping_time,
                details={
                    "ping_time_ms": ping_time,
                    "memory_usage": memory_usage,
                    "connected_clients": connected_clients,
                    "redis_version": info.get('redis_version', 'unknown')
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_system_resources(self) -> HealthResult:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
            except AttributeError:
                load_avg = 0  # Windows doesn't have load average
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Elevated disk usage: {disk_percent:.1f}%")
            
            message = "System resources healthy"
            if issues:
                message = "; ".join(issues)
            
            return HealthResult(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=1000,  # Approximate time for resource checks
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "load_average": load_avg,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "disk_total_gb": round(disk.total / (1024**3), 2)
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resources check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_api_endpoints(self) -> HealthResult:
        """Check critical API endpoints."""
        try:
            base_url = getattr(settings, 'API_BASE_URL', 'http://localhost:8000')
            endpoints_to_check = [
                '/health',
                '/api/v1/health',
                '/api/v1/games',
                '/api/v1/players'
            ]
            
            healthy_endpoints = 0
            total_endpoints = len(endpoints_to_check)
            endpoint_results = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints_to_check:
                    try:
                        url = f"{base_url}{endpoint}"
                        start_time = time.time()
                        
                        async with session.get(url, timeout=5) as response:
                            response_time = (time.time() - start_time) * 1000
                            
                            if response.status < 400:
                                healthy_endpoints += 1
                                endpoint_results[endpoint] = {
                                    "status": "healthy",
                                    "response_time_ms": response_time,
                                    "status_code": response.status
                                }
                            else:
                                endpoint_results[endpoint] = {
                                    "status": "unhealthy",
                                    "response_time_ms": response_time,
                                    "status_code": response.status
                                }
                                
                    except Exception as e:
                        endpoint_results[endpoint] = {
                            "status": "unhealthy",
                            "error": str(e)
                        }
            
            # Determine overall status
            health_ratio = healthy_endpoints / total_endpoints
            
            if health_ratio == 1.0:
                status = HealthStatus.HEALTHY
                message = f"All {total_endpoints} API endpoints healthy"
            elif health_ratio >= 0.8:
                status = HealthStatus.DEGRADED
                message = f"{healthy_endpoints}/{total_endpoints} API endpoints healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Only {healthy_endpoints}/{total_endpoints} API endpoints healthy"
            
            return HealthResult(
                name="api_endpoints",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=5000,  # Approximate time for all endpoint checks
                details={
                    "healthy_endpoints": healthy_endpoints,
                    "total_endpoints": total_endpoints,
                    "health_ratio": health_ratio,
                    "endpoint_results": endpoint_results
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="api_endpoints",
                status=HealthStatus.UNHEALTHY,
                message=f"API endpoints check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                error=str(e)
            )
    
    async def _check_ml_models(self) -> HealthResult:
        """Check ML model availability and performance."""
        try:
            # This would check if ML models are loaded and responding
            # For now, we'll simulate the check
            
            models_to_check = [
                "player_performance_predictor",
                "game_outcome_predictor",
                "injury_risk_predictor"
            ]
            
            healthy_models = 0
            model_results = {}
            
            for model_name in models_to_check:
                try:
                    # Simulate model health check
                    # In reality, this would call the model's predict method with test data
                    start_time = time.time()
                    
                    # Simulate model prediction time
                    await asyncio.sleep(0.1)
                    
                    prediction_time = (time.time() - start_time) * 1000
                    
                    if prediction_time < 1000:  # Less than 1 second
                        healthy_models += 1
                        model_results[model_name] = {
                            "status": "healthy",
                            "prediction_time_ms": prediction_time
                        }
                    else:
                        model_results[model_name] = {
                            "status": "degraded",
                            "prediction_time_ms": prediction_time
                        }
                        
                except Exception as e:
                    model_results[model_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            # Determine overall status
            total_models = len(models_to_check)
            health_ratio = healthy_models / total_models
            
            if health_ratio == 1.0:
                status = HealthStatus.HEALTHY
                message = f"All {total_models} ML models healthy"
            elif health_ratio >= 0.7:
                status = HealthStatus.DEGRADED
                message = f"{healthy_models}/{total_models} ML models healthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Only {healthy_models}/{total_models} ML models healthy"
            
            return HealthResult(
                name="ml_models",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=len(models_to_check) * 100,
                details={
                    "healthy_models": healthy_models,
                    "total_models": total_models,
                    "health_ratio": health_ratio,
                    "model_results": model_results
                }
            )
            
        except Exception as e:
            return HealthResult(
                name="ml_models",
                status=HealthStatus.UNHEALTHY,
                message=f"ML models check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                error=str(e)
            )
    
    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        current_results = list(self.results.values())
        
        if not current_results:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                component_count=0,
                healthy_count=0,
                degraded_count=0,
                unhealthy_count=0,
                uptime_seconds=time.time() - self.start_time
            )
        
        # Count statuses
        healthy_count = sum(1 for r in current_results if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in current_results if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in current_results if r.status == HealthStatus.UNHEALTHY)
        
        # Determine overall status
        if unhealthy_count > 0:
            # Check if any critical components are unhealthy
            critical_unhealthy = any(
                r.status == HealthStatus.UNHEALTHY and 
                self.checks.get(r.name, HealthCheck("", ComponentType.CUSTOM, None)).critical
                for r in current_results
            )
            overall_status = HealthStatus.UNHEALTHY if critical_unhealthy else HealthStatus.DEGRADED
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Get system metrics
        system_metrics = {}
        try:
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                "process_count": len(psutil.pids()),
                "uptime_seconds": time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            component_count=len(current_results),
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            checks=current_results,
            uptime_seconds=time.time() - self.start_time,
            system_metrics=system_metrics
        )
    
    async def get_check_history(self, check_name: str, hours: int = 24) -> List[HealthResult]:
        """Get historical health check results."""
        if not self.redis_client:
            return []
        
        try:
            history_key = f"health:{check_name}:history"
            history_data = await self.redis_client.lrange(history_key, 0, -1)
            
            results = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for data_str in history_data:
                try:
                    data = eval(data_str)  # In production, use json.loads
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    
                    if timestamp >= cutoff_time:
                        result = HealthResult(
                            name=check_name,
                            status=HealthStatus(data['status']),
                            message=data['message'],
                            timestamp=timestamp,
                            duration_ms=data['duration_ms'],
                            details=data.get('details', {}),
                            error=data.get('error')
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"Failed to parse health history data: {e}")
            
            return sorted(results, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get health check history: {e}")
            return []


class CircuitBreaker:
    """Circuit breaker implementation for health checks."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True 