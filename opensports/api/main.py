"""
OpenSports FastAPI Application

Main FastAPI application with comprehensive middleware, error handling,
and integration of all OpenSports modules.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import structlog
from opensports.core.config import settings
from opensports.core.logging import get_logger, setup_logging
from opensports.core.database import get_database, initialize_database
from opensports.core.cache import initialize_cache
from opensports.api.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware
)
from opensports.api.endpoints import (
    games,
    players,
    teams,
    analytics,
    predictions,
    realtime,
    agents
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting OpenSports API")
    
    # Initialize core services
    await initialize_database()
    await initialize_cache()
    
    # Initialize stream processor if enabled
    if settings.ENABLE_REAL_TIME:
        from opensports.realtime.stream_processor import StreamProcessor
        stream_processor = StreamProcessor()
        await stream_processor.initialize()
        app.state.stream_processor = stream_processor
        
        # Start background task for stream processing
        asyncio.create_task(stream_processor.start_processing())
    
    logger.info("OpenSports API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenSports API")
    
    if hasattr(app.state, 'stream_processor'):
        await app.state.stream_processor.stop_processing()
    
    logger.info("OpenSports API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="OpenSports API",
    description="""
    🏀 **OpenSports** - World-class sports analytics platform
    
    ## Features
    
    * **AI-Powered Analysis** - Advanced game analysis using OpenAI and LangChain
    * **Real-time Data** - Live game streaming and real-time analytics
    * **Predictive Models** - ML-based game outcome and player performance prediction
    * **Causal Analysis** - Understanding true causal relationships in sports
    * **Audience Segmentation** - Advanced fan behavior analysis
    * **Multi-Sport Support** - NBA, NFL, Soccer, Formula 1, and more
    
    ## Technology Stack
    
    * **Backend**: FastAPI, Python 3.11+
    * **AI/ML**: OpenAI GPT-4, LangChain, scikit-learn, XGBoost
    * **Real-time**: Kafka, Redis, WebSockets
    * **Database**: SQLite, DuckDB, Datasette
    * **Caching**: Redis with intelligent fallback
    * **Monitoring**: Structured logging, metrics, health checks
    
    ## Author
    
    **Nik Jois** - nikjois@llamaearch.ai
    
    Built to impress top sports organizations and tech companies.
    """,
    version="1.0.0",
    contact={
        "name": "Nik Jois",
        "email": "nikjois@llamaearch.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

if settings.ENABLE_RATE_LIMITING:
    app.add_middleware(RateLimitMiddleware)

if settings.ENABLE_AUTHENTICATION:
    app.add_middleware(AuthenticationMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured logging."""
    logger.error(
        "Unhandled exception occurred",
        exception=str(exc),
        exception_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError",
                "status_code": 500,
                "detail": "An internal server error occurred",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "opensports-api"
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with service dependencies."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "opensports-api",
        "dependencies": {}
    }
    
    # Check database
    try:
        db = get_database()
        await db.execute("SELECT 1")
        health_status["dependencies"]["database"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check cache
    try:
        from opensports.core.cache import get_cache
        cache = get_cache()
        await cache.ping()
        health_status["dependencies"]["cache"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check stream processor
    if hasattr(app.state, 'stream_processor'):
        try:
            metrics = await app.state.stream_processor.get_processing_metrics()
            if metrics.get('is_running'):
                health_status["dependencies"]["stream_processor"] = "healthy"
            else:
                health_status["dependencies"]["stream_processor"] = "stopped"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["dependencies"]["stream_processor"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": 0,  # Would calculate actual uptime
        "requests_total": 0,  # Would get from middleware
        "requests_per_second": 0,  # Would calculate
        "active_connections": 0,  # Would get from monitoring
        "memory_usage_mb": 0,  # Would get system metrics
        "cpu_usage_percent": 0,  # Would get system metrics
    }
    
    # Add stream processor metrics if available
    if hasattr(app.state, 'stream_processor'):
        try:
            stream_metrics = await app.state.stream_processor.get_processing_metrics()
            metrics["stream_processor"] = stream_metrics
        except Exception as e:
            logger.warning(f"Could not get stream processor metrics: {e}")
    
    return metrics


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "OpenSports API",
        "version": "1.0.0",
        "description": "World-class sports analytics platform",
        "author": "Nik Jois <nikjois@llamaearch.ai>",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "AI-powered game analysis",
            "Real-time data streaming",
            "Predictive modeling",
            "Causal analysis",
            "Audience segmentation",
            "Multi-sport support"
        ]
    }


# Include API routers
app.include_router(
    games.router,
    prefix="/api/v1/games",
    tags=["Games"]
)

app.include_router(
    players.router,
    prefix="/api/v1/players",
    tags=["Players"]
)

app.include_router(
    teams.router,
    prefix="/api/v1/teams",
    tags=["Teams"]
)

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["Analytics"]
)

app.include_router(
    predictions.router,
    prefix="/api/v1/predictions",
    tags=["Predictions"]
)

app.include_router(
    realtime.router,
    prefix="/api/v1/realtime",
    tags=["Real-time"]
)

app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["AI Agents"]
)


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="OpenSports API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/opensports-logo.png"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "https://api.opensports.ai",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.opensports.ai", 
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Development server
if __name__ == "__main__":
    # Setup logging for development
    setup_logging()
    
    # Run the server
    uvicorn.run(
        "opensports.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True,
        workers=1 if settings.DEBUG else settings.WORKERS
    ) 