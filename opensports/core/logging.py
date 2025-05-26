"""
Structured logging configuration for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import LoggerFactory
from opensports.core.config import settings


def setup_logging(
    level: Optional[str] = None,
    json_logs: bool = False,
    include_timestamp: bool = True,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
        include_timestamp: Whether to include timestamps in logs
    """
    log_level = level or settings.log_level
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if include_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="ISO"))
    
    # Add context processors
    processors.extend([
        add_app_context,
        add_request_context,
    ])
    
    if json_logs or settings.is_production:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def add_app_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add application context to log events."""
    event_dict["app"] = settings.app_name
    event_dict["version"] = settings.app_version
    event_dict["environment"] = settings.environment
    return event_dict


def add_request_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request context to log events if available."""
    # This will be populated by middleware in the API
    import contextvars
    
    try:
        request_id = contextvars.copy_context().get("request_id")
        if request_id:
            event_dict["request_id"] = request_id
    except (LookupError, AttributeError):
        pass
    
    try:
        user_id = contextvars.copy_context().get("user_id")
        if user_id:
            event_dict["user_id"] = user_id
    except (LookupError, AttributeError):
        pass
    
    return event_dict


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Configured structlog logger
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "opensports")
        else:
            name = "opensports"
    
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get a logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation: str, logger: Optional[structlog.BoundLogger] = None):
        self.operation = operation
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info("Operation started", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation,
                duration_seconds=round(duration, 3)
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation,
                duration_seconds=round(duration, 3),
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )


def log_function_call(func):
    """Decorator to log function calls with parameters and results."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Log function entry
        logger.debug(
            "Function called",
            function=func_name,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys())
        )
        
        try:
            with PerformanceLogger(f"function_{func.__name__}", logger):
                result = func(*args, **kwargs)
            
            logger.debug("Function completed", function=func_name)
            return result
            
        except Exception as e:
            logger.error(
                "Function failed",
                function=func_name,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True
            )
            raise
    
    return wrapper


# Initialize logging on import
setup_logging() 