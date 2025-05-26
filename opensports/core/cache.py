"""
Caching system for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import json
import pickle
import time
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import redis
from opensports.core.config import settings
from opensports.core.logging import get_logger, LoggerMixin

logger = get_logger(__name__)


class Cache(LoggerMixin):
    """Main caching interface with Redis and in-memory fallback."""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = None):
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl or settings.cache_ttl
        self.redis_client = None
        self.memory_cache = {}
        self.memory_cache_timestamps = {}
        self._setup_redis()
    
    def _setup_redis(self) -> None:
        """Initialize Redis connection with fallback to memory cache."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis cache initialized", url=self.redis_url)
            
        except Exception as e:
            self.logger.warning(
                "Redis connection failed, falling back to memory cache",
                error=str(e)
            )
            self.redis_client = None
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            return json.dumps(value)
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value).hex()
    
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Fall back to pickle
            try:
                return pickle.loads(bytes.fromhex(value))
            except Exception:
                return value
    
    def _memory_cache_cleanup(self) -> None:
        """Clean up expired entries from memory cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.memory_cache_timestamps.items():
            if current_time - timestamp > self.default_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.memory_cache_timestamps.pop(key, None)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        try:
            if self.redis_client:
                # Try Redis first
                value = self.redis_client.get(key)
                if value is not None:
                    return self._deserialize_value(value)
            
            # Fall back to memory cache
            if key in self.memory_cache:
                timestamp = self.memory_cache_timestamps.get(key, 0)
                if time.time() - timestamp <= self.default_ttl:
                    return self.memory_cache[key]
                else:
                    # Expired
                    self.memory_cache.pop(key, None)
                    self.memory_cache_timestamps.pop(key, None)
            
            return default
            
        except Exception as e:
            self.logger.error("Cache get failed", key=key, error=str(e))
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                # Try Redis first
                serialized_value = self._serialize_value(value)
                self.redis_client.setex(key, ttl, serialized_value)
                return True
            
            # Fall back to memory cache
            self.memory_cache[key] = value
            self.memory_cache_timestamps[key] = time.time()
            
            # Periodic cleanup
            if len(self.memory_cache) % 100 == 0:
                self._memory_cache_cleanup()
            
            return True
            
        except Exception as e:
            self.logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            deleted = False
            
            if self.redis_client:
                deleted = bool(self.redis_client.delete(key))
            
            # Also remove from memory cache
            if key in self.memory_cache:
                self.memory_cache.pop(key, None)
                self.memory_cache_timestamps.pop(key, None)
                deleted = True
            
            return deleted
            
        except Exception as e:
            self.logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            
            # Check memory cache
            if key in self.memory_cache:
                timestamp = self.memory_cache_timestamps.get(key, 0)
                if time.time() - timestamp <= self.default_ttl:
                    return True
                else:
                    # Expired
                    self.memory_cache.pop(key, None)
                    self.memory_cache_timestamps.pop(key, None)
            
            return False
            
        except Exception as e:
            self.logger.error("Cache exists check failed", key=key, error=str(e))
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
            
            self.memory_cache.clear()
            self.memory_cache_timestamps.clear()
            
            self.logger.info("Cache cleared")
            return True
            
        except Exception as e:
            self.logger.error("Cache clear failed", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": "redis" if self.redis_client else "memory",
            "memory_cache_size": len(self.memory_cache),
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get("used_memory_human", "unknown"),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_total_commands_processed": info.get("total_commands_processed", 0),
                })
            except Exception as e:
                self.logger.warning("Failed to get Redis stats", error=str(e))
        
        return stats


def cache_result(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_func: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return result
            
            # Execute function and cache result
            logger.debug("Cache miss", function=func.__name__, key=cache_key)
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_async_result(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_func: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to cache async function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return result
            
            # Execute function and cache result
            logger.debug("Cache miss", function=func.__name__, key=cache_key)
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class CacheManager(LoggerMixin):
    """Manager for cache operations and monitoring."""
    
    def __init__(self, cache: Cache):
        self.cache = cache
    
    def warm_up_cache(self, data: Dict[str, Any]) -> None:
        """Pre-populate cache with common data."""
        try:
            for key, value in data.items():
                self.cache.set(key, value)
            
            self.logger.info("Cache warmed up", keys_count=len(data))
            
        except Exception as e:
            self.logger.error("Cache warm-up failed", error=str(e))
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        try:
            if not self.cache.redis_client:
                # For memory cache, we need to check all keys
                keys_to_delete = [
                    key for key in self.cache.memory_cache.keys()
                    if pattern in key
                ]
                for key in keys_to_delete:
                    self.cache.delete(key)
                return len(keys_to_delete)
            
            # For Redis, use SCAN to find matching keys
            keys = []
            cursor = 0
            while True:
                cursor, partial_keys = self.cache.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            
            if keys:
                self.cache.redis_client.delete(*keys)
            
            self.logger.info("Cache invalidated", pattern=pattern, keys_count=len(keys))
            return len(keys)
            
        except Exception as e:
            self.logger.error("Cache invalidation failed", pattern=pattern, error=str(e))
            return 0
    
    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health status."""
        try:
            stats = self.cache.get_stats()
            
            health = {
                "status": "healthy",
                "backend": stats["backend"],
                "stats": stats,
            }
            
            # Test cache operations
            test_key = "health_check_test"
            test_value = {"timestamp": time.time()}
            
            if self.cache.set(test_key, test_value, 60):
                retrieved = self.cache.get(test_key)
                if retrieved == test_value:
                    health["operations"] = "working"
                    self.cache.delete(test_key)
                else:
                    health["operations"] = "read_failed"
                    health["status"] = "degraded"
            else:
                health["operations"] = "write_failed"
                health["status"] = "unhealthy"
            
            return health
            
        except Exception as e:
            self.logger.error("Cache health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global cache instance
_cache_instance: Optional[Cache] = None


def get_cache() -> Cache:
    """Get global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = Cache()
    
    return _cache_instance


def init_cache(redis_url: Optional[str] = None, default_ttl: Optional[int] = None) -> Cache:
    """Initialize cache with custom configuration."""
    global _cache_instance
    _cache_instance = Cache(redis_url, default_ttl)
    return _cache_instance 