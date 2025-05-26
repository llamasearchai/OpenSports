"""
Persistence module for OpenInsight Experiment Service.

This module provides storage backends for saving and loading experiment data
from various sources such as:
- Local file system (JSON)
- SQL databases (via SQLAlchemy)
- Redis (for high-performance applications)
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Type
import structlog
from abc import ABC, abstractmethod

from OpenInsight.experiments.experiment_service import (
    ExperimentManager,
    Experiment,
    ExperimentVariant,
    ExperimentType
)

logger = structlog.get_logger(__name__)

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_experiments(self, experiments: Dict[str, Experiment]) -> bool:
        """
        Save experiments to storage.
        
        Args:
            experiments: Dictionary of experiment ID to Experiment
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_experiments(self) -> Dict[str, Experiment]:
        """
        Load experiments from storage.
        
        Returns:
            Dictionary of experiment ID to Experiment
        """
        pass

class JSONFileBackend(StorageBackend):
    """Storage backend using JSON files on the local file system."""
    
    def __init__(self, file_path: str):
        """
        Initialize the JSON file backend.
        
        Args:
            file_path: Path to the JSON file
        """
        self.file_path = file_path
    
    def save_experiments(self, experiments: Dict[str, Experiment]) -> bool:
        """Save experiments to a JSON file."""
        try:
            # Convert experiments to dictionaries
            experiments_data = {
                exp_id: exp.to_dict() for exp_id, exp in experiments.items()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            
            # Write to file
            with open(self.file_path, 'w') as f:
                json.dump(experiments_data, f, indent=2)
            
            logger.info(f"Saved {len(experiments)} experiments to {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save experiments to {self.file_path}", error=str(e))
            return False
    
    def load_experiments(self) -> Dict[str, Experiment]:
        """Load experiments from a JSON file."""
        experiments = {}
        
        if not os.path.exists(self.file_path):
            logger.warning(f"Experiments file not found: {self.file_path}")
            return experiments
        
        try:
            with open(self.file_path, 'r') as f:
                experiments_data = json.load(f)
            
            # Convert dictionaries to Experiment objects
            for exp_id, exp_data in experiments_data.items():
                try:
                    experiment = Experiment.from_dict(exp_data)
                    experiments[exp_id] = experiment
                except Exception as e:
                    logger.error(f"Failed to load experiment {exp_id}", error=str(e))
            
            logger.info(f"Loaded {len(experiments)} experiments from {self.file_path}")
            return experiments
        except Exception as e:
            logger.error(f"Failed to load experiments from {self.file_path}", error=str(e))
            return experiments

class SQLDatabaseBackend(StorageBackend):
    """Storage backend using SQL database via SQLAlchemy."""
    
    def __init__(self, connection_string: str, create_tables: bool = True):
        """
        Initialize the SQL database backend.
        
        Args:
            connection_string: SQLAlchemy connection string (e.g., 'sqlite:///experiments.db')
            create_tables: Whether to create tables if they don't exist
        """
        try:
            import sqlalchemy as sa
            from sqlalchemy.ext.declarative import declarative_base
            from sqlalchemy.orm import sessionmaker
            
            self.sa = sa
            self.Base = declarative_base()
            self.engine = sa.create_engine(connection_string)
            self.Session = sessionmaker(bind=self.engine)
            
            # Define database models
            class ExperimentModel(self.Base):
                __tablename__ = 'experiments'
                
                experiment_id = sa.Column(sa.String, primary_key=True)
                name = sa.Column(sa.String, nullable=False)
                experiment_type = sa.Column(sa.String, nullable=False)
                traffic_allocation = sa.Column(sa.Float, nullable=False)
                description = sa.Column(sa.String, nullable=True)
                is_active = sa.Column(sa.Boolean, nullable=False)
                created_at = sa.Column(sa.String, nullable=False)
                updated_at = sa.Column(sa.String, nullable=False)
                ended_at = sa.Column(sa.String, nullable=True)
                data = sa.Column(sa.JSON, nullable=False)  # Store full serialized experiment
            
            class VariantModel(self.Base):
                __tablename__ = 'variants'
                
                variant_id = sa.Column(sa.String, primary_key=True)
                experiment_id = sa.Column(sa.String, sa.ForeignKey('experiments.experiment_id'), nullable=False)
                name = sa.Column(sa.String, nullable=False)
                description = sa.Column(sa.String, nullable=True)
                impressions = sa.Column(sa.Integer, nullable=False, default=0)
                conversions = sa.Column(sa.Integer, nullable=False, default=0)
                conversion_value = sa.Column(sa.Float, nullable=False, default=0.0)
            
            self.ExperimentModel = ExperimentModel
            self.VariantModel = VariantModel
            
            # Create tables if requested
            if create_tables:
                self.Base.metadata.create_all(self.engine)
                logger.info(f"Created database tables for experiments")
                
        except ImportError:
            logger.error("SQLAlchemy is required for SQLDatabaseBackend but not installed")
            raise ImportError("SQLAlchemy is required for SQLDatabaseBackend. Install with: pip install sqlalchemy")
        
        except Exception as e:
            logger.error(f"Failed to initialize SQL database backend", error=str(e))
            raise
    
    def save_experiments(self, experiments: Dict[str, Experiment]) -> bool:
        """Save experiments to the SQL database."""
        try:
            session = self.Session()
            
            try:
                # Save each experiment
                for exp_id, experiment in experiments.items():
                    # Convert experiment to dict
                    exp_dict = experiment.to_dict()
                    
                    # Check if experiment already exists
                    db_experiment = session.query(self.ExperimentModel).filter_by(experiment_id=exp_id).first()
                    
                    if db_experiment:
                        # Update existing experiment
                        db_experiment.name = exp_dict['name']
                        db_experiment.experiment_type = exp_dict['experiment_type']
                        db_experiment.traffic_allocation = exp_dict['traffic_allocation']
                        db_experiment.description = exp_dict['description']
                        db_experiment.is_active = exp_dict['is_active']
                        db_experiment.updated_at = exp_dict['updated_at']
                        db_experiment.ended_at = exp_dict['ended_at']
                        db_experiment.data = exp_dict
                    else:
                        # Create new experiment
                        db_experiment = self.ExperimentModel(
                            experiment_id=exp_id,
                            name=exp_dict['name'],
                            experiment_type=exp_dict['experiment_type'],
                            traffic_allocation=exp_dict['traffic_allocation'],
                            description=exp_dict['description'],
                            is_active=exp_dict['is_active'],
                            created_at=exp_dict['created_at'],
                            updated_at=exp_dict['updated_at'],
                            ended_at=exp_dict['ended_at'],
                            data=exp_dict
                        )
                        session.add(db_experiment)
                    
                    # Update or create variants
                    for variant_dict in exp_dict['variants']:
                        variant_id = variant_dict['variant_id']
                        
                        # Check if variant already exists
                        db_variant = session.query(self.VariantModel).filter_by(variant_id=variant_id).first()
                        
                        if db_variant:
                            # Update existing variant
                            db_variant.name = variant_dict['name']
                            db_variant.description = variant_dict['description']
                            db_variant.impressions = variant_dict['impressions']
                            db_variant.conversions = variant_dict['conversions']
                            db_variant.conversion_value = variant_dict['conversion_value']
                        else:
                            # Create new variant
                            db_variant = self.VariantModel(
                                variant_id=variant_id,
                                experiment_id=exp_id,
                                name=variant_dict['name'],
                                description=variant_dict['description'],
                                impressions=variant_dict['impressions'],
                                conversions=variant_dict['conversions'],
                                conversion_value=variant_dict['conversion_value']
                            )
                            session.add(db_variant)
                
                # Commit all changes
                session.commit()
                logger.info(f"Saved {len(experiments)} experiments to SQL database")
                return True
                
            except Exception as e:
                # Rollback on error
                session.rollback()
                logger.error(f"Failed to save experiments to SQL database", error=str(e))
                return False
                
            finally:
                # Close session
                session.close()
                
        except Exception as e:
            logger.error(f"Unexpected error saving experiments to SQL database", error=str(e))
            return False
    
    def load_experiments(self) -> Dict[str, Experiment]:
        """Load experiments from the SQL database."""
        experiments = {}
        
        try:
            session = self.Session()
            
            try:
                # Query for all experiments
                db_experiments = session.query(self.ExperimentModel).all()
                
                for db_experiment in db_experiments:
                    try:
                        # Create Experiment from stored JSON data
                        experiment = Experiment.from_dict(db_experiment.data)
                        experiments[db_experiment.experiment_id] = experiment
                    except Exception as e:
                        logger.error(f"Failed to load experiment {db_experiment.experiment_id}", error=str(e))
                
                logger.info(f"Loaded {len(experiments)} experiments from SQL database")
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to load experiments from SQL database", error=str(e))
        
        return experiments

class RedisBackend(StorageBackend):
    """Storage backend using Redis for high-performance access."""
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "experiments:",
        expire_time: Optional[int] = None
    ):
        """
        Initialize the Redis backend.
        
        Args:
            redis_url: Redis connection URL (format: redis://host:port/db)
            key_prefix: Prefix for Redis keys to avoid collisions
            expire_time: Optional expiration time in seconds for keys
        """
        try:
            import redis
            self.redis = redis.from_url(redis_url)
            self.key_prefix = key_prefix
            self.expire_time = expire_time
            
            # Test connection
            self.redis.ping()
            logger.info(f"Connected to Redis at {redis_url}")
            
        except ImportError:
            logger.error("Redis is required for RedisBackend but not installed")
            raise ImportError("Redis is required for RedisBackend. Install with: pip install redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {redis_url}", error=str(e))
            raise
    
    def _get_experiment_key(self, experiment_id: str) -> str:
        """Get the Redis key for an experiment."""
        return f"{self.key_prefix}{experiment_id}"
    
    def _get_all_experiment_keys(self) -> List[str]:
        """Get all experiment keys from Redis."""
        pattern = f"{self.key_prefix}*"
        return [key.decode('utf-8') for key in self.redis.keys(pattern)]
    
    def save_experiments(self, experiments: Dict[str, Experiment]) -> bool:
        """Save experiments to Redis."""
        try:
            # Use a Redis pipeline for better performance with multiple operations
            pipe = self.redis.pipeline()
            
            # Store each experiment as a JSON string
            for exp_id, experiment in experiments.items():
                key = self._get_experiment_key(exp_id)
                exp_data = json.dumps(experiment.to_dict())
                pipe.set(key, exp_data)
                
                # Set expiration time if specified
                if self.expire_time:
                    pipe.expire(key, self.expire_time)
            
            # Execute all commands in the pipeline
            pipe.execute()
            
            # Store a list of all experiment IDs for easier retrieval
            all_experiment_ids = list(experiments.keys())
            self.redis.set(f"{self.key_prefix}all_ids", json.dumps(all_experiment_ids))
            
            logger.info(f"Saved {len(experiments)} experiments to Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiments to Redis", error=str(e))
            return False
    
    def load_experiments(self) -> Dict[str, Experiment]:
        """Load experiments from Redis."""
        experiments = {}
        
        try:
            # Try to get the list of all experiment IDs
            all_ids_key = f"{self.key_prefix}all_ids"
            all_ids_data = self.redis.get(all_ids_key)
            
            if all_ids_data:
                # We have the list of IDs, retrieve each experiment
                experiment_ids = json.loads(all_ids_data.decode('utf-8'))
                
                pipe = self.redis.pipeline()
                for exp_id in experiment_ids:
                    key = self._get_experiment_key(exp_id)
                    pipe.get(key)
                
                # Execute all gets in one batch
                results = pipe.execute()
                
                # Process results
                for i, exp_id in enumerate(experiment_ids):
                    if results[i]:
                        try:
                            exp_data = json.loads(results[i].decode('utf-8'))
                            experiment = Experiment.from_dict(exp_data)
                            experiments[exp_id] = experiment
                        except Exception as e:
                            logger.error(f"Failed to load experiment {exp_id} from Redis", error=str(e))
            else:
                # Fallback: scan keys to find experiments
                logger.warning("Experiment ID list not found in Redis, scanning keys")
                experiment_keys = self._get_all_experiment_keys()
                
                # Skip the all_ids key
                experiment_keys = [key for key in experiment_keys if key != all_ids_key]
                
                if experiment_keys:
                    pipe = self.redis.pipeline()
                    for key in experiment_keys:
                        pipe.get(key)
                    
                    results = pipe.execute()
                    
                    for i, key in enumerate(experiment_keys):
                        if results[i]:
                            try:
                                exp_id = key[len(self.key_prefix):]  # Remove prefix to get the ID
                                exp_data = json.loads(results[i].decode('utf-8'))
                                experiment = Experiment.from_dict(exp_data)
                                experiments[exp_id] = experiment
                            except Exception as e:
                                logger.error(f"Failed to load experiment from key {key}", error=str(e))
            
            logger.info(f"Loaded {len(experiments)} experiments from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load experiments from Redis", error=str(e))
        
        return experiments

class CachingManager:
    """
    Wrapper for ExperimentManager that adds an in-memory cache to improve performance.
    
    This is particularly useful for high-traffic applications where the same experiment
    and variant lookups happen frequently.
    """
    
    def __init__(
        self, 
        manager: Any,  # Can be ExperimentManager or AutoSavingManager
        variant_cache_size: int = 1000,
        variant_cache_ttl: int = 300,  # 5 minutes
        analysis_cache_size: int = 50,
        analysis_cache_ttl: int = 60   # 1 minute
    ):
        """
        Initialize the caching manager.
        
        Args:
            manager: The experiment manager to wrap
            variant_cache_size: Maximum number of variant assignments to cache
            variant_cache_ttl: Time-to-live for variant assignments in seconds
            analysis_cache_size: Maximum number of analysis results to cache
            analysis_cache_ttl: Time-to-live for analysis results in seconds
        """
        self.manager = manager
        self.variant_cache_size = variant_cache_size
        self.variant_cache_ttl = variant_cache_ttl
        self.analysis_cache_size = analysis_cache_size
        self.analysis_cache_ttl = analysis_cache_ttl
        
        # Initialize caches
        self.variant_cache = {}  # {(experiment_id, user_id): (variant, timestamp)}
        self.analysis_cache = {}  # {experiment_id: (analysis_result, timestamp)}
        
        logger.info(f"Created caching manager with variant cache size {variant_cache_size} and analysis cache size {analysis_cache_size}")
    
    def _clean_variant_cache(self) -> None:
        """Remove expired entries from variant cache and trim if too large."""
        now = time.time()
        
        # Remove expired entries
        self.variant_cache = {
            k: v for k, v in self.variant_cache.items() 
            if now - v[1] < self.variant_cache_ttl
        }
        
        # Trim cache if it's too large
        if len(self.variant_cache) > self.variant_cache_size:
            # Sort by timestamp (oldest first) and keep only the newest entries
            sorted_items = sorted(self.variant_cache.items(), key=lambda x: x[1][1])
            self.variant_cache = dict(sorted_items[-self.variant_cache_size:])
    
    def _clean_analysis_cache(self) -> None:
        """Remove expired entries from analysis cache and trim if too large."""
        now = time.time()
        
        # Remove expired entries
        self.analysis_cache = {
            k: v for k, v in self.analysis_cache.items() 
            if now - v[1] < self.analysis_cache_ttl
        }
        
        # Trim cache if it's too large
        if len(self.analysis_cache) > self.analysis_cache_size:
            # Sort by timestamp (oldest first) and keep only the newest entries
            sorted_items = sorted(self.analysis_cache.items(), key=lambda x: x[1][1])
            self.analysis_cache = dict(sorted_items[-self.analysis_cache_size:])
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.variant_cache = {}
        self.analysis_cache = {}
        logger.info("Cleared all caches")
    
    def invalidate_experiment_cache(self, experiment_id: str) -> None:
        """
        Invalidate all cached data for a specific experiment.
        
        Args:
            experiment_id: ID of the experiment to invalidate
        """
        # Remove from analysis cache
        if experiment_id in self.analysis_cache:
            del self.analysis_cache[experiment_id]
        
        # Remove from variant cache
        keys_to_remove = []
        for key in self.variant_cache:
            if key[0] == experiment_id:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.variant_cache[key]
            
        logger.debug(f"Invalidated cache for experiment {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.manager.get_experiment(experiment_id)
    
    def list_experiments(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List all experiments."""
        return self.manager.list_experiments(active_only)
    
    def create_experiment(self, *args, **kwargs) -> Experiment:
        """Create a new experiment."""
        experiment = self.manager.create_experiment(*args, **kwargs)
        # No need to invalidate cache for a new experiment
        return experiment
    
    def get_variant_for_user(
        self, 
        experiment_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the variant for a specific user in an experiment, with caching.
        
        Args:
            experiment_id: ID of the experiment
            user_id: ID of the user
            
        Returns:
            Dictionary with variant details, or None if not in experiment
        """
        cache_key = (experiment_id, user_id)
        
        # Check cache first
        if cache_key in self.variant_cache:
            variant, timestamp = self.variant_cache[cache_key]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.variant_cache_ttl:
                logger.debug(f"Variant cache hit for experiment {experiment_id}, user {user_id}")
                return variant
        
        # Cache miss, get from manager
        variant = self.manager.get_variant_for_user(experiment_id, user_id)
        
        # Cache the result if not None
        if variant is not None:
            self.variant_cache[cache_key] = (variant, time.time())
            
            # Clean cache if it might be too large
            if len(self.variant_cache) >= self.variant_cache_size:
                self._clean_variant_cache()
        
        return variant
    
    def record_conversion(
        self, 
        experiment_id: str, 
        variant_id: str, 
        value: float = 1.0
    ) -> bool:
        """Record a conversion for a variant in an experiment."""
        # Invalidate analysis cache since conversions affect analysis
        if experiment_id in self.analysis_cache:
            del self.analysis_cache[experiment_id]
            
        return self.manager.record_conversion(experiment_id, variant_id, value)
    
    def analyze_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze the results of an experiment, with caching.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with analysis results, or None if experiment not found
        """
        # Check cache first
        if experiment_id in self.analysis_cache:
            analysis, timestamp = self.analysis_cache[experiment_id]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.analysis_cache_ttl:
                logger.debug(f"Analysis cache hit for experiment {experiment_id}")
                return analysis
        
        # Cache miss, get from manager
        analysis = self.manager.analyze_experiment(experiment_id)
        
        # Cache the result if not None
        if analysis is not None:
            self.analysis_cache[experiment_id] = (analysis, time.time())
            
            # Clean cache if it might be too large
            if len(self.analysis_cache) >= self.analysis_cache_size:
                self._clean_analysis_cache()
        
        return analysis
    
    def end_experiment(self, experiment_id: str) -> bool:
        """End an experiment."""
        # Invalidate caches for this experiment
        self.invalidate_experiment_cache(experiment_id)
        
        return self.manager.end_experiment(experiment_id)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        # Invalidate caches for this experiment
        self.invalidate_experiment_cache(experiment_id)
        
        return self.manager.delete_experiment(experiment_id)
    
    # If the wrapped manager has a save method (e.g., AutoSavingManager), delegate to it
    def save(self) -> bool:
        """Save experiments if the wrapped manager supports it."""
        if hasattr(self.manager, 'save') and callable(self.manager.save):
            return self.manager.save()
        return False
    
    # If the wrapped manager has a load method, delegate to it and invalidate caches
    def load(self) -> bool:
        """Load experiments if the wrapped manager supports it."""
        if hasattr(self.manager, 'load') and callable(self.manager.load):
            result = self.manager.load()
            if result:
                self.clear_caches()  # Invalidate all caches after loading
            return result
        return False

class AutoSavingManager:
    """Wrapper for ExperimentManager that automatically saves to a storage backend."""
    
    def __init__(
        self, 
        manager: ExperimentManager, 
        backend: StorageBackend,
        autosave_interval: int = 60  # seconds
    ):
        """
        Initialize the auto-saving manager.
        
        Args:
            manager: The experiment manager to wrap
            backend: The storage backend to use
            autosave_interval: Interval in seconds between auto-saves
        """
        self.manager = manager
        self.backend = backend
        self.autosave_interval = autosave_interval
        self.last_save_time = 0
        
        # Load experiments from backend
        self.load()
    
    def save(self) -> bool:
        """Save experiments to the backend and update last save time."""
        success = self.backend.save_experiments(self.manager.experiments)
        if success:
            self.last_save_time = time.time()
        return success
    
    def load(self) -> bool:
        """Load experiments from the backend."""
        try:
            experiments = self.backend.load_experiments()
            if experiments:
                self.manager.experiments = experiments
                return True
            return False
        except Exception as e:
            logger.error("Failed to load experiments", error=str(e))
            return False
    
    def check_autosave(self) -> None:
        """Check if it's time to autosave and do so if needed."""
        if time.time() - self.last_save_time > self.autosave_interval:
            self.save()
    
    # Delegate methods to the wrapped manager with autosave checks
    
    def create_experiment(self, *args, **kwargs) -> Experiment:
        """Create an experiment and autosave."""
        experiment = self.manager.create_experiment(*args, **kwargs)
        self.check_autosave()
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment."""
        return self.manager.get_experiment(experiment_id)
    
    def list_experiments(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List experiments."""
        return self.manager.list_experiments(active_only)
    
    def get_variant_for_user(self, experiment_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get variant for a user."""
        return self.manager.get_variant_for_user(experiment_id, user_id)
    
    def record_conversion(self, experiment_id: str, variant_id: str, value: float = 1.0) -> bool:
        """Record a conversion and autosave."""
        result = self.manager.record_conversion(experiment_id, variant_id, value)
        self.check_autosave()
        return result
    
    def analyze_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Analyze an experiment."""
        return self.manager.analyze_experiment(experiment_id)
    
    def end_experiment(self, experiment_id: str) -> bool:
        """End an experiment and autosave."""
        result = self.manager.end_experiment(experiment_id)
        self.check_autosave()
        return result
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and autosave."""
        result = self.manager.delete_experiment(experiment_id)
        self.check_autosave()
        return result

# Global instance with JSON persistence
_default_manager = None
_sql_manager = None
_redis_manager = None
_caching_managers = {}

def get_persistent_manager(
    file_path: str = "data/experiments.json",
    autosave_interval: int = 60
) -> AutoSavingManager:
    """
    Get a global AutoSavingManager instance with JSON persistence.
    
    Args:
        file_path: Path to the JSON file
        autosave_interval: Interval in seconds between auto-saves
    
    Returns:
        AutoSavingManager instance
    """
    global _default_manager
    
    if _default_manager is None:
        from OpenInsight.experiments.experiment_service import get_experiment_manager
        
        backend = JSONFileBackend(file_path)
        _default_manager = AutoSavingManager(
            manager=get_experiment_manager(),
            backend=backend,
            autosave_interval=autosave_interval
        )
    
    return _default_manager 

def get_sql_persistent_manager(
    connection_string: str = "sqlite:///data/experiments.db",
    autosave_interval: int = 60,
    create_tables: bool = True
) -> AutoSavingManager:
    """
    Get a global AutoSavingManager instance with SQL database persistence.
    
    Args:
        connection_string: SQLAlchemy connection string
        autosave_interval: Interval in seconds between auto-saves
        create_tables: Whether to create tables if they don't exist
    
    Returns:
        AutoSavingManager instance
    """
    global _sql_manager
    
    if _sql_manager is None:
        from OpenInsight.experiments.experiment_service import get_experiment_manager
        
        try:
            backend = SQLDatabaseBackend(connection_string, create_tables)
            _sql_manager = AutoSavingManager(
                manager=get_experiment_manager(),
                backend=backend,
                autosave_interval=autosave_interval
            )
            logger.info(f"Created SQL persistent experiment manager with connection: {connection_string}")
        except Exception as e:
            logger.error(f"Failed to create SQL persistent manager. Falling back to JSON", error=str(e))
            # Fall back to JSON if SQL fails
            _sql_manager = get_persistent_manager()
    
    return _sql_manager

def get_redis_persistent_manager(
    redis_url: str = "redis://localhost:6379/0",
    key_prefix: str = "experiments:",
    autosave_interval: int = 10,
    expire_time: Optional[int] = None
) -> AutoSavingManager:
    """
    Get a global AutoSavingManager instance with Redis persistence.
    
    Redis offers higher performance than SQL for high-traffic applications
    and distributes data across nodes for scalability.
    
    Args:
        redis_url: Redis connection URL (format: redis://host:port/db)
        key_prefix: Prefix for Redis keys to avoid collisions
        autosave_interval: Interval in seconds between auto-saves
        expire_time: Optional expiration time in seconds for keys
    
    Returns:
        AutoSavingManager instance
    """
    global _redis_manager
    
    if _redis_manager is None:
        from OpenInsight.experiments.experiment_service import get_experiment_manager
        
        try:
            backend = RedisBackend(redis_url, key_prefix, expire_time)
            _redis_manager = AutoSavingManager(
                manager=get_experiment_manager(),
                backend=backend,
                autosave_interval=autosave_interval
            )
            logger.info(f"Created Redis persistent experiment manager with connection: {redis_url}")
        except Exception as e:
            logger.error(f"Failed to create Redis persistent manager. Falling back to JSON", error=str(e))
            # Fall back to JSON if Redis fails
            _redis_manager = get_persistent_manager()
    
    return _redis_manager

def get_caching_manager(
    base_manager_type: str = "json",
    variant_cache_size: int = 1000,
    variant_cache_ttl: int = 300,
    analysis_cache_size: int = 50,
    analysis_cache_ttl: int = 60,
    **kwargs
) -> CachingManager:
    """
    Get a CachingManager that wraps around a persistent manager for improved performance.
    
    This combines the benefits of persistent storage with fast in-memory caching
    for high-traffic applications.
    
    Args:
        base_manager_type: Type of base manager ('json', 'sql', or 'redis')
        variant_cache_size: Maximum number of variant assignments to cache
        variant_cache_ttl: Time-to-live for variant assignments in seconds
        analysis_cache_size: Maximum number of analysis results to cache
        analysis_cache_ttl: Time-to-live for analysis results in seconds
        **kwargs: Additional arguments for the base manager
        
    Returns:
        CachingManager instance
    """
    global _caching_managers
    
    # Create a key based on the parameters
    cache_key = (
        base_manager_type,
        variant_cache_size,
        variant_cache_ttl,
        analysis_cache_size,
        analysis_cache_ttl,
        str(sorted(kwargs.items()))  # Convert kwargs to a stable string representation
    )
    
    # Return existing manager if we have one with these parameters
    if cache_key in _caching_managers:
        return _caching_managers[cache_key]
    
    # Get the appropriate base manager
    if base_manager_type == 'json':
        file_path = kwargs.get('file_path', 'data/experiments.json')
        autosave_interval = kwargs.get('autosave_interval', 60)
        base_manager = get_persistent_manager(
            file_path=file_path,
            autosave_interval=autosave_interval
        )
    elif base_manager_type == 'sql':
        connection_string = kwargs.get('connection_string', 'sqlite:///data/experiments.db')
        autosave_interval = kwargs.get('autosave_interval', 60)
        create_tables = kwargs.get('create_tables', True)
        base_manager = get_sql_persistent_manager(
            connection_string=connection_string,
            autosave_interval=autosave_interval,
            create_tables=create_tables
        )
    elif base_manager_type == 'redis':
        redis_url = kwargs.get('redis_url', 'redis://localhost:6379/0')
        key_prefix = kwargs.get('key_prefix', 'experiments:')
        autosave_interval = kwargs.get('autosave_interval', 10)
        expire_time = kwargs.get('expire_time', None)
        base_manager = get_redis_persistent_manager(
            redis_url=redis_url,
            key_prefix=key_prefix,
            autosave_interval=autosave_interval,
            expire_time=expire_time
        )
    else:
        logger.warning(f"Unknown manager type: {base_manager_type}. Using JSON.")
        base_manager = get_persistent_manager()
    
    # Create and cache the caching manager
    caching_manager = CachingManager(
        manager=base_manager,
        variant_cache_size=variant_cache_size,
        variant_cache_ttl=variant_cache_ttl,
        analysis_cache_size=analysis_cache_size,
        analysis_cache_ttl=analysis_cache_ttl
    )
    
    _caching_managers[cache_key] = caching_manager
    
    logger.info(f"Created caching manager with {base_manager_type} persistence")
    return caching_manager 