"""
OpenInsight Experiments package.

This package provides tools for running A/B tests and other types of experiments.
"""

from OpenInsight.experiments.experiment_service import (
    ExperimentType,
    ExperimentVariant,
    Experiment,
    ExperimentManager,
    get_experiment_manager
)

from OpenInsight.experiments.persistence import (
    StorageBackend,
    JSONFileBackend,
    SQLDatabaseBackend,
    RedisBackend,
    CachingManager,
    AutoSavingManager,
    get_persistent_manager,
    get_sql_persistent_manager,
    get_redis_persistent_manager,
    get_caching_manager
)

from OpenInsight.experiments.advanced_stats import (
    bayesian_ab_analysis,
    sequential_analysis,
    calculate_required_sample_size,
    segment_analysis,
    advanced_experiment_analysis,
    BayesianResult
)

__all__ = [
    # Core experiment service
    "ExperimentType",
    "ExperimentVariant",
    "Experiment",
    "ExperimentManager",
    "get_experiment_manager",
    
    # Persistence
    "StorageBackend",
    "JSONFileBackend",
    "SQLDatabaseBackend",
    "RedisBackend",
    "CachingManager",
    "AutoSavingManager",
    "get_persistent_manager",
    "get_sql_persistent_manager",
    "get_redis_persistent_manager",
    "get_caching_manager",
    
    # Advanced statistics
    "bayesian_ab_analysis",
    "sequential_analysis",
    "calculate_required_sample_size",
    "segment_analysis",
    "advanced_experiment_analysis",
    "BayesianResult"
]



