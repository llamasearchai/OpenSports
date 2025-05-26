"""
OpenSports Machine Learning Module

Advanced machine learning pipeline for sports analytics including automated
model training, hyperparameter optimization, and deployment capabilities.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

from .pipeline import MLPipeline, PipelineConfig
from .models import (
    PlayerPerformancePredictor,
    GameOutcomePredictor,
    InjuryRiskPredictor,
    TeamStrengthPredictor,
    PlayerValuePredictor
)
from .features import (
    FeatureEngineer,
    FeatureSelector,
    FeatureTransformer,
    AdvancedFeatures
)
from .training import (
    ModelTrainer,
    HyperparameterOptimizer,
    CrossValidator,
    EnsembleTrainer
)
from .evaluation import (
    ModelEvaluator,
    PerformanceMetrics,
    ModelComparator,
    ValidationReports
)
from .deployment import (
    ModelDeployer,
    ModelRegistry,
    ModelMonitor,
    ABTestManager
)
from .automl import AutoMLEngine, AutoMLConfig

__all__ = [
    'MLPipeline',
    'PipelineConfig',
    'PlayerPerformancePredictor',
    'GameOutcomePredictor',
    'InjuryRiskPredictor',
    'TeamStrengthPredictor',
    'PlayerValuePredictor',
    'FeatureEngineer',
    'FeatureSelector',
    'FeatureTransformer',
    'AdvancedFeatures',
    'ModelTrainer',
    'HyperparameterOptimizer',
    'CrossValidator',
    'EnsembleTrainer',
    'ModelEvaluator',
    'PerformanceMetrics',
    'ModelComparator',
    'ValidationReports',
    'ModelDeployer',
    'ModelRegistry',
    'ModelMonitor',
    'ABTestManager',
    'AutoMLEngine',
    'AutoMLConfig'
] 