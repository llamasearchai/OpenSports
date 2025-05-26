"""
OpenSports Advanced Analytics Module

Comprehensive sports analytics with advanced statistical modeling,
predictive analytics, and performance optimization.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

from .performance import PerformanceAnalyzer, PlayerPerformanceModel, TeamPerformanceModel
from .predictive import PredictiveModels, GameOutcomePredictor, PlayerInjuryPredictor
from .optimization import StrategyOptimizer, LineupOptimizer, TrainingOptimizer
from .advanced_stats import AdvancedStatsCalculator, ExpectedValueModels
from .network import SportsNetworkAnalysis, PlayerNetworkAnalyzer
from .time_series import TimeSeriesAnalyzer, SeasonalAnalyzer
from .clustering import PlayerClustering, TeamClustering, PlayStyleAnalyzer
from .causal import CausalAnalyzer, ImpactAnalyzer
from .simulation import GameSimulator, SeasonSimulator, MonteCarloAnalyzer

__all__ = [
    'PerformanceAnalyzer',
    'PlayerPerformanceModel',
    'TeamPerformanceModel',
    'PredictiveModels',
    'GameOutcomePredictor',
    'PlayerInjuryPredictor',
    'StrategyOptimizer',
    'LineupOptimizer',
    'TrainingOptimizer',
    'AdvancedStatsCalculator',
    'ExpectedValueModels',
    'SportsNetworkAnalysis',
    'PlayerNetworkAnalyzer',
    'TimeSeriesAnalyzer',
    'SeasonalAnalyzer',
    'PlayerClustering',
    'TeamClustering',
    'PlayStyleAnalyzer',
    'CausalAnalyzer',
    'ImpactAnalyzer',
    'GameSimulator',
    'SeasonSimulator',
    'MonteCarloAnalyzer'
]

__version__ = "1.0.0" 