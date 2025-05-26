"""
OpenSports Experiments Module

Advanced experimentation framework for sports analytics including:
- A/B testing and multivariate testing
- Causal inference analysis
- Multi-armed bandit optimization
- Statistical significance testing
"""

from opensports.experiments.ab_testing import ExperimentManager
from opensports.experiments.causal_analysis import CausalAnalyzer
from opensports.experiments.bandits import BanditOptimizer
from opensports.experiments.statistical_tests import StatisticalTester

__all__ = [
    "ExperimentManager",
    "CausalAnalyzer",
    "BanditOptimizer",
    "StatisticalTester",
] 