"""
OpenSports: Elite Global Sports Data Analytics Platform

A cutting-edge sports analytics platform combining advanced machine learning,
real-time data processing, and AI agents for professional sports organizations.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamaearch.ai"
__license__ = "MIT"

# Core imports for easy access
from opensports.core.config import settings
from opensports.core.logging import get_logger

# Main module imports
from opensports.ingestion import SportsDataCollector
from opensports.modeling import (
    PlayerPerformanceModel,
    GameOutcomePredictor,
    LeadScoringService,
    TimeForecaster,
)
from opensports.segmentation import AudienceSegmenter
from opensports.experiments import ExperimentManager, CausalAnalyzer
from opensports.agents import SportsAnalystAgent
from opensports.realtime import LiveGameAnalyzer

__all__ = [
    # Core
    "settings",
    "get_logger",
    # Data Collection
    "SportsDataCollector",
    # Modeling
    "PlayerPerformanceModel",
    "GameOutcomePredictor", 
    "LeadScoringService",
    "TimeForecaster",
    # Segmentation
    "AudienceSegmenter",
    # Experiments
    "ExperimentManager",
    "CausalAnalyzer",
    # AI Agents
    "SportsAnalystAgent",
    # Real-time
    "LiveGameAnalyzer",
] 