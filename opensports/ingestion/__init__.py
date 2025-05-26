"""
Sports data ingestion module for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

from opensports.ingestion.collector import SportsDataCollector
from opensports.ingestion.nba import NBADataCollector
from opensports.ingestion.nfl import NFLDataCollector
from opensports.ingestion.soccer import SoccerDataCollector
from opensports.ingestion.formula1 import Formula1DataCollector
from opensports.ingestion.realtime import RealTimeDataStreamer
from opensports.ingestion.odds import OddsDataCollector

__all__ = [
    "SportsDataCollector",
    "NBADataCollector",
    "NFLDataCollector", 
    "SoccerDataCollector",
    "Formula1DataCollector",
    "RealTimeDataStreamer",
    "OddsDataCollector",
] 