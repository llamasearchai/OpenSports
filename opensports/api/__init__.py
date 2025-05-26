"""
OpenSports API Module

FastAPI-based REST API for the OpenSports platform including:
- Game analysis and predictions
- Player and team statistics
- Real-time data streaming
- AI-powered insights
"""

from opensports.api.main import app
from opensports.api.endpoints import (
    games,
    players,
    teams,
    analytics,
    predictions,
    realtime,
    agents
)

__all__ = [
    "app",
    "games",
    "players", 
    "teams",
    "analytics",
    "predictions",
    "realtime",
    "agents",
] 