"""
Main sports data collector for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from opensports.core.config import settings
from opensports.core.logging import get_logger, LoggerMixin, PerformanceLogger
from opensports.core.database import get_database
from opensports.core.cache import get_cache, cache_async_result
from opensports.ingestion.nba import NBADataCollector
from opensports.ingestion.nfl import NFLDataCollector
from opensports.ingestion.soccer import SoccerDataCollector
from opensports.ingestion.formula1 import Formula1DataCollector
from opensports.ingestion.odds import OddsDataCollector

logger = get_logger(__name__)


class SportsDataCollector(LoggerMixin):
    """
    Main sports data collector that orchestrates data collection from multiple sources.
    """
    
    def __init__(self):
        self.database = get_database()
        self.cache = get_cache()
        
        # Initialize sport-specific collectors
        self.nba_collector = NBADataCollector()
        self.nfl_collector = NFLDataCollector()
        self.soccer_collector = SoccerDataCollector()
        self.f1_collector = Formula1DataCollector()
        self.odds_collector = OddsDataCollector()
        
        self.session = None
        self.logger.info("SportsDataCollector initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": "OpenSports/1.0.0 (nikjois@llamaearch.ai)",
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @cache_async_result(ttl=3600, key_prefix="games")
    async def collect_games(
        self,
        sport: str,
        date_range: Optional[tuple] = None,
        season: Optional[str] = None,
        team: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect game data for a specific sport.
        
        Args:
            sport: Sport type (nba, nfl, soccer, formula1)
            date_range: Tuple of (start_date, end_date)
            season: Season identifier
            team: Specific team to filter by
            
        Returns:
            List of game data dictionaries
        """
        with PerformanceLogger(f"collect_games_{sport}", self.logger):
            try:
                collector = self._get_sport_collector(sport)
                if not collector:
                    raise ValueError(f"Unsupported sport: {sport}")
                
                games = await collector.collect_games(
                    date_range=date_range,
                    season=season,
                    team=team
                )
                
                # Store in database
                if games:
                    self.database.insert_data("games", games)
                    self.logger.info(
                        "Games collected and stored",
                        sport=sport,
                        count=len(games)
                    )
                
                return games
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect games",
                    sport=sport,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    @cache_async_result(ttl=1800, key_prefix="players")
    async def collect_players(
        self,
        sport: str,
        team: Optional[str] = None,
        position: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Collect player data for a specific sport.
        
        Args:
            sport: Sport type
            team: Specific team to filter by
            position: Player position to filter by
            active_only: Whether to include only active players
            
        Returns:
            List of player data dictionaries
        """
        with PerformanceLogger(f"collect_players_{sport}", self.logger):
            try:
                collector = self._get_sport_collector(sport)
                if not collector:
                    raise ValueError(f"Unsupported sport: {sport}")
                
                players = await collector.collect_players(
                    team=team,
                    position=position,
                    active_only=active_only
                )
                
                # Store in database
                if players:
                    self.database.insert_data("players", players)
                    self.logger.info(
                        "Players collected and stored",
                        sport=sport,
                        count=len(players)
                    )
                
                return players
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect players",
                    sport=sport,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    @cache_async_result(ttl=900, key_prefix="player_stats")
    async def collect_player_stats(
        self,
        sport: str,
        player_id: str,
        season: Optional[str] = None,
        game_type: str = "regular",
    ) -> List[Dict[str, Any]]:
        """
        Collect detailed player statistics.
        
        Args:
            sport: Sport type
            player_id: Player identifier
            season: Season to collect stats for
            game_type: Type of games (regular, playoffs, etc.)
            
        Returns:
            List of player statistics
        """
        with PerformanceLogger(f"collect_player_stats_{sport}", self.logger):
            try:
                collector = self._get_sport_collector(sport)
                if not collector:
                    raise ValueError(f"Unsupported sport: {sport}")
                
                stats = await collector.collect_player_stats(
                    player_id=player_id,
                    season=season,
                    game_type=game_type
                )
                
                # Store in database
                if stats:
                    self.database.insert_data("player_stats", stats)
                    self.logger.info(
                        "Player stats collected and stored",
                        sport=sport,
                        player_id=player_id,
                        count=len(stats)
                    )
                
                return stats
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect player stats",
                    sport=sport,
                    player_id=player_id,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    @cache_async_result(ttl=1800, key_prefix="teams")
    async def collect_teams(
        self,
        sport: str,
        conference: Optional[str] = None,
        division: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect team data for a specific sport.
        
        Args:
            sport: Sport type
            conference: Conference to filter by
            division: Division to filter by
            
        Returns:
            List of team data dictionaries
        """
        with PerformanceLogger(f"collect_teams_{sport}", self.logger):
            try:
                collector = self._get_sport_collector(sport)
                if not collector:
                    raise ValueError(f"Unsupported sport: {sport}")
                
                teams = await collector.collect_teams(
                    conference=conference,
                    division=division
                )
                
                # Store in database
                if teams:
                    self.database.insert_data("teams", teams)
                    self.logger.info(
                        "Teams collected and stored",
                        sport=sport,
                        count=len(teams)
                    )
                
                return teams
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect teams",
                    sport=sport,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def collect_live_data(
        self,
        sport: str,
        game_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Collect live game data.
        
        Args:
            sport: Sport type
            game_id: Specific game ID (if None, collects all live games)
            
        Returns:
            Live game data
        """
        with PerformanceLogger(f"collect_live_data_{sport}", self.logger):
            try:
                collector = self._get_sport_collector(sport)
                if not collector:
                    raise ValueError(f"Unsupported sport: {sport}")
                
                live_data = await collector.collect_live_data(game_id=game_id)
                
                self.logger.info(
                    "Live data collected",
                    sport=sport,
                    game_id=game_id
                )
                
                return live_data
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect live data",
                    sport=sport,
                    game_id=game_id,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def collect_odds_data(
        self,
        sport: str,
        market_type: str = "h2h",
        region: str = "us",
    ) -> List[Dict[str, Any]]:
        """
        Collect betting odds data.
        
        Args:
            sport: Sport type
            market_type: Type of betting market
            region: Geographic region for odds
            
        Returns:
            List of odds data
        """
        with PerformanceLogger(f"collect_odds_{sport}", self.logger):
            try:
                odds_data = await self.odds_collector.collect_odds(
                    sport=sport,
                    market_type=market_type,
                    region=region
                )
                
                self.logger.info(
                    "Odds data collected",
                    sport=sport,
                    market_type=market_type,
                    count=len(odds_data)
                )
                
                return odds_data
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect odds data",
                    sport=sport,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def collect_historical_data(
        self,
        sport: str,
        start_date: datetime,
        end_date: datetime,
        data_types: List[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect historical data for a date range.
        
        Args:
            sport: Sport type
            start_date: Start date for data collection
            end_date: End date for data collection
            data_types: Types of data to collect (games, players, stats, etc.)
            
        Returns:
            Dictionary with collected data by type
        """
        data_types = data_types or ["games", "players", "stats"]
        
        with PerformanceLogger(f"collect_historical_{sport}", self.logger):
            try:
                collector = self._get_sport_collector(sport)
                if not collector:
                    raise ValueError(f"Unsupported sport: {sport}")
                
                historical_data = {}
                
                # Collect different types of data
                for data_type in data_types:
                    if data_type == "games":
                        historical_data["games"] = await collector.collect_games(
                            date_range=(start_date, end_date)
                        )
                    elif data_type == "players":
                        historical_data["players"] = await collector.collect_players()
                    elif data_type == "stats":
                        # This would need to be implemented per sport
                        pass
                
                self.logger.info(
                    "Historical data collected",
                    sport=sport,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    data_types=data_types
                )
                
                return historical_data
                
            except Exception as e:
                self.logger.error(
                    "Failed to collect historical data",
                    sport=sport,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def bulk_data_collection(
        self,
        sports: List[str],
        data_types: List[str] = None,
        parallel: bool = True,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Collect data for multiple sports in parallel.
        
        Args:
            sports: List of sports to collect data for
            data_types: Types of data to collect
            parallel: Whether to collect in parallel
            
        Returns:
            Nested dictionary with data by sport and type
        """
        data_types = data_types or ["games", "players", "teams"]
        
        with PerformanceLogger("bulk_data_collection", self.logger):
            try:
                if parallel:
                    # Collect data in parallel
                    tasks = []
                    for sport in sports:
                        for data_type in data_types:
                            if data_type == "games":
                                task = self.collect_games(sport)
                            elif data_type == "players":
                                task = self.collect_players(sport)
                            elif data_type == "teams":
                                task = self.collect_teams(sport)
                            else:
                                continue
                            
                            tasks.append((sport, data_type, task))
                    
                    # Execute all tasks
                    results = await asyncio.gather(
                        *[task for _, _, task in tasks],
                        return_exceptions=True
                    )
                    
                    # Organize results
                    bulk_data = {}
                    for i, (sport, data_type, _) in enumerate(tasks):
                        if sport not in bulk_data:
                            bulk_data[sport] = {}
                        
                        result = results[i]
                        if isinstance(result, Exception):
                            self.logger.error(
                                "Task failed in bulk collection",
                                sport=sport,
                                data_type=data_type,
                                error=str(result)
                            )
                            bulk_data[sport][data_type] = []
                        else:
                            bulk_data[sport][data_type] = result
                
                else:
                    # Collect data sequentially
                    bulk_data = {}
                    for sport in sports:
                        bulk_data[sport] = {}
                        for data_type in data_types:
                            try:
                                if data_type == "games":
                                    bulk_data[sport][data_type] = await self.collect_games(sport)
                                elif data_type == "players":
                                    bulk_data[sport][data_type] = await self.collect_players(sport)
                                elif data_type == "teams":
                                    bulk_data[sport][data_type] = await self.collect_teams(sport)
                            except Exception as e:
                                self.logger.error(
                                    "Failed to collect data in bulk",
                                    sport=sport,
                                    data_type=data_type,
                                    error=str(e)
                                )
                                bulk_data[sport][data_type] = []
                
                self.logger.info(
                    "Bulk data collection completed",
                    sports=sports,
                    data_types=data_types,
                    parallel=parallel
                )
                
                return bulk_data
                
            except Exception as e:
                self.logger.error(
                    "Bulk data collection failed",
                    sports=sports,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    def _get_sport_collector(self, sport: str):
        """Get the appropriate collector for a sport."""
        collectors = {
            "nba": self.nba_collector,
            "nfl": self.nfl_collector,
            "soccer": self.soccer_collector,
            "football": self.soccer_collector,  # Alias
            "formula1": self.f1_collector,
            "f1": self.f1_collector,  # Alias
        }
        return collectors.get(sport.lower())
    
    def get_supported_sports(self) -> List[str]:
        """Get list of supported sports."""
        return ["nba", "nfl", "soccer", "formula1"]
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get status of data collection operations."""
        try:
            summary = self.database.get_sports_data_summary()
            
            status = {
                "supported_sports": self.get_supported_sports(),
                "database_summary": summary,
                "cache_stats": self.cache.get_stats(),
                "last_updated": datetime.now().isoformat(),
            }
            
            return status
            
        except Exception as e:
            self.logger.error("Failed to get collection status", error=str(e))
            return {
                "error": str(e),
                "last_updated": datetime.now().isoformat(),
            } 