"""
NBA data collector for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from opensports.core.config import settings
from opensports.core.logging import get_logger, LoggerMixin, PerformanceLogger
from opensports.core.cache import cache_async_result

logger = get_logger(__name__)


class NBADataCollector(LoggerMixin):
    """
    NBA data collector using multiple data sources including NBA API,
    ESPN, and SportRadar.
    """
    
    def __init__(self):
        self.base_urls = {
            "nba_api": "https://stats.nba.com/stats",
            "espn": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba",
            "balldontlie": "https://www.balldontlie.io/api/v1",
        }
        
        self.headers = {
            "User-Agent": "OpenSports/1.0.0 (nikjois@llamaearch.ai)",
            "Accept": "application/json",
            "Referer": "https://stats.nba.com/",
        }
        
        self.session = None
        self.logger.info("NBADataCollector initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @cache_async_result(ttl=3600, key_prefix="nba_games")
    async def collect_games(
        self,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        season: Optional[str] = None,
        team: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect NBA game data.
        
        Args:
            date_range: Tuple of (start_date, end_date)
            season: Season in format "2023-24"
            team: Team abbreviation to filter by
            
        Returns:
            List of game data dictionaries
        """
        with PerformanceLogger("collect_nba_games", self.logger):
            try:
                if not self.session:
                    async with self:
                        return await self._fetch_games(date_range, season, team)
                else:
                    return await self._fetch_games(date_range, season, team)
                    
            except Exception as e:
                self.logger.error("Failed to collect NBA games", error=str(e))
                raise
    
    async def _fetch_games(
        self,
        date_range: Optional[Tuple[datetime, datetime]],
        season: Optional[str],
        team: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Internal method to fetch games from multiple sources."""
        games = []
        
        # Try ESPN API first (more reliable)
        try:
            espn_games = await self._fetch_espn_games(date_range, season)
            games.extend(espn_games)
        except Exception as e:
            self.logger.warning("ESPN API failed", error=str(e))
        
        # Try Ball Don't Lie API as backup
        if not games:
            try:
                bdl_games = await self._fetch_balldontlie_games(date_range, season)
                games.extend(bdl_games)
            except Exception as e:
                self.logger.warning("Ball Don't Lie API failed", error=str(e))
        
        # Filter by team if specified
        if team and games:
            team_upper = team.upper()
            games = [
                game for game in games
                if game.get("home_team", "").upper() == team_upper or
                   game.get("away_team", "").upper() == team_upper
            ]
        
        self.logger.info("NBA games collected", count=len(games))
        return games
    
    async def _fetch_espn_games(
        self,
        date_range: Optional[Tuple[datetime, datetime]],
        season: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fetch games from ESPN API."""
        games = []
        
        if date_range:
            start_date, end_date = date_range
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                url = f"{self.base_urls['espn']}/scoreboard"
                
                params = {"dates": date_str}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for event in data.get("events", []):
                            game = self._parse_espn_game(event)
                            if game:
                                games.append(game)
                
                current_date += timedelta(days=1)
                await asyncio.sleep(0.1)  # Rate limiting
        
        return games
    
    def _parse_espn_game(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse ESPN game data into standard format."""
        try:
            competitions = event.get("competitions", [])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) != 2:
                return None
            
            # Determine home and away teams
            home_team = None
            away_team = None
            home_score = 0
            away_score = 0
            
            for competitor in competitors:
                team_info = competitor.get("team", {})
                team_abbr = team_info.get("abbreviation", "")
                score = int(competitor.get("score", 0))
                
                if competitor.get("homeAway") == "home":
                    home_team = team_abbr
                    home_score = score
                else:
                    away_team = team_abbr
                    away_score = score
            
            game_date = event.get("date", "")
            if game_date:
                game_date = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
            
            status = competition.get("status", {})
            game_status = status.get("type", {}).get("description", "Unknown")
            
            return {
                "id": event.get("id", ""),
                "sport": "nba",
                "home_team": home_team,
                "away_team": away_team,
                "game_date": game_date.isoformat() if isinstance(game_date, datetime) else game_date,
                "season": self._extract_season_from_date(game_date),
                "home_score": home_score,
                "away_score": away_score,
                "status": game_status,
                "venue": competition.get("venue", {}).get("fullName", ""),
                "attendance": competition.get("attendance", 0),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.warning("Failed to parse ESPN game", error=str(e))
            return None
    
    async def _fetch_balldontlie_games(
        self,
        date_range: Optional[Tuple[datetime, datetime]],
        season: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fetch games from Ball Don't Lie API."""
        games = []
        
        url = f"{self.base_urls['balldontlie']}/games"
        params = {"per_page": 100}
        
        if season:
            # Extract year from season format "2023-24"
            year = int(season.split("-")[0])
            params["seasons[]"] = year
        
        if date_range:
            start_date, end_date = date_range
            params["start_date"] = start_date.strftime("%Y-%m-%d")
            params["end_date"] = end_date.strftime("%Y-%m-%d")
        
        page = 1
        while True:
            params["page"] = page
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for game_data in data.get("data", []):
                        game = self._parse_balldontlie_game(game_data)
                        if game:
                            games.append(game)
                    
                    # Check if there are more pages
                    meta = data.get("meta", {})
                    if page >= meta.get("total_pages", 1):
                        break
                    
                    page += 1
                    await asyncio.sleep(0.1)  # Rate limiting
                else:
                    break
        
        return games
    
    def _parse_balldontlie_game(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Ball Don't Lie game data into standard format."""
        try:
            home_team = game_data.get("home_team", {}).get("abbreviation", "")
            visitor_team = game_data.get("visitor_team", {}).get("abbreviation", "")
            
            game_date = game_data.get("date", "")
            if game_date:
                game_date = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
            
            return {
                "id": str(game_data.get("id", "")),
                "sport": "nba",
                "home_team": home_team,
                "away_team": visitor_team,
                "game_date": game_date.isoformat() if isinstance(game_date, datetime) else game_date,
                "season": str(game_data.get("season", "")),
                "home_score": game_data.get("home_team_score", 0) or 0,
                "away_score": game_data.get("visitor_team_score", 0) or 0,
                "status": "Final" if game_data.get("home_team_score") else "Scheduled",
                "venue": "",
                "attendance": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.warning("Failed to parse Ball Don't Lie game", error=str(e))
            return None
    
    @cache_async_result(ttl=1800, key_prefix="nba_players")
    async def collect_players(
        self,
        team: Optional[str] = None,
        position: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Collect NBA player data.
        
        Args:
            team: Team abbreviation to filter by
            position: Position to filter by
            active_only: Whether to include only active players
            
        Returns:
            List of player data dictionaries
        """
        with PerformanceLogger("collect_nba_players", self.logger):
            try:
                if not self.session:
                    async with self:
                        return await self._fetch_players(team, position, active_only)
                else:
                    return await self._fetch_players(team, position, active_only)
                    
            except Exception as e:
                self.logger.error("Failed to collect NBA players", error=str(e))
                raise
    
    async def _fetch_players(
        self,
        team: Optional[str],
        position: Optional[str],
        active_only: bool,
    ) -> List[Dict[str, Any]]:
        """Internal method to fetch players."""
        players = []
        
        # Use Ball Don't Lie API for players
        url = f"{self.base_urls['balldontlie']}/players"
        params = {"per_page": 100}
        
        page = 1
        while True:
            params["page"] = page
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for player_data in data.get("data", []):
                        player = self._parse_player_data(player_data)
                        if player:
                            # Apply filters
                            if team and player.get("team", "").upper() != team.upper():
                                continue
                            if position and player.get("position", "").upper() != position.upper():
                                continue
                            
                            players.append(player)
                    
                    # Check if there are more pages
                    meta = data.get("meta", {})
                    if page >= meta.get("total_pages", 1):
                        break
                    
                    page += 1
                    await asyncio.sleep(0.1)  # Rate limiting
                else:
                    break
        
        self.logger.info("NBA players collected", count=len(players))
        return players
    
    def _parse_player_data(self, player_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse player data into standard format."""
        try:
            team_info = player_data.get("team", {})
            
            return {
                "id": str(player_data.get("id", "")),
                "name": f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip(),
                "team": team_info.get("abbreviation", ""),
                "position": player_data.get("position", ""),
                "sport": "nba",
                "age": 0,  # Not available in this API
                "height": self._parse_height(player_data.get("height_feet"), player_data.get("height_inches")),
                "weight": player_data.get("weight_pounds", 0) or 0,
                "jersey_number": 0,  # Not available in this API
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.warning("Failed to parse player data", error=str(e))
            return None
    
    def _parse_height(self, feet: Optional[int], inches: Optional[int]) -> float:
        """Convert height to total inches."""
        if feet is None:
            return 0.0
        
        total_inches = feet * 12
        if inches:
            total_inches += inches
        
        return float(total_inches)
    
    @cache_async_result(ttl=1800, key_prefix="nba_teams")
    async def collect_teams(
        self,
        conference: Optional[str] = None,
        division: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect NBA team data.
        
        Args:
            conference: Conference to filter by (Eastern, Western)
            division: Division to filter by
            
        Returns:
            List of team data dictionaries
        """
        with PerformanceLogger("collect_nba_teams", self.logger):
            try:
                if not self.session:
                    async with self:
                        return await self._fetch_teams(conference, division)
                else:
                    return await self._fetch_teams(conference, division)
                    
            except Exception as e:
                self.logger.error("Failed to collect NBA teams", error=str(e))
                raise
    
    async def _fetch_teams(
        self,
        conference: Optional[str],
        division: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Internal method to fetch teams."""
        teams = []
        
        # Use Ball Don't Lie API for teams
        url = f"{self.base_urls['balldontlie']}/teams"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                
                for team_data in data.get("data", []):
                    team = self._parse_team_data(team_data)
                    if team:
                        # Apply filters
                        if conference and team.get("conference", "").lower() != conference.lower():
                            continue
                        if division and team.get("division", "").lower() != division.lower():
                            continue
                        
                        teams.append(team)
        
        self.logger.info("NBA teams collected", count=len(teams))
        return teams
    
    def _parse_team_data(self, team_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse team data into standard format."""
        try:
            return {
                "id": str(team_data.get("id", "")),
                "name": team_data.get("full_name", ""),
                "city": team_data.get("city", ""),
                "abbreviation": team_data.get("abbreviation", ""),
                "sport": "nba",
                "conference": team_data.get("conference", ""),
                "division": team_data.get("division", ""),
                "founded": 0,  # Not available in this API
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.warning("Failed to parse team data", error=str(e))
            return None
    
    async def collect_player_stats(
        self,
        player_id: str,
        season: Optional[str] = None,
        game_type: str = "regular",
    ) -> List[Dict[str, Any]]:
        """
        Collect detailed player statistics.
        
        Args:
            player_id: Player identifier
            season: Season to collect stats for
            game_type: Type of games (regular, playoffs)
            
        Returns:
            List of player statistics
        """
        with PerformanceLogger("collect_nba_player_stats", self.logger):
            try:
                # This would require more advanced API access
                # For now, return empty list
                self.logger.info("Player stats collection not yet implemented")
                return []
                
            except Exception as e:
                self.logger.error("Failed to collect player stats", error=str(e))
                raise
    
    async def collect_live_data(self, game_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect live NBA game data.
        
        Args:
            game_id: Specific game ID
            
        Returns:
            Live game data
        """
        with PerformanceLogger("collect_nba_live_data", self.logger):
            try:
                # Use ESPN API for live data
                url = f"{self.base_urls['espn']}/scoreboard"
                
                if not self.session:
                    async with self:
                        return await self._fetch_live_data(url, game_id)
                else:
                    return await self._fetch_live_data(url, game_id)
                    
            except Exception as e:
                self.logger.error("Failed to collect live NBA data", error=str(e))
                raise
    
    async def _fetch_live_data(self, url: str, game_id: Optional[str]) -> Dict[str, Any]:
        """Internal method to fetch live data."""
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                
                live_games = []
                for event in data.get("events", []):
                    if game_id and event.get("id") != game_id:
                        continue
                    
                    status = event.get("status", {})
                    if status.get("type", {}).get("state") == "in":
                        live_game = self._parse_espn_game(event)
                        if live_game:
                            # Add live-specific data
                            live_game["period"] = status.get("period", 0)
                            live_game["clock"] = status.get("displayClock", "")
                            live_games.append(live_game)
                
                return {
                    "live_games": live_games,
                    "last_updated": datetime.now().isoformat(),
                }
            
            return {"live_games": [], "error": "Failed to fetch live data"}
    
    def _extract_season_from_date(self, game_date: datetime) -> str:
        """Extract season from game date."""
        if not isinstance(game_date, datetime):
            return ""
        
        year = game_date.year
        month = game_date.month
        
        # NBA season runs from October to June
        if month >= 10:
            return f"{year}-{str(year + 1)[2:]}"
        else:
            return f"{year - 1}-{str(year)[2:]}" 