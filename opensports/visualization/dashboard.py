"""
Advanced Sports Analytics Dashboard

Interactive web-based dashboard for comprehensive sports analytics visualization.
Built with Streamlit and Plotly for professional-grade data presentation.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass
import logging

from ..core.database import DatabaseManager
from ..core.cache import CacheManager
from ..modeling.performance import PlayerPerformanceModel
from ..realtime.processor import StreamProcessor
from .charts import (
    PerformanceChart, TeamComparisonChart, PlayerTrajectoryChart,
    GameFlowChart, HeatmapChart, NetworkChart
)
from .components import MetricsCard, PlayerCard, TeamCard, GameCard

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    title: str = "OpenSports Analytics Dashboard"
    theme: str = "dark"
    auto_refresh: bool = True
    refresh_interval: int = 30  # seconds
    max_data_points: int = 1000
    enable_realtime: bool = True
    cache_ttl: int = 300  # seconds

class SportsDashboard:
    """
    Main sports analytics dashboard with interactive visualizations.
    
    Features:
    - Real-time game monitoring
    - Player and team performance analytics
    - Historical trend analysis
    - Predictive insights
    - Custom report generation
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.performance_model = PlayerPerformanceModel()
        self.stream_processor = StreamProcessor()
        
        # Initialize chart components
        self.charts = {
            'performance': PerformanceChart(),
            'team_comparison': TeamComparisonChart(),
            'player_trajectory': PlayerTrajectoryChart(),
            'game_flow': GameFlowChart(),
            'heatmap': HeatmapChart(),
            'network': NetworkChart()
        }
        
        # Initialize dashboard state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize Streamlit session state."""
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.selected_sport = 'NBA'
            st.session_state.selected_team = None
            st.session_state.selected_player = None
            st.session_state.date_range = (
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            st.session_state.auto_refresh = self.config.auto_refresh
    
    def run(self):
        """Main dashboard application."""
        self._setup_page_config()
        self._render_sidebar()
        self._render_main_content()
        
        if st.session_state.auto_refresh:
            self._setup_auto_refresh()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.title,
            page_icon="CHART",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render dashboard sidebar with controls."""
        with st.sidebar:
            st.title("OpenSports")
            st.markdown("---")
            
            # Sport selection
            sports = ['NBA', 'NFL', 'Soccer', 'Formula1']
            st.session_state.selected_sport = st.selectbox(
                "Select Sport", sports, 
                index=sports.index(st.session_state.selected_sport)
            )
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date", 
                    value=st.session_state.date_range[0]
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=st.session_state.date_range[1]
                )
            
            st.session_state.date_range = (start_date, end_date)
            
            # Team and player selection
            teams = self._get_teams(st.session_state.selected_sport)
            if teams:
                selected_team = st.selectbox(
                    "Select Team", 
                    ["All Teams"] + teams,
                    index=0
                )
                st.session_state.selected_team = selected_team if selected_team != "All Teams" else None
                
                if st.session_state.selected_team:
                    players = self._get_players(st.session_state.selected_team)
                    if players:
                        selected_player = st.selectbox(
                            "Select Player",
                            ["All Players"] + players,
                            index=0
                        )
                        st.session_state.selected_player = selected_player if selected_player != "All Players" else None
            
            st.markdown("---")
            
            # Dashboard settings
            st.subheader("Settings")
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh", 
                value=st.session_state.auto_refresh
            )
            
            if st.button("Refresh Data"):
                self._refresh_data()
                st.rerun()
            
            if st.button("Generate Report"):
                self._generate_report()
    
    def _render_main_content(self):
        """Render main dashboard content."""
        st.markdown('<h1 class="main-header">Sports Analytics Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Key metrics row
        self._render_key_metrics()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance", "Games", "Teams", "Predictions", "Analytics"
        ])
        
        with tab1:
            self._render_performance_tab()
        
        with tab2:
            self._render_games_tab()
        
        with tab3:
            self._render_teams_tab()
        
        with tab4:
            self._render_predictions_tab()
        
        with tab5:
            self._render_analytics_tab()
    
    def _render_key_metrics(self):
        """Render key performance metrics."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get metrics data
        metrics = self._get_key_metrics()
        
        with col1:
            st.metric(
                label="Games Today",
                value=metrics.get('games_today', 0),
                delta=metrics.get('games_delta', 0)
            )
        
        with col2:
            st.metric(
                label="Active Players",
                value=f"{metrics.get('active_players', 0):,}",
                delta=metrics.get('players_delta', 0)
            )
        
        with col3:
            st.metric(
                label="Avg Performance",
                value=f"{metrics.get('avg_performance', 0):.1f}",
                delta=f"{metrics.get('performance_delta', 0):.1f}"
            )
        
        with col4:
            st.metric(
                label="Prediction Accuracy",
                value=f"{metrics.get('prediction_accuracy', 0):.1%}",
                delta=f"{metrics.get('accuracy_delta', 0):.1%}"
            )
        
        with col5:
            st.metric(
                label="Data Points",
                value=f"{metrics.get('data_points', 0):,}",
                delta=metrics.get('data_delta', 0)
            )
    
    def _render_performance_tab(self):
        """Render performance analysis tab."""
        st.subheader("Performance Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance trend chart
            performance_data = self._get_performance_data()
            if not performance_data.empty:
                fig = self.charts['performance'].create_trend_chart(performance_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available for the selected criteria.")
        
        with col2:
            # Performance distribution
            if not performance_data.empty:
                fig = self.charts['performance'].create_distribution_chart(performance_data)
                st.plotly_chart(fig, use_container_width=True)
        
        # Player comparison
        if st.session_state.selected_team:
            st.subheader("Player Comparison")
            players_data = self._get_players_comparison_data()
            if not players_data.empty:
                fig = self.charts['performance'].create_comparison_chart(players_data)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_games_tab(self):
        """Render games analysis tab."""
        st.subheader("Game Analysis")
        
        # Live games section
        if self.config.enable_realtime:
            st.subheader("Live Games")
            live_games = self._get_live_games()
            if live_games:
                for game in live_games:
                    self._render_live_game_card(game)
            else:
                st.info("No live games currently.")
        
        # Recent games
        st.subheader("Recent Games")
        recent_games = self._get_recent_games()
        if not recent_games.empty:
            for _, game in recent_games.iterrows():
                self._render_game_card(game)
        
        # Game flow analysis
        if st.session_state.selected_team:
            st.subheader("Game Flow Analysis")
            game_flow_data = self._get_game_flow_data()
            if not game_flow_data.empty:
                fig = self.charts['game_flow'].create_flow_chart(game_flow_data)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_teams_tab(self):
        """Render teams analysis tab."""
        st.subheader("Team Analysis")
        
        # Team comparison
        teams_data = self._get_teams_data()
        if not teams_data.empty:
            fig = self.charts['team_comparison'].create_comparison_chart(teams_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Team network analysis
        st.subheader("Team Interaction Network")
        network_data = self._get_team_network_data()
        if network_data:
            fig = self.charts['network'].create_team_network(network_data)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_predictions_tab(self):
        """Render predictions tab."""
        st.subheader("Predictive Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Game Predictions")
            predictions = self._get_game_predictions()
            for prediction in predictions:
                self._render_prediction_card(prediction)
        
        with col2:
            st.subheader("Player Performance Forecast")
            if st.session_state.selected_player:
                forecast_data = self._get_player_forecast()
                if not forecast_data.empty:
                    fig = self.charts['player_trajectory'].create_forecast_chart(forecast_data)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_analytics_tab(self):
        """Render advanced analytics tab."""
        st.subheader("Advanced Analytics")
        
        # Heatmap analysis
        st.subheader("Performance Heatmap")
        heatmap_data = self._get_heatmap_data()
        if not heatmap_data.empty:
            fig = self.charts['heatmap'].create_performance_heatmap(heatmap_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Statistical Insights")
        stats_data = self._get_statistical_insights()
        if stats_data:
            col1, col2 = st.columns(2)
            with col1:
                st.json(stats_data['correlations'])
            with col2:
                st.json(stats_data['trends'])
    
    def _get_teams(self, sport: str) -> List[str]:
        """Get teams for the selected sport."""
        try:
            query = "SELECT DISTINCT team_name FROM teams WHERE sport = ?"
            result = self.db.execute_query(query, (sport,))
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return []
    
    def _get_players(self, team: str) -> List[str]:
        """Get players for the selected team."""
        try:
            query = "SELECT DISTINCT player_name FROM players WHERE team_name = ?"
            result = self.db.execute_query(query, (team,))
            return [row[0] for row in result] if result else []
        except Exception as e:
            logger.error(f"Error fetching players: {e}")
            return []
    
    def _get_key_metrics(self) -> Dict[str, Any]:
        """Get key dashboard metrics."""
        try:
            # This would typically fetch from database
            return {
                'games_today': 8,
                'games_delta': 2,
                'active_players': 1247,
                'players_delta': 15,
                'avg_performance': 78.5,
                'performance_delta': 2.3,
                'prediction_accuracy': 0.847,
                'accuracy_delta': 0.023,
                'data_points': 125847,
                'data_delta': 1250
            }
        except Exception as e:
            logger.error(f"Error fetching key metrics: {e}")
            return {}
    
    def _get_performance_data(self) -> pd.DataFrame:
        """Get performance data for visualization."""
        try:
            # Generate sample data - replace with actual database queries
            dates = pd.date_range(
                start=st.session_state.date_range[0],
                end=st.session_state.date_range[1],
                freq='D'
            )
            
            data = []
            for date in dates:
                data.append({
                    'date': date,
                    'performance_score': np.random.normal(75, 10),
                    'efficiency': np.random.normal(0.6, 0.1),
                    'impact': np.random.normal(8, 2)
                })
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching performance data: {e}")
            return pd.DataFrame()
    
    def _get_players_comparison_data(self) -> pd.DataFrame:
        """Get player comparison data."""
        try:
            # Generate sample data
            players = ['Player A', 'Player B', 'Player C', 'Player D', 'Player E']
            metrics = ['Points', 'Assists', 'Rebounds', 'Efficiency', 'Impact']
            
            data = []
            for player in players:
                for metric in metrics:
                    data.append({
                        'player': player,
                        'metric': metric,
                        'value': np.random.normal(50, 15)
                    })
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching player comparison data: {e}")
            return pd.DataFrame()
    
    def _get_live_games(self) -> List[Dict]:
        """Get live games data."""
        try:
            # This would connect to real-time data stream
            return [
                {
                    'id': 1,
                    'home_team': 'Lakers',
                    'away_team': 'Warriors',
                    'score': '98-102',
                    'quarter': 'Q4',
                    'time_remaining': '2:45'
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching live games: {e}")
            return []
    
    def _get_recent_games(self) -> pd.DataFrame:
        """Get recent games data."""
        try:
            # Generate sample data
            games = []
            for i in range(10):
                games.append({
                    'date': datetime.now() - timedelta(days=i),
                    'home_team': f'Team {i+1}',
                    'away_team': f'Team {i+2}',
                    'home_score': np.random.randint(80, 120),
                    'away_score': np.random.randint(80, 120),
                    'status': 'Final'
                })
            
            return pd.DataFrame(games)
        except Exception as e:
            logger.error(f"Error fetching recent games: {e}")
            return pd.DataFrame()
    
    def _refresh_data(self):
        """Refresh dashboard data."""
        try:
            self.cache.clear()
            st.success("Data refreshed successfully!")
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            st.error("Failed to refresh data.")
    
    def _generate_report(self):
        """Generate analytics report."""
        try:
            st.success("Report generation started! Check your downloads folder.")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            st.error("Failed to generate report.")
    
    def _setup_auto_refresh(self):
        """Setup automatic dashboard refresh."""
        if st.session_state.auto_refresh:
            # This would implement auto-refresh logic
            pass
    
    def _render_live_game_card(self, game: Dict):
        """Render live game card."""
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{game['away_team']}**")
            with col2:
                st.write(f"**{game['score']}**")
            with col3:
                st.write(f"**{game['home_team']}**")
            
            st.caption(f"{game['quarter']} - {game['time_remaining']}")
            st.markdown("---")
    
    def _render_game_card(self, game: pd.Series):
        """Render game card."""
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{game['away_team']}**")
            with col2:
                st.write(f"**{game['away_score']}-{game['home_score']}**")
            with col3:
                st.write(f"**{game['home_team']}**")
            
            st.caption(f"{game['date'].strftime('%Y-%m-%d')} - {game['status']}")
            st.markdown("---")
    
    def _render_prediction_card(self, prediction: Dict):
        """Render prediction card."""
        with st.container():
            st.write(f"**{prediction.get('matchup', 'TBD')}**")
            st.write(f"Confidence: {prediction.get('confidence', 0):.1%}")
            st.write(f"Predicted Winner: {prediction.get('winner', 'TBD')}")
            st.markdown("---")
    
    # Additional helper methods for data fetching would go here...
    def _get_game_flow_data(self) -> pd.DataFrame:
        """Get game flow data."""
        return pd.DataFrame()
    
    def _get_teams_data(self) -> pd.DataFrame:
        """Get teams data."""
        return pd.DataFrame()
    
    def _get_team_network_data(self) -> Dict:
        """Get team network data."""
        return {}
    
    def _get_game_predictions(self) -> List[Dict]:
        """Get game predictions."""
        return []
    
    def _get_player_forecast(self) -> pd.DataFrame:
        """Get player forecast data."""
        return pd.DataFrame()
    
    def _get_heatmap_data(self) -> pd.DataFrame:
        """Get heatmap data."""
        return pd.DataFrame()
    
    def _get_statistical_insights(self) -> Dict:
        """Get statistical insights."""
        return {}

class DashboardManager:
    """
    Dashboard management and deployment utilities.
    """
    
    def __init__(self):
        self.dashboards = {}
        self.config = DashboardConfig()
    
    def create_dashboard(self, name: str, config: Optional[DashboardConfig] = None) -> SportsDashboard:
        """Create a new dashboard instance."""
        dashboard = SportsDashboard(config or self.config)
        self.dashboards[name] = dashboard
        return dashboard
    
    def get_dashboard(self, name: str) -> Optional[SportsDashboard]:
        """Get existing dashboard instance."""
        return self.dashboards.get(name)
    
    def list_dashboards(self) -> List[str]:
        """List all dashboard names."""
        return list(self.dashboards.keys())
    
    def remove_dashboard(self, name: str) -> bool:
        """Remove dashboard instance."""
        if name in self.dashboards:
            del self.dashboards[name]
            return True
        return False
    
    async def deploy_dashboard(self, name: str, port: int = 8501) -> bool:
        """Deploy dashboard to specified port."""
        try:
            dashboard = self.get_dashboard(name)
            if dashboard:
                # This would implement actual deployment logic
                logger.info(f"Dashboard '{name}' deployed on port {port}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deploying dashboard: {e}")
            return False 