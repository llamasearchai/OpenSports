"""
Advanced Performance Analysis System

Comprehensive performance analytics with statistical modeling,
trend analysis, and performance optimization for sports data.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    player_id: str
    metric_name: str
    value: float
    percentile: float
    z_score: float
    trend: str  # 'improving', 'declining', 'stable'
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'player_id': self.player_id,
            'metric_name': self.metric_name,
            'value': self.value,
            'percentile': self.percentile,
            'z_score': self.z_score,
            'trend': self.trend,
            'confidence_interval': self.confidence_interval,
            'sample_size': self.sample_size,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    subject_id: str
    subject_type: str  # 'player' or 'team'
    sport: str
    metrics: List[PerformanceMetrics]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    comparison_data: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'subject_id': self.subject_id,
            'subject_type': self.subject_type,
            'sport': self.sport,
            'metrics': [m.to_dict() for m in self.metrics],
            'overall_score': self.overall_score,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'recommendations': self.recommendations,
            'comparison_data': self.comparison_data,
            'visualizations': self.visualizations,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceAnalyzer:
    """Advanced performance analysis system"""
    
    def __init__(self, sport: str):
        self.sport = sport.lower()
        self.scaler = StandardScaler()
        self.models = {}
        self.metric_definitions = self._get_sport_metrics()
        
    def _get_sport_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get sport-specific metric definitions"""
        metrics = {
            'nba': {
                'points_per_game': {'weight': 0.25, 'higher_better': True},
                'assists_per_game': {'weight': 0.15, 'higher_better': True},
                'rebounds_per_game': {'weight': 0.15, 'higher_better': True},
                'field_goal_percentage': {'weight': 0.20, 'higher_better': True},
                'three_point_percentage': {'weight': 0.10, 'higher_better': True},
                'free_throw_percentage': {'weight': 0.05, 'higher_better': True},
                'steals_per_game': {'weight': 0.05, 'higher_better': True},
                'blocks_per_game': {'weight': 0.05, 'higher_better': True}
            },
            'nfl': {
                'passing_yards': {'weight': 0.20, 'higher_better': True},
                'rushing_yards': {'weight': 0.20, 'higher_better': True},
                'touchdowns': {'weight': 0.25, 'higher_better': True},
                'completion_percentage': {'weight': 0.15, 'higher_better': True},
                'yards_per_attempt': {'weight': 0.10, 'higher_better': True},
                'interceptions': {'weight': 0.10, 'higher_better': False}
            },
            'soccer': {
                'goals': {'weight': 0.30, 'higher_better': True},
                'assists': {'weight': 0.20, 'higher_better': True},
                'pass_accuracy': {'weight': 0.15, 'higher_better': True},
                'shots_on_target': {'weight': 0.15, 'higher_better': True},
                'tackles_won': {'weight': 0.10, 'higher_better': True},
                'distance_covered': {'weight': 0.10, 'higher_better': True}
            },
            'formula1': {
                'points': {'weight': 0.40, 'higher_better': True},
                'podium_finishes': {'weight': 0.25, 'higher_better': True},
                'qualifying_position': {'weight': 0.15, 'higher_better': False},
                'fastest_laps': {'weight': 0.10, 'higher_better': True},
                'dnf_rate': {'weight': 0.10, 'higher_better': False}
            }
        }
        
        return metrics.get(self.sport, {})
    
    async def analyze_player_performance(self, player_data: pd.DataFrame, 
                                       player_id: str) -> PerformanceReport:
        """Analyze individual player performance"""
        # Filter data for specific player
        player_stats = player_data[player_data['player_id'] == player_id].copy()
        
        if len(player_stats) == 0:
            raise ValueError(f"No data found for player {player_id}")
        
        # Calculate performance metrics
        metrics = await self._calculate_performance_metrics(player_stats, player_id)
        
        # Calculate overall performance score
        overall_score = self._calculate_overall_score(metrics)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, strengths, weaknesses)
        
        # Create comparison data
        comparison_data = await self._create_comparison_data(player_data, player_stats)
        
        return PerformanceReport(
            subject_id=player_id,
            subject_type='player',
            sport=self.sport,
            metrics=metrics,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            comparison_data=comparison_data
        )
    
    async def analyze_team_performance(self, team_data: pd.DataFrame, 
                                     team_id: str) -> PerformanceReport:
        """Analyze team performance"""
        # Filter data for specific team
        team_stats = team_data[team_data['team_id'] == team_id].copy()
        
        if len(team_stats) == 0:
            raise ValueError(f"No data found for team {team_id}")
        
        # Calculate team metrics
        metrics = await self._calculate_team_metrics(team_stats, team_id)
        
        # Calculate overall team score
        overall_score = self._calculate_overall_score(metrics)
        
        # Identify team strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(metrics)
        
        # Generate team recommendations
        recommendations = self._generate_team_recommendations(metrics, strengths, weaknesses)
        
        # Create team comparison data
        comparison_data = await self._create_team_comparison_data(team_data, team_stats)
        
        return PerformanceReport(
            subject_id=team_id,
            subject_type='team',
            sport=self.sport,
            metrics=metrics,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            comparison_data=comparison_data
        )
    
    async def _calculate_performance_metrics(self, player_stats: pd.DataFrame, 
                                           player_id: str) -> List[PerformanceMetrics]:
        """Calculate performance metrics for a player"""
        metrics = []
        
        for metric_name, config in self.metric_definitions.items():
            if metric_name in player_stats.columns:
                values = player_stats[metric_name].dropna()
                
                if len(values) > 0:
                    # Calculate basic statistics
                    mean_value = values.mean()
                    std_value = values.std()
                    
                    # Calculate percentile (compared to league)
                    percentile = self._calculate_percentile(mean_value, metric_name)
                    
                    # Calculate z-score
                    z_score = (mean_value - values.mean()) / (std_value + 1e-8)
                    
                    # Determine trend
                    trend = self._calculate_trend(values)
                    
                    # Calculate confidence interval
                    confidence_interval = self._calculate_confidence_interval(values)
                    
                    metric = PerformanceMetrics(
                        player_id=player_id,
                        metric_name=metric_name,
                        value=mean_value,
                        percentile=percentile,
                        z_score=z_score,
                        trend=trend,
                        confidence_interval=confidence_interval,
                        sample_size=len(values)
                    )
                    
                    metrics.append(metric)
        
        return metrics
    
    async def _calculate_team_metrics(self, team_stats: pd.DataFrame, 
                                    team_id: str) -> List[PerformanceMetrics]:
        """Calculate performance metrics for a team"""
        metrics = []
        
        # Aggregate team statistics
        team_aggregated = team_stats.groupby('game_id').agg({
            metric: 'mean' for metric in self.metric_definitions.keys()
            if metric in team_stats.columns
        }).reset_index()
        
        for metric_name, config in self.metric_definitions.items():
            if metric_name in team_aggregated.columns:
                values = team_aggregated[metric_name].dropna()
                
                if len(values) > 0:
                    mean_value = values.mean()
                    std_value = values.std()
                    
                    percentile = self._calculate_percentile(mean_value, metric_name)
                    z_score = (mean_value - values.mean()) / (std_value + 1e-8)
                    trend = self._calculate_trend(values)
                    confidence_interval = self._calculate_confidence_interval(values)
                    
                    metric = PerformanceMetrics(
                        player_id=team_id,
                        metric_name=metric_name,
                        value=mean_value,
                        percentile=percentile,
                        z_score=z_score,
                        trend=trend,
                        confidence_interval=confidence_interval,
                        sample_size=len(values)
                    )
                    
                    metrics.append(metric)
        
        return metrics
    
    def _calculate_percentile(self, value: float, metric_name: str) -> float:
        """Calculate percentile ranking for a metric value"""
        # This would typically use league-wide data for comparison
        # For now, using a simplified approach
        return min(max(stats.norm.cdf(value, loc=0, scale=1) * 100, 0), 100)
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate performance trend"""
        if len(values) < 3:
            return 'stable'
        
        # Use linear regression to determine trend
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                return 'improving'
            else:
                return 'declining'
        else:
            return 'stable'
    
    def _calculate_confidence_interval(self, values: pd.Series, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for metric"""
        if len(values) < 2:
            return (values.iloc[0], values.iloc[0])
        
        mean = values.mean()
        sem = stats.sem(values)
        h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        
        return (mean - h, mean + h)
    
    def _calculate_overall_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score"""
        if not metrics:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            if metric.metric_name in self.metric_definitions:
                weight = self.metric_definitions[metric.metric_name]['weight']
                
                # Normalize percentile to 0-1 scale
                normalized_score = metric.percentile / 100.0
                
                weighted_score += normalized_score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_strengths_weaknesses(self, metrics: List[PerformanceMetrics]) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on metrics"""
        strengths = []
        weaknesses = []
        
        for metric in metrics:
            if metric.percentile >= 75:
                strengths.append(f"{metric.metric_name}: {metric.percentile:.1f}th percentile")
            elif metric.percentile <= 25:
                weaknesses.append(f"{metric.metric_name}: {metric.percentile:.1f}th percentile")
        
        return strengths, weaknesses
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics], 
                                strengths: List[str], weaknesses: List[str]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze weaknesses and suggest improvements
        for weakness in weaknesses:
            metric_name = weakness.split(':')[0]
            
            if 'shooting' in metric_name.lower() or 'percentage' in metric_name.lower():
                recommendations.append(f"Focus on {metric_name} through targeted practice sessions")
            elif 'fitness' in metric_name.lower() or 'endurance' in metric_name.lower():
                recommendations.append(f"Improve {metric_name} through conditioning programs")
            else:
                recommendations.append(f"Work on improving {metric_name} through specialized training")
        
        # Leverage strengths
        for strength in strengths:
            metric_name = strength.split(':')[0]
            recommendations.append(f"Continue to leverage strength in {metric_name}")
        
        return recommendations
    
    def _generate_team_recommendations(self, metrics: List[PerformanceMetrics], 
                                     strengths: List[str], weaknesses: List[str]) -> List[str]:
        """Generate team-specific recommendations"""
        recommendations = []
        
        for weakness in weaknesses:
            metric_name = weakness.split(':')[0]
            
            if 'offense' in metric_name.lower():
                recommendations.append("Focus on offensive strategy and player development")
            elif 'defense' in metric_name.lower():
                recommendations.append("Strengthen defensive schemes and positioning")
            elif 'teamwork' in metric_name.lower():
                recommendations.append("Improve team chemistry and communication")
            else:
                recommendations.append(f"Address team weakness in {metric_name}")
        
        return recommendations
    
    async def _create_comparison_data(self, all_data: pd.DataFrame, 
                                    player_data: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison data for player vs league"""
        comparison = {}
        
        for metric_name in self.metric_definitions.keys():
            if metric_name in all_data.columns:
                league_stats = all_data[metric_name].dropna()
                player_stats = player_data[metric_name].dropna()
                
                if len(league_stats) > 0 and len(player_stats) > 0:
                    comparison[metric_name] = {
                        'player_avg': player_stats.mean(),
                        'league_avg': league_stats.mean(),
                        'player_std': player_stats.std(),
                        'league_std': league_stats.std(),
                        'percentile': stats.percentileofscore(league_stats, player_stats.mean())
                    }
        
        return comparison
    
    async def _create_team_comparison_data(self, all_data: pd.DataFrame, 
                                         team_data: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison data for team vs league"""
        # Similar to player comparison but aggregated at team level
        return await self._create_comparison_data(all_data, team_data)


class PlayerPerformanceModel:
    """Advanced player performance modeling"""
    
    def __init__(self, sport: str):
        self.sport = sport
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    async def train_performance_model(self, training_data: pd.DataFrame, 
                                    target_metric: str) -> Dict[str, Any]:
        """Train performance prediction model"""
        # Prepare features
        features = self._prepare_features(training_data)
        target = training_data[target_metric].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    features.columns, model.feature_importances_
                ))
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.models[target_metric] = results[best_model_name]['model']
        
        return results
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling"""
        # Select relevant features based on sport
        feature_columns = []
        
        if self.sport == 'nba':
            feature_columns = [
                'minutes_played', 'field_goals_attempted', 'three_pointers_attempted',
                'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds',
                'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls'
            ]
        elif self.sport == 'nfl':
            feature_columns = [
                'attempts', 'completions', 'yards', 'touchdowns', 'interceptions',
                'rushing_attempts', 'rushing_yards', 'fumbles'
            ]
        elif self.sport == 'soccer':
            feature_columns = [
                'shots', 'shots_on_target', 'passes', 'pass_accuracy',
                'tackles', 'interceptions', 'clearances', 'crosses'
            ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features:
            # Use all numeric columns as fallback
            available_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        return data[available_features].fillna(0)
    
    async def predict_performance(self, player_data: pd.DataFrame, 
                                target_metric: str) -> np.ndarray:
        """Predict player performance"""
        if target_metric not in self.models:
            raise ValueError(f"No trained model for {target_metric}")
        
        features = self._prepare_features(player_data)
        features_scaled = self.scaler.transform(features)
        
        return self.models[target_metric].predict(features_scaled)
    
    def get_feature_importance(self, target_metric: str) -> Dict[str, float]:
        """Get feature importance for a specific metric"""
        return self.feature_importance.get(target_metric, {})


class TeamPerformanceModel:
    """Advanced team performance modeling"""
    
    def __init__(self, sport: str):
        self.sport = sport
        self.models = {}
        self.team_metrics = {}
        
    async def analyze_team_chemistry(self, team_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze team chemistry and cohesion"""
        chemistry_metrics = {}
        
        # Calculate passing network metrics (for sports with passing)
        if 'passes' in team_data.columns and 'assists' in team_data.columns:
            chemistry_metrics['passing_efficiency'] = (
                team_data['assists'].sum() / team_data['passes'].sum()
            )
        
        # Calculate performance variance (lower is better for consistency)
        numeric_cols = team_data.select_dtypes(include=[np.number]).columns
        chemistry_metrics['performance_consistency'] = 1 / (
            team_data[numeric_cols].std().mean() + 1e-8
        )
        
        # Calculate win probability based on team performance
        if 'wins' in team_data.columns and 'games_played' in team_data.columns:
            chemistry_metrics['win_rate'] = (
                team_data['wins'].sum() / team_data['games_played'].sum()
            )
        
        return chemistry_metrics
    
    async def optimize_lineup(self, player_data: pd.DataFrame, 
                            positions: List[str]) -> Dict[str, Any]:
        """Optimize team lineup based on player performance"""
        optimization_results = {}
        
        # Group players by position
        position_groups = {}
        for position in positions:
            position_players = player_data[
                player_data['position'] == position
            ] if 'position' in player_data.columns else player_data
            
            position_groups[position] = position_players
        
        # Calculate optimal lineup based on performance metrics
        optimal_lineup = {}
        for position, players in position_groups.items():
            if len(players) > 0:
                # Select best player for position based on overall performance
                best_player = players.loc[players['overall_score'].idxmax()] \
                    if 'overall_score' in players.columns else players.iloc[0]
                optimal_lineup[position] = best_player.to_dict()
        
        optimization_results['optimal_lineup'] = optimal_lineup
        optimization_results['expected_performance'] = self._calculate_lineup_performance(optimal_lineup)
        
        return optimization_results
    
    def _calculate_lineup_performance(self, lineup: Dict[str, Any]) -> float:
        """Calculate expected performance for a lineup"""
        if not lineup:
            return 0.0
        
        total_score = sum(
            player.get('overall_score', 0) for player in lineup.values()
        )
        
        return total_score / len(lineup) 