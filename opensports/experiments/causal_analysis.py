"""
Causal Analysis for Sports Analytics

Advanced causal inference using DoWhy framework to understand true causal relationships
in sports data, including coaching impacts, player trades, and strategy changes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import get_database
from opensports.core.cache import cache_async_result

logger = get_logger(__name__)


class CausalAnalyzer:
    """
    Advanced causal inference system for sports analytics.
    
    Features:
    - DoWhy integration for causal modeling
    - Multiple identification methods
    - Robustness testing with refutation methods
    - Treatment effect estimation
    - Confounding variable detection
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.db = get_database()
        
    async def estimate_coaching_impact(
        self,
        team: str,
        coach_change_date: str,
        outcome_metric: str = "win_percentage",
        lookback_days: int = 365,
        lookahead_days: int = 365
    ) -> Dict[str, Any]:
        """
        Estimate the causal impact of a coaching change on team performance.
        
        Args:
            team: Team identifier
            coach_change_date: Date of coaching change
            outcome_metric: Performance metric to analyze
            lookback_days: Days before change to include
            lookahead_days: Days after change to include
            
        Returns:
            Causal analysis results with treatment effects
        """
        logger.info(f"Analyzing coaching impact for {team} on {coach_change_date}")
        
        # Get team performance data
        team_data = await self._get_team_performance_data(
            team, coach_change_date, lookback_days, lookahead_days
        )
        
        if team_data.empty:
            raise ValueError(f"No data found for team {team}")
        
        # Prepare causal model
        causal_graph = self._create_coaching_causal_graph()
        
        # Create treatment variable (before/after coaching change)
        change_date = pd.to_datetime(coach_change_date)
        team_data['treatment'] = (pd.to_datetime(team_data['game_date']) >= change_date).astype(int)
        
        # Create causal model
        model = CausalModel(
            data=team_data,
            treatment='treatment',
            outcome=outcome_metric,
            graph=causal_graph
        )
        
        # Identify causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate causal effect using multiple methods
        results = {}
        
        # Method 1: Linear regression
        try:
            estimate_lr = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            results['linear_regression'] = {
                'effect': float(estimate_lr.value),
                'confidence_interval': [float(x) for x in estimate_lr.get_confidence_intervals()],
                'p_value': float(estimate_lr.get_statistical_significance())
            }
        except Exception as e:
            logger.warning(f"Linear regression estimation failed: {e}")
        
        # Method 2: Propensity score matching
        try:
            estimate_psm = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching"
            )
            results['propensity_score_matching'] = {
                'effect': float(estimate_psm.value),
                'confidence_interval': [float(x) for x in estimate_psm.get_confidence_intervals()],
                'p_value': float(estimate_psm.get_statistical_significance())
            }
        except Exception as e:
            logger.warning(f"Propensity score matching failed: {e}")
        
        # Refutation tests
        refutation_results = await self._run_refutation_tests(model, identified_estimand)
        
        # Calculate additional metrics
        pre_treatment_mean = team_data[team_data['treatment'] == 0][outcome_metric].mean()
        post_treatment_mean = team_data[team_data['treatment'] == 1][outcome_metric].mean()
        
        analysis_result = {
            'team': team,
            'coach_change_date': coach_change_date,
            'outcome_metric': outcome_metric,
            'treatment_effects': results,
            'refutation_tests': refutation_results,
            'descriptive_stats': {
                'pre_treatment_mean': float(pre_treatment_mean),
                'post_treatment_mean': float(post_treatment_mean),
                'naive_difference': float(post_treatment_mean - pre_treatment_mean),
                'sample_size_pre': int((team_data['treatment'] == 0).sum()),
                'sample_size_post': int((team_data['treatment'] == 1).sum())
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Coaching impact analysis complete for {team}")
        return analysis_result
    
    async def estimate_player_trade_impact(
        self,
        player: str,
        from_team: str,
        to_team: str,
        trade_date: str,
        outcome_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate the causal impact of a player trade on team and player performance.
        
        Args:
            player: Player identifier
            from_team: Team player was traded from
            to_team: Team player was traded to
            trade_date: Date of trade
            outcome_metrics: Performance metrics to analyze
            
        Returns:
            Causal analysis results for the trade impact
        """
        outcome_metrics = outcome_metrics or ['win_percentage', 'points_per_game', 'defensive_rating']
        
        logger.info(f"Analyzing trade impact: {player} from {from_team} to {to_team}")
        
        results = {}
        
        # Analyze impact on both teams
        for team, team_type in [(from_team, 'losing_team'), (to_team, 'gaining_team')]:
            team_results = {}
            
            for metric in outcome_metrics:
                try:
                    # Get team data around trade date
                    team_data = await self._get_team_performance_data(
                        team, trade_date, lookback_days=180, lookahead_days=180
                    )
                    
                    if not team_data.empty:
                        # Create treatment variable
                        trade_dt = pd.to_datetime(trade_date)
                        team_data['treatment'] = (pd.to_datetime(team_data['game_date']) >= trade_dt).astype(int)
                        
                        # Adjust treatment for losing vs gaining team
                        if team_type == 'losing_team':
                            # For losing team, treatment is losing the player
                            team_data['treatment'] = 1 - team_data['treatment']
                        
                        # Create causal model
                        causal_graph = self._create_trade_causal_graph()
                        model = CausalModel(
                            data=team_data,
                            treatment='treatment',
                            outcome=metric,
                            graph=causal_graph
                        )
                        
                        # Estimate effect
                        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                        estimate = model.estimate_effect(
                            identified_estimand,
                            method_name="backdoor.linear_regression"
                        )
                        
                        team_results[metric] = {
                            'effect': float(estimate.value),
                            'confidence_interval': [float(x) for x in estimate.get_confidence_intervals()],
                            'p_value': float(estimate.get_statistical_significance())
                        }
                        
                except Exception as e:
                    logger.warning(f"Trade analysis failed for {team} - {metric}: {e}")
                    team_results[metric] = {'error': str(e)}
            
            results[team_type] = {
                'team': team,
                'metrics': team_results
            }
        
        # Analyze player performance impact
        player_results = await self._analyze_player_trade_performance(player, trade_date)
        results['player_impact'] = player_results
        
        return {
            'player': player,
            'from_team': from_team,
            'to_team': to_team,
            'trade_date': trade_date,
            'causal_effects': results,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def estimate_strategy_change_impact(
        self,
        team: str,
        strategy_change_date: str,
        strategy_type: str,
        outcome_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate the causal impact of a strategy change on team performance.
        
        Args:
            team: Team identifier
            strategy_change_date: Date of strategy change
            strategy_type: Type of strategy change (offensive, defensive, etc.)
            outcome_metrics: Performance metrics to analyze
            
        Returns:
            Causal analysis results for strategy change impact
        """
        outcome_metrics = outcome_metrics or ['offensive_rating', 'defensive_rating', 'pace']
        
        logger.info(f"Analyzing strategy change impact for {team}: {strategy_type}")
        
        # Get team data
        team_data = await self._get_team_performance_data(
            team, strategy_change_date, lookback_days=120, lookahead_days=120
        )
        
        if team_data.empty:
            raise ValueError(f"No data found for team {team}")
        
        results = {}
        
        for metric in outcome_metrics:
            try:
                # Create treatment variable
                change_date = pd.to_datetime(strategy_change_date)
                team_data['treatment'] = (pd.to_datetime(team_data['game_date']) >= change_date).astype(int)
                
                # Create causal model
                causal_graph = self._create_strategy_causal_graph()
                model = CausalModel(
                    data=team_data,
                    treatment='treatment',
                    outcome=metric,
                    graph=causal_graph
                )
                
                # Identify and estimate effect
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression"
                )
                
                # Run refutation tests
                refutation_results = await self._run_refutation_tests(model, identified_estimand)
                
                results[metric] = {
                    'effect': float(estimate.value),
                    'confidence_interval': [float(x) for x in estimate.get_confidence_intervals()],
                    'p_value': float(estimate.get_statistical_significance()),
                    'refutation_tests': refutation_results
                }
                
            except Exception as e:
                logger.warning(f"Strategy analysis failed for {metric}: {e}")
                results[metric] = {'error': str(e)}
        
        return {
            'team': team,
            'strategy_change_date': strategy_change_date,
            'strategy_type': strategy_type,
            'causal_effects': results,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_coaching_causal_graph(self) -> str:
        """Create causal graph for coaching change analysis."""
        return """
        digraph {
            treatment -> outcome;
            team_quality -> outcome;
            team_quality -> treatment;
            opponent_strength -> outcome;
            home_advantage -> outcome;
            injuries -> outcome;
            injuries -> treatment;
            season_progress -> outcome;
            recent_performance -> treatment;
            recent_performance -> outcome;
        }
        """
    
    def _create_trade_causal_graph(self) -> str:
        """Create causal graph for player trade analysis."""
        return """
        digraph {
            treatment -> outcome;
            team_quality -> outcome;
            team_quality -> treatment;
            player_quality -> outcome;
            player_quality -> treatment;
            opponent_strength -> outcome;
            home_advantage -> outcome;
            injuries -> outcome;
            chemistry -> outcome;
            chemistry -> treatment;
        }
        """
    
    def _create_strategy_causal_graph(self) -> str:
        """Create causal graph for strategy change analysis."""
        return """
        digraph {
            treatment -> outcome;
            team_talent -> outcome;
            team_talent -> treatment;
            opponent_strength -> outcome;
            home_advantage -> outcome;
            fatigue -> outcome;
            matchup_advantage -> outcome;
            recent_performance -> treatment;
        }
        """
    
    async def _run_refutation_tests(
        self,
        model: CausalModel,
        identified_estimand: Any
    ) -> Dict[str, Any]:
        """Run refutation tests to validate causal estimates."""
        refutation_results = {}
        
        try:
            # Random common cause
            refute_random = model.refute_estimate(
                identified_estimand,
                method_name="random_common_cause"
            )
            refutation_results['random_common_cause'] = {
                'new_effect': float(refute_random.new_effect),
                'p_value': float(refute_random.refutation_result.get('p_value', 0))
            }
        except Exception as e:
            logger.warning(f"Random common cause refutation failed: {e}")
        
        try:
            # Placebo treatment
            refute_placebo = model.refute_estimate(
                identified_estimand,
                method_name="placebo_treatment_refuter"
            )
            refutation_results['placebo_treatment'] = {
                'new_effect': float(refute_placebo.new_effect),
                'p_value': float(refute_placebo.refutation_result.get('p_value', 0))
            }
        except Exception as e:
            logger.warning(f"Placebo treatment refutation failed: {e}")
        
        try:
            # Data subset validation
            refute_subset = model.refute_estimate(
                identified_estimand,
                method_name="data_subset_refuter",
                subset_fraction=0.8
            )
            refutation_results['data_subset'] = {
                'new_effect': float(refute_subset.new_effect),
                'p_value': float(refute_subset.refutation_result.get('p_value', 0))
            }
        except Exception as e:
            logger.warning(f"Data subset refutation failed: {e}")
        
        return refutation_results
    
    async def _get_team_performance_data(
        self,
        team: str,
        reference_date: str,
        lookback_days: int,
        lookahead_days: int
    ) -> pd.DataFrame:
        """Get team performance data around a reference date."""
        ref_date = pd.to_datetime(reference_date)
        start_date = ref_date - timedelta(days=lookback_days)
        end_date = ref_date + timedelta(days=lookahead_days)
        
        # This would query the actual database
        # For now, return a mock DataFrame
        dates = pd.date_range(start_date, end_date, freq='3D')  # Games every 3 days
        
        # Generate mock data
        np.random.seed(42)
        n_games = len(dates)
        
        data = {
            'game_date': dates,
            'team': [team] * n_games,
            'win_percentage': np.random.normal(0.5, 0.1, n_games).clip(0, 1),
            'points_per_game': np.random.normal(110, 10, n_games),
            'offensive_rating': np.random.normal(110, 8, n_games),
            'defensive_rating': np.random.normal(108, 8, n_games),
            'pace': np.random.normal(100, 5, n_games),
            'opponent_strength': np.random.normal(0.5, 0.15, n_games),
            'home_advantage': np.random.choice([0, 1], n_games),
            'injuries': np.random.poisson(1, n_games),
            'recent_performance': np.random.normal(0.5, 0.1, n_games)
        }
        
        return pd.DataFrame(data)
    
    async def _analyze_player_trade_performance(
        self,
        player: str,
        trade_date: str
    ) -> Dict[str, Any]:
        """Analyze player performance before and after trade."""
        # This would query actual player performance data
        # For now, return mock analysis
        
        return {
            'pre_trade_performance': {
                'points_per_game': 18.5,
                'efficiency': 0.55,
                'usage_rate': 0.28
            },
            'post_trade_performance': {
                'points_per_game': 21.2,
                'efficiency': 0.58,
                'usage_rate': 0.32
            },
            'performance_change': {
                'points_per_game': 2.7,
                'efficiency': 0.03,
                'usage_rate': 0.04
            },
            'statistical_significance': {
                'points_per_game': 0.02,
                'efficiency': 0.15,
                'usage_rate': 0.08
            }
        }
    
    async def detect_confounders(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        potential_confounders: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect potential confounding variables in the data.
        
        Args:
            data: Dataset for analysis
            treatment: Treatment variable name
            outcome: Outcome variable name
            potential_confounders: List of potential confounding variables
            
        Returns:
            Analysis of potential confounders
        """
        if potential_confounders is None:
            # Use all other numeric columns as potential confounders
            potential_confounders = [
                col for col in data.select_dtypes(include=[np.number]).columns
                if col not in [treatment, outcome]
            ]
        
        confounder_analysis = {}
        
        for confounder in potential_confounders:
            try:
                # Check correlation with treatment
                treatment_corr = data[treatment].corr(data[confounder])
                
                # Check correlation with outcome
                outcome_corr = data[outcome].corr(data[confounder])
                
                # A variable is a potential confounder if it's correlated with both
                # treatment and outcome
                is_potential_confounder = (
                    abs(treatment_corr) > 0.1 and abs(outcome_corr) > 0.1
                )
                
                confounder_analysis[confounder] = {
                    'treatment_correlation': float(treatment_corr),
                    'outcome_correlation': float(outcome_corr),
                    'is_potential_confounder': is_potential_confounder,
                    'confounder_strength': float(abs(treatment_corr) * abs(outcome_corr))
                }
                
            except Exception as e:
                logger.warning(f"Confounder analysis failed for {confounder}: {e}")
        
        # Sort by confounder strength
        sorted_confounders = dict(
            sorted(
                confounder_analysis.items(),
                key=lambda x: x[1]['confounder_strength'],
                reverse=True
            )
        )
        
        return {
            'potential_confounders': sorted_confounders,
            'top_confounders': list(sorted_confounders.keys())[:5],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    @cache_async_result(ttl=3600)
    async def get_causal_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all causal analyses performed."""
        return {
            'total_analyses': len(self.models),
            'analysis_types': list(self.models.keys()),
            'last_updated': datetime.now().isoformat()
        } 