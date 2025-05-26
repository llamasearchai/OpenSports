"""
Player Performance Prediction Model

Advanced machine learning model for predicting player performance across multiple sports.
Uses ensemble methods, feature engineering, and SHAP explainability.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import DatabaseManager
from opensports.core.cache import cache_result

logger = get_logger(__name__)


class PlayerPerformanceModel:
    """
    Advanced player performance prediction model using ensemble methods.
    
    Features:
    - Multi-sport support (NBA, NFL, Soccer, etc.)
    - Ensemble of XGBoost, LightGBM, and Random Forest
    - Advanced feature engineering
    - SHAP explainability
    - Real-time prediction updates
    """
    
    def __init__(self, sport: str = "nba"):
        self.sport = sport.lower()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_metrics = self._get_target_metrics()
        self.db = DatabaseManager()
        
    def _get_target_metrics(self) -> List[str]:
        """Get target metrics based on sport."""
        sport_metrics = {
            "nba": ["points", "rebounds", "assists", "steals", "blocks", "fg_pct", "three_pt_pct"],
            "nfl": ["passing_yards", "rushing_yards", "receiving_yards", "touchdowns", "interceptions"],
            "soccer": ["goals", "assists", "shots", "passes", "tackles", "saves"],
            "baseball": ["batting_avg", "home_runs", "rbi", "era", "strikeouts"],
            "hockey": ["goals", "assists", "shots", "saves", "penalty_minutes"]
        }
        return sport_metrics.get(self.sport, sport_metrics["nba"])
    
    async def prepare_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for player performance prediction.
        
        Args:
            player_data: Raw player statistics DataFrame
            
        Returns:
            Engineered features DataFrame
        """
        logger.info(f"Preparing features for {len(player_data)} player records")
        
        # Basic features
        features = player_data.copy()
        
        # Time-based features
        if 'game_date' in features.columns:
            features['game_date'] = pd.to_datetime(features['game_date'])
            features['day_of_week'] = features['game_date'].dt.dayofweek
            features['month'] = features['game_date'].dt.month
            features['days_since_last_game'] = features.groupby('player_id')['game_date'].diff().dt.days
            
        # Rolling averages (last 5, 10, 20 games)
        for window in [5, 10, 20]:
            for metric in self.target_metrics:
                if metric in features.columns:
                    features[f'{metric}_avg_{window}'] = (
                        features.groupby('player_id')[metric]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    
        # Trend features (performance direction)
        for metric in self.target_metrics:
            if metric in features.columns:
                features[f'{metric}_trend'] = (
                    features.groupby('player_id')[metric]
                    .pct_change(periods=5)
                    .fillna(0)
                )
                
        # Opponent strength features
        if 'opponent_team' in features.columns:
            opponent_stats = features.groupby('opponent_team')[self.target_metrics].mean()
            for metric in self.target_metrics:
                if metric in opponent_stats.columns:
                    features[f'opp_{metric}_avg'] = features['opponent_team'].map(
                        opponent_stats[metric]
                    )
                    
        # Home/Away performance
        if 'is_home' in features.columns:
            for metric in self.target_metrics:
                if metric in features.columns:
                    home_avg = features.groupby(['player_id', 'is_home'])[metric].transform('mean')
                    features[f'{metric}_home_avg'] = home_avg
                    
        # Rest days impact
        if 'days_rest' in features.columns:
            features['rest_category'] = pd.cut(
                features['days_rest'], 
                bins=[-1, 0, 1, 2, 7, float('inf')], 
                labels=['back_to_back', 'one_day', 'two_days', 'week', 'long_rest']
            )
            
        # Season progression
        if 'game_number' in features.columns:
            features['season_progress'] = features['game_number'] / features['game_number'].max()
            
        # Player age and experience
        if 'player_age' in features.columns:
            features['age_squared'] = features['player_age'] ** 2
            features['prime_years'] = ((features['player_age'] >= 25) & 
                                     (features['player_age'] <= 30)).astype(int)
                                     
        # Injury history impact
        if 'games_missed_last_30' in features.columns:
            features['injury_prone'] = (features['games_missed_last_30'] > 3).astype(int)
            
        # Team performance context
        if 'team_wins' in features.columns and 'team_losses' in features.columns:
            features['team_win_pct'] = features['team_wins'] / (features['team_wins'] + features['team_losses'])
            
        logger.info(f"Feature engineering complete. Shape: {features.shape}")
        return features
    
    async def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ensemble of models for each target metric.
        
        Args:
            training_data: Prepared training data with features and targets
            
        Returns:
            Training results and model performance metrics
        """
        logger.info("Starting model training for player performance prediction")
        
        # Prepare features
        features = await self.prepare_features(training_data)
        
        # Select feature columns (exclude targets and metadata)
        exclude_cols = self.target_metrics + ['player_id', 'game_id', 'game_date', 'player_name']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        X = features[self.feature_columns]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
                
        # Scale features
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X_scaled = self.scalers['scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['scaler'].transform(X)
            
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        results = {}
        
        # Train models for each target metric
        for metric in self.target_metrics:
            if metric not in features.columns:
                logger.warning(f"Target metric {metric} not found in data")
                continue
                
            logger.info(f"Training models for {metric}")
            
            y = features[metric].dropna()
            X_metric = X_scaled.loc[y.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_metric, y, test_size=0.2, random_state=42
            )
            
            # Initialize models
            models = {
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            metric_results = {}
            trained_models = {}
            
            # Train and evaluate each model
            for model_name, model in models.items():
                logger.info(f"Training {model_name} for {metric}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                metric_results[model_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                trained_models[model_name] = model
                
                logger.info(f"{model_name} - Test R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")
            
            # Create ensemble model (weighted average based on CV performance)
            weights = {}
            total_cv = sum(metric_results[name]['cv_mean'] for name in trained_models.keys())
            for name in trained_models.keys():
                weights[name] = metric_results[name]['cv_mean'] / total_cv
                
            # Store models and results
            self.models[metric] = {
                'models': trained_models,
                'weights': weights,
                'X_test': X_test,
                'y_test': y_test
            }
            
            results[metric] = metric_results
            
        logger.info("Model training completed successfully")
        return results
    
    async def predict_performance(
        self, 
        player_id: str, 
        upcoming_games: List[Dict], 
        use_ensemble: bool = True
    ) -> Dict[str, Any]:
        """
        Predict player performance for upcoming games.
        
        Args:
            player_id: Player identifier
            upcoming_games: List of upcoming game information
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Performance predictions with confidence intervals
        """
        logger.info(f"Predicting performance for player {player_id}")
        
        # Get player historical data
        player_data = await self._get_player_data(player_id)
        
        if player_data.empty:
            raise ValueError(f"No historical data found for player {player_id}")
            
        predictions = {}
        
        for game_info in upcoming_games:
            game_predictions = {}
            
            # Prepare features for this game
            game_features = await self._prepare_game_features(player_data, game_info)
            
            for metric in self.target_metrics:
                if metric not in self.models:
                    continue
                    
                model_info = self.models[metric]
                
                if use_ensemble:
                    # Ensemble prediction
                    ensemble_pred = 0
                    for model_name, model in model_info['models'].items():
                        pred = model.predict(game_features)[0]
                        weight = model_info['weights'][model_name]
                        ensemble_pred += pred * weight
                        
                    prediction = ensemble_pred
                else:
                    # Use best single model (highest CV score)
                    best_model_name = max(
                        model_info['models'].keys(),
                        key=lambda x: model_info['weights'][x]
                    )
                    best_model = model_info['models'][best_model_name]
                    prediction = best_model.predict(game_features)[0]
                
                # Calculate confidence interval using model uncertainty
                confidence_interval = await self._calculate_confidence_interval(
                    metric, game_features, prediction
                )
                
                game_predictions[metric] = {
                    'prediction': float(prediction),
                    'confidence_lower': float(confidence_interval[0]),
                    'confidence_upper': float(confidence_interval[1]),
                    'confidence_level': 0.95
                }
                
            predictions[game_info['game_id']] = {
                'game_info': game_info,
                'predictions': game_predictions,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        logger.info(f"Generated predictions for {len(predictions)} games")
        return predictions
    
    async def explain_prediction(
        self, 
        player_id: str, 
        game_info: Dict, 
        metric: str
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a specific prediction.
        
        Args:
            player_id: Player identifier
            game_info: Game information
            metric: Target metric to explain
            
        Returns:
            SHAP explanation values and feature importance
        """
        if metric not in self.models:
            raise ValueError(f"No trained model found for metric: {metric}")
            
        # Get player data and prepare features
        player_data = await self._get_player_data(player_id)
        game_features = await self._prepare_game_features(player_data, game_info)
        
        # Use the best model for explanation
        model_info = self.models[metric]
        best_model_name = max(
            model_info['models'].keys(),
            key=lambda x: model_info['weights'][x]
        )
        best_model = model_info['models'][best_model_name]
        
        # Generate SHAP explanations
        if hasattr(best_model, 'predict_proba'):
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.Explainer(best_model)
            
        shap_values = explainer.shap_values(game_features)
        
        # Format explanation
        feature_importance = dict(zip(self.feature_columns, shap_values[0]))
        
        return {
            'metric': metric,
            'prediction': float(best_model.predict(game_features)[0]),
            'base_value': float(explainer.expected_value),
            'feature_importance': {
                k: float(v) for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
            },
            'top_positive_factors': [
                {'feature': k, 'impact': float(v)} 
                for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            ],
            'top_negative_factors': [
                {'feature': k, 'impact': float(v)} 
                for k, v in sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1]
                )[:5]
            ]
        }
    
    async def _get_player_data(self, player_id: str) -> pd.DataFrame:
        """Get historical player data from database."""
        query = f"""
        SELECT * FROM player_stats 
        WHERE player_id = '{player_id}' 
        ORDER BY game_date DESC 
        LIMIT 100
        """
        return await self.db.fetch_dataframe(query)
    
    async def _prepare_game_features(
        self, 
        player_data: pd.DataFrame, 
        game_info: Dict
    ) -> pd.DataFrame:
        """Prepare features for a specific upcoming game."""
        # Create a row for the upcoming game
        upcoming_row = player_data.iloc[-1:].copy()
        
        # Update with game-specific information
        for key, value in game_info.items():
            if key in upcoming_row.columns:
                upcoming_row[key] = value
                
        # Prepare features
        features = await self.prepare_features(
            pd.concat([player_data, upcoming_row])
        )
        
        # Return only the last row (upcoming game)
        game_features = features.iloc[-1:][self.feature_columns]
        
        # Handle categorical encoding
        for col in game_features.columns:
            if col in self.encoders:
                game_features[col] = self.encoders[col].transform(
                    game_features[col].astype(str)
                )
                
        # Scale features
        if 'scaler' in self.scalers:
            game_features = pd.DataFrame(
                self.scalers['scaler'].transform(game_features),
                columns=self.feature_columns,
                index=game_features.index
            )
            
        return game_features
    
    async def _calculate_confidence_interval(
        self, 
        metric: str, 
        features: pd.DataFrame, 
        prediction: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        model_info = self.models[metric]
        
        # Use test set residuals to estimate uncertainty
        test_predictions = []
        for model_name, model in model_info['models'].items():
            test_pred = model.predict(model_info['X_test'])
            test_predictions.append(test_pred)
            
        # Calculate ensemble test predictions
        ensemble_test_pred = np.zeros_like(test_predictions[0])
        for i, (model_name, weight) in enumerate(model_info['weights'].items()):
            ensemble_test_pred += test_predictions[i] * weight
            
        # Calculate residuals
        residuals = model_info['y_test'] - ensemble_test_pred
        residual_std = np.std(residuals)
        
        # 95% confidence interval
        margin = 1.96 * residual_std
        
        return (prediction - margin, prediction + margin)
    
    @cache_result(ttl=3600)
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics."""
        performance = {}
        
        for metric, model_info in self.models.items():
            metric_performance = {}
            
            for model_name, model in model_info['models'].items():
                # Test set predictions
                y_pred = model.predict(model_info['X_test'])
                y_true = model_info['y_test']
                
                metric_performance[model_name] = {
                    'r2_score': float(r2_score(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'weight': float(model_info['weights'][model_name])
                }
                
            performance[metric] = metric_performance
            
        return performance
    
    async def save_models(self, filepath: str) -> None:
        """Save trained models to disk."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'target_metrics': self.target_metrics,
            'sport': self.sport
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    async def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_metrics = model_data['target_metrics']
        self.sport = model_data['sport']
        
        logger.info(f"Models loaded from {filepath}") 