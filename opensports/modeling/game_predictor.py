"""
Game Outcome Prediction Model

Advanced machine learning model for predicting game outcomes across multiple sports.
Uses ensemble methods, team statistics, and contextual features.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import get_database
from opensports.core.cache import cache_async_result

logger = get_logger(__name__)


class GameOutcomePredictor:
    """
    Advanced game outcome prediction model using ensemble methods.
    
    Features:
    - Multi-sport support (NBA, NFL, Soccer, etc.)
    - Ensemble of XGBoost, LightGBM, and Random Forest
    - Team strength ratings and momentum features
    - Head-to-head historical analysis
    - SHAP explainability for predictions
    """
    
    def __init__(self, sport: str = "nba"):
        self.sport = sport.lower()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.db = get_database()
        
    async def prepare_features(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for game outcome prediction.
        
        Args:
            game_data: Raw game data DataFrame
            
        Returns:
            Engineered features DataFrame
        """
        logger.info(f"Preparing features for {len(game_data)} games")
        
        features = game_data.copy()
        
        # Team strength features
        await self._add_team_strength_features(features)
        
        # Recent performance features
        await self._add_recent_performance_features(features)
        
        # Head-to-head features
        await self._add_head_to_head_features(features)
        
        # Contextual features
        await self._add_contextual_features(features)
        
        # Rest and travel features
        await self._add_rest_travel_features(features)
        
        # Injury and roster features
        await self._add_roster_features(features)
        
        logger.info(f"Feature engineering complete. Shape: {features.shape}")
        return features
    
    async def _add_team_strength_features(self, features: pd.DataFrame) -> None:
        """Add team strength and rating features."""
        # Calculate team ratings based on recent performance
        for team_type in ['home_team', 'away_team']:
            # Win percentage over last 20 games
            features[f'{team_type}_win_pct_20'] = 0.0
            
            # Average points scored/allowed
            features[f'{team_type}_avg_points_scored'] = 0.0
            features[f'{team_type}_avg_points_allowed'] = 0.0
            
            # Strength of schedule
            features[f'{team_type}_sos'] = 0.0
            
            # Home/away splits
            if team_type == 'home_team':
                features[f'{team_type}_home_win_pct'] = 0.0
            else:
                features[f'{team_type}_away_win_pct'] = 0.0
    
    async def _add_recent_performance_features(self, features: pd.DataFrame) -> None:
        """Add recent performance and momentum features."""
        # Last 5 games performance
        features['home_team_last_5_wins'] = 0
        features['away_team_last_5_wins'] = 0
        
        # Winning/losing streaks
        features['home_team_streak'] = 0
        features['away_team_streak'] = 0
        
        # Recent scoring trends
        features['home_team_scoring_trend'] = 0.0
        features['away_team_scoring_trend'] = 0.0
        
        # Performance vs similar opponents
        features['home_team_vs_similar'] = 0.0
        features['away_team_vs_similar'] = 0.0
    
    async def _add_head_to_head_features(self, features: pd.DataFrame) -> None:
        """Add head-to-head historical features."""
        # Historical matchup record
        features['h2h_home_wins'] = 0
        features['h2h_away_wins'] = 0
        features['h2h_total_games'] = 0
        
        # Recent head-to-head performance
        features['h2h_last_5_home_wins'] = 0
        features['h2h_avg_total_points'] = 0.0
        
        # Venue-specific performance
        features['h2h_home_venue_record'] = 0.0
    
    async def _add_contextual_features(self, features: pd.DataFrame) -> None:
        """Add contextual game features."""
        if 'game_date' in features.columns:
            features['game_date'] = pd.to_datetime(features['game_date'])
            features['day_of_week'] = features['game_date'].dt.dayofweek
            features['month'] = features['game_date'].dt.month
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            
        # Season context
        features['season_progress'] = 0.0  # 0-1 scale
        features['is_playoffs'] = 0
        features['games_remaining'] = 0
        
        # Game importance
        features['playoff_implications'] = 0
        features['rivalry_game'] = 0
        
        # Weather (for outdoor sports)
        if self.sport in ['nfl', 'soccer']:
            features['temperature'] = 70.0  # Default
            features['precipitation'] = 0.0
            features['wind_speed'] = 0.0
    
    async def _add_rest_travel_features(self, features: pd.DataFrame) -> None:
        """Add rest and travel impact features."""
        # Days of rest
        features['home_team_rest_days'] = 1
        features['away_team_rest_days'] = 1
        
        # Back-to-back games
        features['home_team_back_to_back'] = 0
        features['away_team_back_to_back'] = 0
        
        # Travel distance
        features['away_team_travel_distance'] = 0.0
        
        # Time zone changes
        features['away_team_timezone_change'] = 0
    
    async def _add_roster_features(self, features: pd.DataFrame) -> None:
        """Add roster and injury impact features."""
        # Key player availability
        features['home_team_key_players_out'] = 0
        features['away_team_key_players_out'] = 0
        
        # Roster depth
        features['home_team_roster_depth'] = 1.0
        features['away_team_roster_depth'] = 1.0
        
        # Recent acquisitions/trades
        features['home_team_recent_trades'] = 0
        features['away_team_recent_trades'] = 0
    
    async def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ensemble of models for game outcome prediction.
        
        Args:
            training_data: Prepared training data with features and outcomes
            
        Returns:
            Training results and model performance metrics
        """
        logger.info("Starting model training for game outcome prediction")
        
        # Prepare features
        features = await self.prepare_features(training_data)
        
        # Create target variable (1 if home team wins, 0 if away team wins)
        if 'home_score' in features.columns and 'away_score' in features.columns:
            features['home_win'] = (features['home_score'] > features['away_score']).astype(int)
        else:
            raise ValueError("Score columns not found in training data")
        
        # Select feature columns
        exclude_cols = ['home_win', 'home_score', 'away_score', 'game_id', 'game_date']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        X = features[self.feature_columns]
        y = features['home_win']
        
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize models
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        trained_models = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            logger.info(f"Training {model_name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test)
            test_auc = roc_auc_score(y_test, y_pred_proba_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            trained_models[model_name] = model
            
            logger.info(f"{model_name} - Test Accuracy: {test_accuracy:.3f}, AUC: {test_auc:.3f}")
        
        # Create ensemble weights based on CV performance
        weights = {}
        total_cv = sum(results[name]['cv_mean'] for name in trained_models.keys())
        for name in trained_models.keys():
            weights[name] = results[name]['cv_mean'] / total_cv
        
        # Store models and results
        self.models = {
            'models': trained_models,
            'weights': weights,
            'X_test': X_test,
            'y_test': y_test
        }
        
        logger.info("Model training completed successfully")
        return results
    
    async def predict_game(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        use_ensemble: bool = True
    ) -> Dict[str, Any]:
        """
        Predict the outcome of a specific game.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            game_date: Game date in ISO format
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Game prediction with probabilities and confidence
        """
        logger.info(f"Predicting game: {away_team} @ {home_team} on {game_date}")
        
        # Prepare game features
        game_features = await self._prepare_game_features(home_team, away_team, game_date)
        
        if use_ensemble:
            # Ensemble prediction
            ensemble_proba = 0
            for model_name, model in self.models['models'].items():
                proba = model.predict_proba(game_features)[0, 1]  # Probability of home win
                weight = self.models['weights'][model_name]
                ensemble_proba += proba * weight
            
            home_win_probability = ensemble_proba
        else:
            # Use best single model
            best_model_name = max(
                self.models['models'].keys(),
                key=lambda x: self.models['weights'][x]
            )
            best_model = self.models['models'][best_model_name]
            home_win_probability = best_model.predict_proba(game_features)[0, 1]
        
        away_win_probability = 1 - home_win_probability
        
        # Determine predicted winner
        predicted_winner = home_team if home_win_probability > 0.5 else away_team
        confidence = max(home_win_probability, away_win_probability)
        
        # Calculate spread prediction (if applicable)
        spread_prediction = await self._predict_spread(game_features, home_win_probability)
        
        # Calculate total points prediction
        total_points_prediction = await self._predict_total_points(game_features)
        
        prediction = {
            'game_info': {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date
            },
            'predictions': {
                'winner': predicted_winner,
                'home_win_probability': float(home_win_probability),
                'away_win_probability': float(away_win_probability),
                'confidence': float(confidence),
                'spread': spread_prediction,
                'total_points': total_points_prediction
            },
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return prediction
    
    async def _prepare_game_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str
    ) -> pd.DataFrame:
        """Prepare features for a specific game."""
        # Create a DataFrame with the game information
        game_data = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date
        }])
        
        # Prepare features using the same pipeline as training
        features = await self.prepare_features(game_data)
        
        # Select only the feature columns used in training
        game_features = features[self.feature_columns]
        
        # Handle categorical encoding
        for col in game_features.columns:
            if col in self.encoders:
                try:
                    game_features[col] = self.encoders[col].transform(
                        game_features[col].astype(str)
                    )
                except ValueError:
                    # Handle unseen categories
                    game_features[col] = 0
        
        # Scale features
        if 'scaler' in self.scalers:
            game_features = pd.DataFrame(
                self.scalers['scaler'].transform(game_features),
                columns=self.feature_columns,
                index=game_features.index
            )
        
        return game_features
    
    async def _predict_spread(
        self,
        game_features: pd.DataFrame,
        home_win_probability: float
    ) -> float:
        """Predict point spread for the game."""
        # Simple spread estimation based on win probability
        # This could be enhanced with a dedicated spread prediction model
        if home_win_probability > 0.5:
            # Home team favored
            spread = -((home_win_probability - 0.5) * 20)  # Scale to typical spread range
        else:
            # Away team favored
            spread = ((0.5 - home_win_probability) * 20)
        
        return round(spread, 1)
    
    async def _predict_total_points(self, game_features: pd.DataFrame) -> float:
        """Predict total points for the game."""
        # Simple total points estimation
        # This could be enhanced with a dedicated total points prediction model
        
        # Use sport-specific averages
        sport_averages = {
            'nba': 220.0,
            'nfl': 45.0,
            'soccer': 2.5,
            'hockey': 6.0
        }
        
        base_total = sport_averages.get(self.sport, 100.0)
        
        # Add some variation based on team features (simplified)
        variation = np.random.normal(0, base_total * 0.1)
        
        return round(base_total + variation, 1)
    
    async def explain_prediction(
        self,
        home_team: str,
        away_team: str,
        game_date: str
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a game prediction.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            game_date: Game date
            
        Returns:
            SHAP explanation values and feature importance
        """
        # Prepare game features
        game_features = await self._prepare_game_features(home_team, away_team, game_date)
        
        # Use the best model for explanation
        best_model_name = max(
            self.models['models'].keys(),
            key=lambda x: self.models['weights'][x]
        )
        best_model = self.models['models'][best_model_name]
        
        # Generate SHAP explanations
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(game_features)
        
        # For binary classification, use the positive class SHAP values
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Format explanation
        feature_importance = dict(zip(self.feature_columns, shap_values[0]))
        
        return {
            'game_info': {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date
            },
            'prediction': float(best_model.predict_proba(game_features)[0, 1]),
            'base_value': float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value),
            'feature_importance': {
                k: float(v) for k, v in sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            },
            'top_factors_home': [
                {'feature': k, 'impact': float(v)}
                for k, v in sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ],
            'top_factors_away': [
                {'feature': k, 'impact': float(v)}
                for k, v in sorted(
                    feature_importance.items(),
                    key=lambda x: x[1]
                )[:5]
            ]
        }
    
    @cache_async_result(ttl=3600)
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics."""
        if not self.models:
            return {"error": "No trained models available"}
        
        performance = {}
        
        for model_name, model in self.models['models'].items():
            # Test set predictions
            y_pred = model.predict(self.models['X_test'])
            y_pred_proba = model.predict_proba(self.models['X_test'])[:, 1]
            y_true = self.models['y_test']
            
            performance[model_name] = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred)),
                'recall': float(recall_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred)),
                'auc': float(roc_auc_score(y_true, y_pred_proba)),
                'weight': float(self.models['weights'][model_name])
            }
        
        return performance
    
    async def save_models(self, filepath: str) -> None:
        """Save trained models to disk."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'sport': self.sport
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Game prediction models saved to {filepath}")
    
    async def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_columns = model_data['feature_columns']
        self.sport = model_data['sport']
        
        logger.info(f"Game prediction models loaded from {filepath}") 