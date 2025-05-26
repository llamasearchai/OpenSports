"""
Advanced Machine Learning Pipeline for Sports Analytics

Comprehensive ML pipeline with automated feature engineering, model training,
hyperparameter optimization, validation, and deployment capabilities.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import logging
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import optuna
import mlflow
import mlflow.sklearn
from joblib import Parallel, delayed

from ..core.database import DatabaseManager
from ..core.cache import CacheManager
from ..core.config import Config
from .features import FeatureEngineer, FeatureSelector
from .models import PlayerPerformancePredictor, GameOutcomePredictor
from .training import ModelTrainer, HyperparameterOptimizer
from .evaluation import ModelEvaluator
from .deployment import ModelDeployer

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""
    # Data settings
    train_test_split_ratio: float = 0.8
    validation_split_ratio: float = 0.2
    time_series_split: bool = True
    n_splits: int = 5
    
    # Feature engineering
    feature_selection: bool = True
    feature_selection_method: str = "mutual_info"
    max_features: int = 100
    polynomial_features: bool = False
    interaction_features: bool = True
    
    # Model settings
    models_to_train: List[str] = field(default_factory=lambda: [
        'random_forest', 'gradient_boosting', 'linear_regression', 'xgboost'
    ])
    ensemble_method: str = "voting"
    
    # Hyperparameter optimization
    hyperparameter_optimization: bool = True
    optimization_trials: int = 100
    optimization_timeout: int = 3600  # seconds
    
    # Training settings
    cross_validation: bool = True
    early_stopping: bool = True
    patience: int = 10
    
    # Deployment settings
    auto_deploy: bool = False
    deployment_threshold: float = 0.85  # minimum accuracy for deployment
    model_registry: bool = True
    
    # Monitoring
    drift_detection: bool = True
    performance_monitoring: bool = True
    alert_threshold: float = 0.1  # performance degradation threshold

class MLPipeline:
    """
    Comprehensive machine learning pipeline for sports analytics.
    
    Features:
    - Automated feature engineering and selection
    - Multiple model training with hyperparameter optimization
    - Cross-validation and ensemble methods
    - Model evaluation and comparison
    - Automated deployment and monitoring
    - Experiment tracking with MLflow
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.app_config = Config()
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.hyperopt = HyperparameterOptimizer()
        self.evaluator = ModelEvaluator()
        self.deployer = ModelDeployer()
        
        # Pipeline state
        self.models = {}
        self.feature_columns = []
        self.scalers = {}
        self.encoders = {}
        self.pipeline_metadata = {}
        
        # MLflow setup
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("opensports_ml_pipeline")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    
    async def run_full_pipeline(self, 
                               data: pd.DataFrame,
                               target_column: str,
                               task_type: str = "regression",
                               experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete ML pipeline from data to deployment.
        
        Args:
            data: Input dataset
            target_column: Name of target variable
            task_type: Type of ML task (regression, classification)
            experiment_name: Name for the experiment
            
        Returns:
            Dictionary containing pipeline results and metadata
        """
        try:
            logger.info("Starting ML pipeline execution")
            
            with mlflow.start_run(run_name=experiment_name):
                # Log pipeline configuration
                mlflow.log_params(self.config.__dict__)
                
                # Step 1: Data preprocessing
                logger.info("Step 1: Data preprocessing")
                processed_data = await self._preprocess_data(data, target_column)
                
                # Step 2: Feature engineering
                logger.info("Step 2: Feature engineering")
                engineered_data = await self._engineer_features(processed_data, target_column)
                
                # Step 3: Feature selection
                logger.info("Step 3: Feature selection")
                selected_data = await self._select_features(engineered_data, target_column, task_type)
                
                # Step 4: Data splitting
                logger.info("Step 4: Data splitting")
                train_data, test_data = await self._split_data(selected_data, target_column)
                
                # Step 5: Model training
                logger.info("Step 5: Model training")
                trained_models = await self._train_models(train_data, target_column, task_type)
                
                # Step 6: Model evaluation
                logger.info("Step 6: Model evaluation")
                evaluation_results = await self._evaluate_models(trained_models, test_data, target_column, task_type)
                
                # Step 7: Model ensemble
                logger.info("Step 7: Model ensemble")
                ensemble_model = await self._create_ensemble(trained_models, train_data, target_column, task_type)
                
                # Step 8: Final evaluation
                logger.info("Step 8: Final evaluation")
                final_results = await self._final_evaluation(ensemble_model, test_data, target_column, task_type)
                
                # Step 9: Model deployment (if configured)
                deployment_result = None
                if self.config.auto_deploy and final_results['accuracy'] >= self.config.deployment_threshold:
                    logger.info("Step 9: Model deployment")
                    deployment_result = await self._deploy_model(ensemble_model, final_results)
                
                # Compile results
                pipeline_results = {
                    'experiment_name': experiment_name,
                    'config': self.config.__dict__,
                    'data_shape': data.shape,
                    'feature_count': len(self.feature_columns),
                    'models_trained': list(trained_models.keys()),
                    'evaluation_results': evaluation_results,
                    'final_results': final_results,
                    'deployment_result': deployment_result,
                    'pipeline_metadata': self.pipeline_metadata,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Log final results
                mlflow.log_metrics(final_results)
                mlflow.log_artifact("pipeline_results.json")
                
                logger.info("ML pipeline execution completed successfully")
                return pipeline_results
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def _preprocess_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Preprocess the input data."""
        try:
            processed_data = data.copy()
            
            # Handle missing values
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            categorical_columns = processed_data.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                if col != target_column:
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if col != target_column:
                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
            
            # Encode categorical variables
            for col in categorical_columns:
                if col != target_column:
                    encoder = LabelEncoder()
                    processed_data[col] = encoder.fit_transform(processed_data[col].astype(str))
                    self.encoders[col] = encoder
            
            # Remove outliers using IQR method
            for col in numeric_columns:
                if col != target_column:
                    Q1 = processed_data[col].quantile(0.25)
                    Q3 = processed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_data = processed_data[
                        (processed_data[col] >= lower_bound) & 
                        (processed_data[col] <= upper_bound)
                    ]
            
            self.pipeline_metadata['preprocessing'] = {
                'original_shape': data.shape,
                'processed_shape': processed_data.shape,
                'outliers_removed': data.shape[0] - processed_data.shape[0],
                'encoders_used': list(self.encoders.keys())
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    async def _engineer_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Engineer new features from the data."""
        try:
            engineered_data = await self.feature_engineer.engineer_features(
                data, 
                target_column,
                polynomial_features=self.config.polynomial_features,
                interaction_features=self.config.interaction_features
            )
            
            self.pipeline_metadata['feature_engineering'] = {
                'original_features': data.shape[1] - 1,
                'engineered_features': engineered_data.shape[1] - 1,
                'new_features_created': engineered_data.shape[1] - data.shape[1]
            }
            
            return engineered_data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    async def _select_features(self, data: pd.DataFrame, target_column: str, task_type: str) -> pd.DataFrame:
        """Select the most important features."""
        try:
            if not self.config.feature_selection:
                self.feature_columns = [col for col in data.columns if col != target_column]
                return data
            
            selected_data = await self.feature_selector.select_features(
                data,
                target_column,
                method=self.config.feature_selection_method,
                max_features=self.config.max_features,
                task_type=task_type
            )
            
            self.feature_columns = [col for col in selected_data.columns if col != target_column]
            
            self.pipeline_metadata['feature_selection'] = {
                'method': self.config.feature_selection_method,
                'features_before': data.shape[1] - 1,
                'features_after': len(self.feature_columns),
                'selected_features': self.feature_columns[:10]  # Top 10 for logging
            }
            
            return selected_data
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise
    
    async def _split_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        try:
            if self.config.time_series_split:
                # For time series data, use temporal split
                split_index = int(len(data) * self.config.train_test_split_ratio)
                train_data = data.iloc[:split_index]
                test_data = data.iloc[split_index:]
            else:
                # Random split for non-time series data
                train_data, test_data = train_test_split(
                    data,
                    test_size=1 - self.config.train_test_split_ratio,
                    random_state=42,
                    stratify=data[target_column] if data[target_column].dtype == 'object' else None
                )
            
            # Scale features
            scaler = StandardScaler()
            feature_columns = [col for col in data.columns if col != target_column]
            
            train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
            test_data[feature_columns] = scaler.transform(test_data[feature_columns])
            
            self.scalers['features'] = scaler
            
            self.pipeline_metadata['data_split'] = {
                'train_size': len(train_data),
                'test_size': len(test_data),
                'split_method': 'time_series' if self.config.time_series_split else 'random'
            }
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            raise
    
    async def _train_models(self, train_data: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        """Train multiple models with hyperparameter optimization."""
        try:
            trained_models = {}
            X_train = train_data[self.feature_columns]
            y_train = train_data[target_column]
            
            # Train models in parallel
            training_tasks = []
            for model_name in self.config.models_to_train:
                task = self._train_single_model(model_name, X_train, y_train, task_type)
                training_tasks.append(task)
            
            # Wait for all training tasks to complete
            training_results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(training_results):
                model_name = self.config.models_to_train[i]
                if isinstance(result, Exception):
                    logger.error(f"Training failed for {model_name}: {result}")
                else:
                    trained_models[model_name] = result
            
            self.models = trained_models
            
            self.pipeline_metadata['model_training'] = {
                'models_trained': list(trained_models.keys()),
                'training_samples': len(X_train),
                'feature_count': len(self.feature_columns)
            }
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def _train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, task_type: str) -> Any:
        """Train a single model with hyperparameter optimization."""
        try:
            # Get base model
            model = self.model_trainer.get_model(model_name, task_type)
            
            if self.config.hyperparameter_optimization:
                # Optimize hyperparameters
                optimized_model = await self.hyperopt.optimize_model(
                    model, X_train, y_train, task_type,
                    n_trials=self.config.optimization_trials,
                    timeout=self.config.optimization_timeout
                )
                return optimized_model
            else:
                # Train with default parameters
                model.fit(X_train, y_train)
                return model
                
        except Exception as e:
            logger.error(f"Single model training failed for {model_name}: {e}")
            raise
    
    async def _evaluate_models(self, models: Dict[str, Any], test_data: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        try:
            evaluation_results = {}
            X_test = test_data[self.feature_columns]
            y_test = test_data[target_column]
            
            for model_name, model in models.items():
                try:
                    results = await self.evaluator.evaluate_model(
                        model, X_test, y_test, task_type
                    )
                    evaluation_results[model_name] = results
                    
                    # Log to MLflow
                    for metric_name, metric_value in results.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
                        
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name}: {e}")
                    evaluation_results[model_name] = {'error': str(e)}
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    async def _create_ensemble(self, models: Dict[str, Any], train_data: pd.DataFrame, target_column: str, task_type: str) -> Any:
        """Create ensemble model from trained models."""
        try:
            from sklearn.ensemble import VotingRegressor, VotingClassifier
            
            # Prepare models for ensemble
            model_list = [(name, model) for name, model in models.items()]
            
            if task_type == "regression":
                ensemble = VotingRegressor(estimators=model_list, n_jobs=-1)
            else:
                ensemble = VotingClassifier(estimators=model_list, voting='soft', n_jobs=-1)
            
            # Train ensemble
            X_train = train_data[self.feature_columns]
            y_train = train_data[target_column]
            ensemble.fit(X_train, y_train)
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            raise
    
    async def _final_evaluation(self, ensemble_model: Any, test_data: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, float]:
        """Perform final evaluation on ensemble model."""
        try:
            X_test = test_data[self.feature_columns]
            y_test = test_data[target_column]
            
            final_results = await self.evaluator.evaluate_model(
                ensemble_model, X_test, y_test, task_type
            )
            
            # Add ensemble-specific metrics
            final_results['ensemble_size'] = len(ensemble_model.estimators_)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            raise
    
    async def _deploy_model(self, model: Any, evaluation_results: Dict[str, float]) -> Dict[str, Any]:
        """Deploy the trained model."""
        try:
            deployment_result = await self.deployer.deploy_model(
                model=model,
                model_metadata={
                    'feature_columns': self.feature_columns,
                    'scalers': self.scalers,
                    'encoders': self.encoders,
                    'evaluation_results': evaluation_results,
                    'pipeline_config': self.config.__dict__
                }
            )
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the entire pipeline to disk."""
        try:
            pipeline_data = {
                'config': self.config,
                'models': self.models,
                'feature_columns': self.feature_columns,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'pipeline_metadata': self.pipeline_metadata
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(pipeline_data, f)
                
            logger.info(f"Pipeline saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise
    
    def load_pipeline(self, filepath: str) -> None:
        """Load pipeline from disk."""
        try:
            with open(filepath, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            self.config = pipeline_data['config']
            self.models = pipeline_data['models']
            self.feature_columns = pipeline_data['feature_columns']
            self.scalers = pipeline_data['scalers']
            self.encoders = pipeline_data['encoders']
            self.pipeline_metadata = pipeline_data['pipeline_metadata']
            
            logger.info(f"Pipeline loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    async def predict(self, data: pd.DataFrame, model_name: str = "ensemble") -> np.ndarray:
        """Make predictions using trained models."""
        try:
            # Preprocess input data
            processed_data = data.copy()
            
            # Apply encoders
            for col, encoder in self.encoders.items():
                if col in processed_data.columns:
                    processed_data[col] = encoder.transform(processed_data[col].astype(str))
            
            # Apply scalers
            if 'features' in self.scalers:
                feature_cols = [col for col in self.feature_columns if col in processed_data.columns]
                processed_data[feature_cols] = self.scalers['features'].transform(processed_data[feature_cols])
            
            # Select features
            X = processed_data[self.feature_columns]
            
            # Make predictions
            if model_name == "ensemble" and hasattr(self, 'ensemble_model'):
                predictions = self.ensemble_model.predict(X)
            elif model_name in self.models:
                predictions = self.models[model_name].predict(X)
            else:
                raise ValueError(f"Model {model_name} not found")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise 