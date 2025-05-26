import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import joblib
import os
import shap
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

logger = structlog.get_logger(__name__)

class LeadScoringModel:
    """
    A machine learning model for scoring leads based on features.
    Uses scikit-learn for predictive modeling and SHAP for explainability.
    """
    
    VALID_MODEL_TYPES = ["random_forest", "gradient_boosting", "logistic_regression"]
    
    def __init__(self, model_id: str, model_type: str = "random_forest"):
        """
        Initialize a lead scoring model.
        
        Args:
            model_id: Unique identifier for this model
            model_type: Type of model to use (random_forest, gradient_boosting, logistic_regression)
        """
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type. Must be one of: {self.VALID_MODEL_TYPES}")
            
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.explainer = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.trained_at = None
        
    def _create_model(self) -> Any:
        """Create a scikit-learn model based on model_type."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        
    def train(self, 
             features: pd.DataFrame, 
             target: pd.Series,
             test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the lead scoring model.
        
        Args:
            features: DataFrame containing feature columns
            target: Series containing target values (1 for converted, 0 for not converted)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Training lead scoring model", 
                   model_id=self.model_id, 
                   model_type=self.model_type,
                   num_samples=len(features))
        
        # Store feature names for later use
        self.feature_names = features.columns.tolist()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        # Create and train the model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Create explainer for the model
        if self.model_type in ["random_forest", "gradient_boosting"]:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(
                self.model, 
                X_train,
                feature_dependence="independent"
            )
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
        
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        
        if y_prob is not None:
            self.metrics["auc"] = float(roc_auc_score(y_test, y_prob))
        
        self.trained_at = datetime.now()
        
        logger.info("Lead scoring model trained successfully", 
                   model_id=self.model_id,
                   metrics=self.metrics)
                   
        return self.metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate lead scores (probability of conversion) for input features.
        
        Args:
            features: DataFrame with same columns as training data
            
        Returns:
            Array of lead scores (probabilities between 0 and 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if not all(col in features.columns for col in self.feature_names):
            missing = set(self.feature_names) - set(features.columns)
            raise ValueError(f"Missing features in input data: {missing}")
        
        # Ensure features are in the same order as during training
        features = features[self.feature_names]
        
        # Get probability scores (class 1)
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(features)[:, 1]
        else:
            # Fall back to decision values if probabilities aren't available
            scores = self.model.decision_function(features)
            scores = 1 / (1 + np.exp(-scores))  # Convert to probability-like values
            
        return scores
        
    def explain(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain predictions using SHAP values.
        
        Args:
            features: DataFrame with same columns as training data
            
        Returns:
            Dictionary with SHAP explanations
        """
        if self.model is None or self.explainer is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if not all(col in features.columns for col in self.feature_names):
            missing = set(self.feature_names) - set(features.columns)
            raise ValueError(f"Missing features in input data: {missing}")
        
        # Ensure features are in the same order as during training
        features = features[self.feature_names]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # For tree models, shap_values can be a list where index 1 corresponds to class 1
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
            
        # Calculate base_value (expected_value)
        if hasattr(self.explainer, "expected_value"):
            expected_value = self.explainer.expected_value
            # If expected_value is a list, get the value for class 1
            if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
                expected_value = expected_value[1]
        else:
            expected_value = 0.5  # Default if not available
        
        # Create a list of dictionaries with feature names and SHAP values
        explanations = []
        for i in range(len(features)):
            feature_impacts = []
            for j, feature_name in enumerate(self.feature_names):
                feature_impacts.append({
                    "feature": feature_name,
                    "value": float(features.iloc[i, j]),
                    "impact": float(shap_values[i, j])
                })
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
            
            explanations.append({
                "base_value": float(expected_value),
                "feature_impacts": feature_impacts
            })
            
        return {
            "explanations": explanations
        }
    
    def save(self, directory: str = "./models") -> str:
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model in
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model, explainer and metadata
        model_path = os.path.join(directory, f"{self.model_id}.joblib")
        
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "trained_at": self.trained_at,
            # Note: SHAP explainers may not serialize well in all cases,
            # so we'll recreate them when loading
        }
        
        joblib.dump(model_data, model_path)
        logger.info("Model saved", model_id=self.model_id, path=model_path)
        
        return model_path
        
    @classmethod
    def load(cls, model_path: str) -> "LeadScoringModel":
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded LeadScoringModel instance
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Extract model ID from filename
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        
        # Create a new instance
        lead_scoring_model = cls(model_id=model_id, model_type=model_data["model_type"])
        
        # Restore model attributes
        lead_scoring_model.model = model_data["model"]
        lead_scoring_model.feature_names = model_data["feature_names"]
        lead_scoring_model.metrics = model_data.get("metrics", {})
        lead_scoring_model.trained_at = model_data.get("trained_at")
        
        # Create new explainer for the loaded model
        if lead_scoring_model.model_type in ["random_forest", "gradient_boosting"]:
            lead_scoring_model.explainer = shap.TreeExplainer(lead_scoring_model.model)
        else:
            # For linear models, we need some data to create the explainer
            # This will be handled when the model is used
            pass
            
        logger.info("Model loaded", model_id=model_id)
        
        return lead_scoring_model
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "is_trained": self.model is not None
        }


class LeadScoringService:
    """
    Service for managing multiple lead scoring models.
    """
    
    def __init__(self, models_directory: str = "./models"):
        """
        Initialize the lead scoring service.
        
        Args:
            models_directory: Directory to store models
        """
        self.models_directory = models_directory
        self.models: Dict[str, LeadScoringModel] = {}
        
        # Create directory if it doesn't exist
        os.makedirs(models_directory, exist_ok=True)
        
        # Load any existing models
        self._load_existing_models()
        
    def _load_existing_models(self) -> None:
        """Load existing models from the models directory."""
        if not os.path.exists(self.models_directory):
            return
            
        for filename in os.listdir(self.models_directory):
            if filename.endswith(".joblib"):
                try:
                    model_path = os.path.join(self.models_directory, filename)
                    model = LeadScoringModel.load(model_path)
                    self.models[model.model_id] = model
                    logger.info("Loaded existing model", model_id=model.model_id)
                except Exception as e:
                    logger.error("Failed to load model", filename=filename, error=str(e))
    
    def create_model(self, model_id: str, model_type: str = "random_forest") -> LeadScoringModel:
        """
        Create a new lead scoring model.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model to create
            
        Returns:
            The created LeadScoringModel
        """
        if model_id in self.models:
            raise ValueError(f"Model with ID '{model_id}' already exists")
            
        model = LeadScoringModel(model_id=model_id, model_type=model_type)
        self.models[model_id] = model
        
        logger.info("Created new model", model_id=model_id, model_type=model_type)
        
        return model
        
    def get_model(self, model_id: str) -> Optional[LeadScoringModel]:
        """
        Get a model by ID.
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            The LeadScoringModel if found, None otherwise
        """
        return self.models.get(model_id)
        
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if the model was deleted, False otherwise
        """
        if model_id not in self.models:
            return False
            
        # Remove from memory
        del self.models[model_id]
        
        # Remove from disk if exists
        model_path = os.path.join(self.models_directory, f"{model_id}.joblib")
        if os.path.exists(model_path):
            os.remove(model_path)
            
        logger.info("Deleted model", model_id=model_id)
        
        return True
        
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        return [model.get_info() for model in self.models.values()]
        
    def train_model(self, 
                   model_id: str, 
                   features: pd.DataFrame, 
                   target: pd.Series,
                   test_size: float = 0.2) -> Dict[str, float]:
        """
        Train a model.
        
        Args:
            model_id: ID of the model to train
            features: Feature data
            target: Target data
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        metrics = model.train(features, target, test_size)
        
        # Save the trained model
        model.save(self.models_directory)
        
        return metrics
        
    def score_leads(self, model_id: str, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Score leads using a model.
        
        Args:
            model_id: ID of the model to use
            features: Feature data for the leads to score
            
        Returns:
            Dictionary with scores
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        scores = model.predict(features)
        
        return {
            "model_id": model_id,
            "scores": scores.tolist()
        }
        
    def explain_scores(self, model_id: str, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain lead scores.
        
        Args:
            model_id: ID of the model to use
            features: Feature data to explain
            
        Returns:
            Dictionary with explanations
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        explanations = model.explain(features)
        
        return {
            "model_id": model_id,
            **explanations
        }

# Global instance
lead_scoring_service = LeadScoringService()

def get_lead_scoring_service() -> LeadScoringService:
    """Get the global lead scoring service instance."""
    return lead_scoring_service 