import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pandas.testing import assert_frame_equal, assert_series_equal
from OpenInsight.modeling.lead_scoring import LeadScoringModel, LeadScoringService

# Test data generation
def generate_test_data(n_samples=100, random_state=42):
    """Generate synthetic data for testing lead scoring models."""
    rng = np.random.RandomState(random_state)
    
    # Generate features
    X = pd.DataFrame({
        'feature1': rng.normal(0, 1, n_samples),
        'feature2': rng.normal(0, 1, n_samples),
        'feature3': rng.uniform(0, 1, n_samples),
        'feature4': rng.choice([0, 1], n_samples)
    })
    
    # Generate target: Higher values of feature1 and feature3 increase likelihood of conversion
    # feature2 has a smaller negative impact, feature4 has no impact
    logits = 0.5 * X['feature1'] + 0.8 * X['feature3'] - 0.2 * X['feature2']
    probabilities = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, n_samples) < probabilities).astype(int)
    
    return X, pd.Series(y)

class TestLeadScoringModel:
    @pytest.fixture
    def test_data(self):
        """Fixture for test data."""
        X, y = generate_test_data()
        return X, y
    
    @pytest.fixture
    def model(self):
        """Fixture for a lead scoring model."""
        return LeadScoringModel(model_id="test_model", model_type="random_forest")
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        model = LeadScoringModel(model_id="test_init", model_type="random_forest")
        assert model.model_id == "test_init"
        assert model.model_type == "random_forest"
        assert model.model is None
        assert model.feature_names is None
        assert model.explainer is None
        assert model.metrics == {}
        assert model.trained_at is None
    
    def test_initialization_invalid_model_type(self):
        """Test that initialization fails with invalid model type."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            LeadScoringModel(model_id="test_invalid", model_type="invalid_model_type")
    
    def test_create_model(self, model):
        """Test that _create_model returns the correct type of model."""
        rf_model = model._create_model()
        assert rf_model.__class__.__name__ == "RandomForestClassifier"
        
        gb_model = LeadScoringModel(model_id="test_gb", model_type="gradient_boosting")._create_model()
        assert gb_model.__class__.__name__ == "GradientBoostingClassifier"
        
        lr_model = LeadScoringModel(model_id="test_lr", model_type="logistic_regression")._create_model()
        assert lr_model.__class__.__name__ == "LogisticRegression"
    
    def test_train(self, model, test_data):
        """Test model training."""
        X, y = test_data
        metrics = model.train(X, y)
        
        # Check that the model was trained
        assert model.model is not None
        assert model.feature_names == X.columns.tolist()
        assert model.explainer is not None
        assert model.trained_at is not None
        
        # Check that metrics were computed
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        
        # Check that all metrics are between 0 and 1
        for metric, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_predict_without_training(self, model, test_data):
        """Test that predict fails if the model is not trained."""
        X, _ = test_data
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)
    
    def test_predict(self, model, test_data):
        """Test model prediction."""
        X, y = test_data
        
        # Train the model
        model.train(X, y)
        
        # Test with the same data
        scores = model.predict(X)
        
        # Check that scores have the right shape and range
        assert len(scores) == len(X)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_predict_missing_features(self, model, test_data):
        """Test that predict fails if features are missing."""
        X, y = test_data
        
        # Train the model
        model.train(X, y)
        
        # Test with missing features
        X_missing = X.drop(columns=['feature1'])
        with pytest.raises(ValueError, match="Missing features"):
            model.predict(X_missing)
    
    def test_explain_without_training(self, model, test_data):
        """Test that explain fails if the model is not trained."""
        X, _ = test_data
        with pytest.raises(ValueError, match="Model not trained"):
            model.explain(X)
    
    def test_explain(self, model, test_data):
        """Test model explanation."""
        X, y = test_data
        
        # Train the model
        model.train(X, y)
        
        # Get explanations for a few samples
        X_sample = X.head(5)
        explanations = model.explain(X_sample)
        
        # Check that explanations have the right structure
        assert "explanations" in explanations
        assert len(explanations["explanations"]) == len(X_sample)
        
        # Check the structure of an explanation
        explanation = explanations["explanations"][0]
        assert "base_value" in explanation
        assert "feature_impacts" in explanation
        assert len(explanation["feature_impacts"]) == len(X.columns)
        
        # Check that feature impacts are sorted by absolute impact
        feature_impacts = explanation["feature_impacts"]
        absolute_impacts = [abs(fi["impact"]) for fi in feature_impacts]
        assert all(absolute_impacts[i] >= absolute_impacts[i+1] for i in range(len(absolute_impacts)-1))
    
    def test_explain_missing_features(self, model, test_data):
        """Test that explain fails if features are missing."""
        X, y = test_data
        
        # Train the model
        model.train(X, y)
        
        # Test with missing features
        X_missing = X.drop(columns=['feature1'])
        with pytest.raises(ValueError, match="Missing features"):
            model.explain(X_missing)
    
    def test_save_load(self, model, test_data, tmpdir):
        """Test saving and loading a model."""
        X, y = test_data
        
        # Train the model
        model.train(X, y)
        
        # Save the model
        model_path = model.save(tmpdir)
        assert os.path.exists(model_path)
        
        # Load the model
        loaded_model = LeadScoringModel.load(model_path)
        assert loaded_model.model_id == model.model_id
        assert loaded_model.model_type == model.model_type
        assert loaded_model.feature_names == model.feature_names
        assert loaded_model.metrics == model.metrics
        
        # Test that the loaded model makes the same predictions
        scores_original = model.predict(X)
        scores_loaded = loaded_model.predict(X)
        np.testing.assert_allclose(scores_original, scores_loaded)
    
    def test_save_without_training(self, model, tmpdir):
        """Test that save fails if the model is not trained."""
        with pytest.raises(ValueError, match="Model not trained"):
            model.save(tmpdir)
    
    def test_load_nonexistent_file(self):
        """Test that load fails if the file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LeadScoringModel.load("/path/to/nonexistent/file")
    
    def test_get_info(self, model, test_data):
        """Test getting model info."""
        X, y = test_data
        
        # Before training
        info = model.get_info()
        assert info["model_id"] == model.model_id
        assert info["model_type"] == model.model_type
        assert info["feature_names"] is None
        assert info["metrics"] == {}
        assert info["trained_at"] is None
        assert info["is_trained"] is False
        
        # After training
        model.train(X, y)
        info = model.get_info()
        assert info["model_id"] == model.model_id
        assert info["model_type"] == model.model_type
        assert info["feature_names"] == X.columns.tolist()
        assert info["metrics"] == model.metrics
        assert info["trained_at"] is not None
        assert info["is_trained"] is True

class TestLeadScoringService:
    @pytest.fixture
    def service(self):
        """Fixture for a lead scoring service with a temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield LeadScoringService(models_directory=tmp_dir)
    
    @pytest.fixture
    def test_data(self):
        """Fixture for test data."""
        X, y = generate_test_data()
        return X, y
    
    def test_initialization(self, service):
        """Test that the service initializes correctly."""
        assert hasattr(service, "models_directory")
        assert hasattr(service, "models")
        assert isinstance(service.models, dict)
        assert len(service.models) == 0
    
    def test_create_model(self, service):
        """Test creating a model."""
        model = service.create_model("test_create", "random_forest")
        assert model is not None
        assert model.model_id == "test_create"
        assert model.model_type == "random_forest"
        assert "test_create" in service.models
        assert service.models["test_create"] is model
    
    def test_create_model_duplicate(self, service):
        """Test that creating a duplicate model fails."""
        service.create_model("test_dup", "random_forest")
        with pytest.raises(ValueError, match="already exists"):
            service.create_model("test_dup", "gradient_boosting")
    
    def test_get_model(self, service):
        """Test getting a model."""
        # Create a model
        model = service.create_model("test_get", "random_forest")
        
        # Get the model
        retrieved_model = service.get_model("test_get")
        assert retrieved_model is model
        
        # Try to get a non-existent model
        assert service.get_model("nonexistent") is None
    
    def test_delete_model(self, service):
        """Test deleting a model."""
        # Create a model
        service.create_model("test_delete", "random_forest")
        
        # Delete the model
        result = service.delete_model("test_delete")
        assert result is True
        assert "test_delete" not in service.models
        
        # Try to delete a non-existent model
        result = service.delete_model("nonexistent")
        assert result is False
    
    def test_list_models(self, service):
        """Test listing models."""
        # Create some models
        service.create_model("model1", "random_forest")
        service.create_model("model2", "gradient_boosting")
        
        # List models
        models = service.list_models()
        assert len(models) == 2
        assert any(m["model_id"] == "model1" for m in models)
        assert any(m["model_id"] == "model2" for m in models)
    
    def test_train_model(self, service, test_data):
        """Test training a model."""
        X, y = test_data
        
        # Create a model
        service.create_model("test_train", "random_forest")
        
        # Train the model
        metrics = service.train_model("test_train", X, y)
        
        # Check that metrics were returned
        assert metrics is not None
        assert "accuracy" in metrics
        
        # Check that the model was trained
        model = service.get_model("test_train")
        assert model.model is not None
        assert model.feature_names == X.columns.tolist()
    
    def test_train_model_nonexistent(self, service, test_data):
        """Test that training a non-existent model fails."""
        X, y = test_data
        with pytest.raises(ValueError, match="not found"):
            service.train_model("nonexistent", X, y)
    
    def test_score_leads(self, service, test_data):
        """Test scoring leads."""
        X, y = test_data
        
        # Create and train a model
        service.create_model("test_score", "random_forest")
        service.train_model("test_score", X, y)
        
        # Score some leads
        result = service.score_leads("test_score", X.head(5))
        
        # Check that scores were returned
        assert "model_id" in result
        assert result["model_id"] == "test_score"
        assert "scores" in result
        assert len(result["scores"]) == 5
        assert all(0 <= score <= 1 for score in result["scores"])
    
    def test_score_leads_nonexistent(self, service, test_data):
        """Test that scoring with a non-existent model fails."""
        X, _ = test_data
        with pytest.raises(ValueError, match="not found"):
            service.score_leads("nonexistent", X)
    
    def test_explain_scores(self, service, test_data):
        """Test explaining scores."""
        X, y = test_data
        
        # Create and train a model
        service.create_model("test_explain", "random_forest")
        service.train_model("test_explain", X, y)
        
        # Explain some scores
        result = service.explain_scores("test_explain", X.head(3))
        
        # Check that explanations were returned
        assert "model_id" in result
        assert result["model_id"] == "test_explain"
        assert "explanations" in result
        assert len(result["explanations"]) == 3
    
    def test_explain_scores_nonexistent(self, service, test_data):
        """Test that explaining with a non-existent model fails."""
        X, _ = test_data
        with pytest.raises(ValueError, match="not found"):
            service.explain_scores("nonexistent", X) 