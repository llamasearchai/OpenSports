import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import json
import io
from OpenInsight.api.main import app
from OpenInsight.modeling.lead_scoring import get_lead_scoring_service

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the FastAPI app."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def clean_service():
    """Provides a clean lead scoring service."""
    service = get_lead_scoring_service()
    # Keep track of existing models to avoid deleting them
    existing_models = set(service.models.keys())
    
    yield service
    
    # Clean up any models created during the test
    for model_id in list(service.models.keys()):
        if model_id not in existing_models:
            service.delete_model(model_id)

def generate_test_data(n_samples=20, random_state=42):
    """Generate synthetic data for testing lead scoring endpoints."""
    rng = np.random.RandomState(random_state)
    
    # Generate features
    X = pd.DataFrame({
        'feature1': rng.normal(0, 1, n_samples),
        'feature2': rng.normal(0, 1, n_samples),
        'feature3': rng.uniform(0, 1, n_samples),
        'feature4': rng.choice([0, 1], n_samples)
    })
    
    # Generate target: Higher values of feature1 and feature3 increase likelihood of conversion
    logits = 0.5 * X['feature1'] + 0.8 * X['feature3'] - 0.2 * X['feature2']
    probabilities = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, n_samples) < probabilities).astype(int)
    
    return X, pd.Series(y)

def create_test_model(client, model_id="test_model", model_type="random_forest"):
    """Create a test model through the API."""
    response = client.post(
        "/lead_scoring/models",
        json={"model_id": model_id, "model_type": model_type}
    )
    return response

def train_test_model(client, model_id, X, y):
    """Train a test model through the API."""
    # Convert pandas DataFrame to list of dicts for JSON serialization
    features_list = X.to_dict(orient='records')
    
    response = client.post(
        f"/lead_scoring/models/{model_id}/train",
        json={
            "features": features_list,
            "target": y.tolist(),
            "test_size": 0.2
        }
    )
    return response

def create_and_train_test_model(client, clean_service, model_id="test_model"):
    """Helper to create and train a test model."""
    X, y = generate_test_data()
    create_test_model(client, model_id)
    train_test_model(client, model_id, X, y)
    return X, y

def dataframe_to_csv(df):
    """Convert a DataFrame to a CSV string."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

class TestLeadScoringAPI:
    def test_create_model_valid(self, client, clean_service):
        """Test creating a valid model."""
        response = create_test_model(client, "test_create_valid")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test_create_valid"
        assert data["model_type"] == "random_forest"
        assert data["is_trained"] is False
    
    def test_create_model_invalid_type(self, client, clean_service):
        """Test creating a model with invalid type."""
        response = client.post(
            "/lead_scoring/models",
            json={"model_id": "test_invalid_type", "model_type": "invalid_type"}
        )
        assert response.status_code == 400
        assert "Invalid model_type" in response.json()["detail"]
    
    def test_create_model_duplicate(self, client, clean_service):
        """Test creating a duplicate model."""
        # Create first model
        create_test_model(client, "test_duplicate")
        
        # Try to create duplicate
        response = create_test_model(client, "test_duplicate")
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    def test_list_models_empty(self, client, clean_service):
        """Test listing models when none exist."""
        response = client.get("/lead_scoring/models")
        assert response.status_code == 200
        # There might be existing models from other tests
        # We can't assert the length is 0, but we can check the response is a list
        assert isinstance(response.json(), list)
    
    def test_list_models_with_models(self, client, clean_service):
        """Test listing models when some exist."""
        # Create a couple of models
        create_test_model(client, "test_list_1")
        create_test_model(client, "test_list_2")
        
        # List models
        response = client.get("/lead_scoring/models")
        assert response.status_code == 200
        models = response.json()
        
        # Find our test models
        test_models = [m for m in models if m["model_id"] in ["test_list_1", "test_list_2"]]
        assert len(test_models) == 2
        assert {"test_list_1", "test_list_2"} == {m["model_id"] for m in test_models}
    
    def test_get_model(self, client, clean_service):
        """Test getting a specific model."""
        # Create a model
        create_test_model(client, "test_get")
        
        # Get the model
        response = client.get("/lead_scoring/models/test_get")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test_get"
        assert data["model_type"] == "random_forest"
    
    def test_get_model_nonexistent(self, client, clean_service):
        """Test getting a non-existent model."""
        response = client.get("/lead_scoring/models/nonexistent_model")
        assert response.status_code == 404
    
    def test_delete_model(self, client, clean_service):
        """Test deleting a model."""
        # Create a model
        create_test_model(client, "test_delete")
        
        # Delete the model
        response = client.delete("/lead_scoring/models/test_delete")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Try to get the deleted model
        response = client.get("/lead_scoring/models/test_delete")
        assert response.status_code == 404
    
    def test_delete_model_nonexistent(self, client, clean_service):
        """Test deleting a non-existent model."""
        response = client.delete("/lead_scoring/models/nonexistent_model")
        assert response.status_code == 404
    
    def test_train_model_json(self, client, clean_service):
        """Test training a model with JSON data."""
        # Create a model
        model_id = "test_train_json"
        create_test_model(client, model_id)
        
        # Generate test data
        X, y = generate_test_data()
        
        # Train the model
        response = train_test_model(client, model_id, X, y)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_id"] == model_id
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        
        # Verify the model was trained
        response = client.get(f"/lead_scoring/models/{model_id}")
        assert response.status_code == 200
        model_info = response.json()
        assert model_info["is_trained"] is True
        assert model_info["feature_names"] == X.columns.tolist()
    
    def test_train_model_json_nonexistent(self, client, clean_service):
        """Test training a non-existent model with JSON data."""
        X, y = generate_test_data()
        response = train_test_model(client, "nonexistent_model", X, y)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]
    
    def test_train_model_json_invalid_data(self, client, clean_service):
        """Test training a model with invalid JSON data."""
        # Create a model
        model_id = "test_train_invalid"
        create_test_model(client, model_id)
        
        # Train with invalid data (empty features)
        response = client.post(
            f"/lead_scoring/models/{model_id}/train",
            json={
                "features": [],
                "target": [],
                "test_size": 0.2
            }
        )
        assert response.status_code == 400  # Bad request or internal server error
    
    def test_train_model_csv(self, client, clean_service):
        """Test training a model with CSV data."""
        # Create a model
        model_id = "test_train_csv"
        create_test_model(client, model_id)
        
        # Generate test data
        X, y = generate_test_data()
        df = X.copy()
        df['target'] = y
        
        # Convert to CSV
        csv_data = dataframe_to_csv(df)
        
        # Train the model
        response = client.post(
            f"/lead_scoring/models/{model_id}/train/csv",
            files={"features_file": ("data.csv", csv_data, "text/csv")},
            data={"target_column": "target", "test_size": 0.2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_id"] == model_id
        assert "metrics" in data
    
    def test_train_model_csv_missing_target(self, client, clean_service):
        """Test training a model with CSV missing the target column."""
        # Create a model
        model_id = "test_train_csv_missing"
        create_test_model(client, model_id)
        
        # Generate test data without target column
        X, _ = generate_test_data()
        
        # Convert to CSV
        csv_data = dataframe_to_csv(X)
        
        # Train the model
        response = client.post(
            f"/lead_scoring/models/{model_id}/train/csv",
            files={"features_file": ("data.csv", csv_data, "text/csv")},
            data={"target_column": "nonexistent_column", "test_size": 0.2}
        )
        assert response.status_code == 400
        assert "not found in CSV" in response.json()["detail"]
    
    def test_score_leads_json(self, client, clean_service):
        """Test scoring leads with JSON data."""
        # Create and train a model
        model_id = "test_score_json"
        X, _ = create_and_train_test_model(client, clean_service, model_id)
        
        # Score some leads
        features_to_score = X.head(3).to_dict(orient='records')
        response = client.post(
            f"/lead_scoring/models/{model_id}/score",
            json={"features": features_to_score}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert "scores" in data
        assert len(data["scores"]) == 3
        assert all(0 <= score <= 1 for score in data["scores"])
    
    def test_score_leads_json_nonexistent_model(self, client, clean_service):
        """Test scoring leads with a non-existent model."""
        X, _ = generate_test_data()
        features_to_score = X.head(3).to_dict(orient='records')
        response = client.post(
            "/lead_scoring/models/nonexistent_model/score",
            json={"features": features_to_score}
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]
    
    def test_score_leads_json_missing_features(self, client, clean_service):
        """Test scoring leads with JSON missing required features."""
        # Create and train a model
        model_id = "test_score_missing"
        X, _ = create_and_train_test_model(client, clean_service, model_id)
        
        # Remove a required feature
        features_to_score = X.head(3).drop(columns=['feature1']).to_dict(orient='records')
        response = client.post(
            f"/lead_scoring/models/{model_id}/score",
            json={"features": features_to_score}
        )
        assert response.status_code == 400
        assert "Missing features" in response.json()["detail"]
    
    def test_score_leads_csv(self, client, clean_service):
        """Test scoring leads with CSV data."""
        # Create and train a model
        model_id = "test_score_csv"
        X, _ = create_and_train_test_model(client, clean_service, model_id)
        
        # Convert to CSV
        features_to_score = X.head(3)
        csv_data = dataframe_to_csv(features_to_score)
        
        # Score the leads
        response = client.post(
            f"/lead_scoring/models/{model_id}/score/csv",
            files={"features_file": ("data.csv", csv_data, "text/csv")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert "scores" in data
        assert len(data["scores"]) == 3
    
    def test_explain_scores_json(self, client, clean_service):
        """Test explaining scores with JSON data."""
        # Create and train a model
        model_id = "test_explain_json"
        X, _ = create_and_train_test_model(client, clean_service, model_id)
        
        # Explain some lead scores
        features_to_explain = X.head(2).to_dict(orient='records')
        response = client.post(
            f"/lead_scoring/models/{model_id}/explain",
            json={"features": features_to_explain}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert "explanations" in data
        assert len(data["explanations"]) == 2
        
        # Check the structure of explanations
        explanation = data["explanations"][0]
        assert "base_value" in explanation
        assert "feature_impacts" in explanation
        assert len(explanation["feature_impacts"]) == len(X.columns)
    
    def test_explain_scores_csv(self, client, clean_service):
        """Test explaining scores with CSV data."""
        # Create and train a model
        model_id = "test_explain_csv"
        X, _ = create_and_train_test_model(client, clean_service, model_id)
        
        # Add id column for testing row identification
        features_to_explain = X.head(2).copy()
        features_to_explain['id'] = [1001, 1002]
        
        # Convert to CSV
        csv_data = dataframe_to_csv(features_to_explain)
        
        # Explain the lead scores
        response = client.post(
            f"/lead_scoring/models/{model_id}/explain/csv",
            files={"features_file": ("data.csv", csv_data, "text/csv")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert "explanations" in data
        assert len(data["explanations"]) == 2
        
        # Check that lead_id was added
        assert "lead_id" in data["explanations"][0]
        assert data["explanations"][0]["lead_id"] == 1001
        assert data["explanations"][1]["lead_id"] == 1002 