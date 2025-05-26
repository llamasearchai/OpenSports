import pytest
from fastapi.testclient import TestClient
# Ensure the app is imported from the correct location
# This might require setting PYTHONPATH or installing the package in editable mode
from OpenInsight.api.main import app 
import datetime
from OpenInsight.experiments.experiment_service import get_experiment_manager

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the FastAPI app."""
    with TestClient(app) as c:
        # Ensure startup events are run if your app relies on them for setup
        # TestClient(app) typically handles this, but explicit call if needed:
        # app.router.startup()
        yield c
        # app.router.shutdown() # if cleanup is needed

def test_read_root(client: TestClient):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the OpenInsight API"}

def test_health_check(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

class TestAudienceSegmentationEndpoint:
    def test_create_audience_segment_valid_request(self, client: TestClient):
        """Test successful audience segmentation."""
        payload = {
            "feature_vectors": [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [5.0, 6.0, 7.0],
                [5.1, 6.1, 7.1],
                [9.0, 0.0, 1.0]
            ],
            "n_clusters": 3,
            "random_state": 42
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "segmentation_results" in data
        results = data["segmentation_results"]
        assert "labels" in results
        assert "cluster_centers" in results
        assert "inertia" in results # Assuming segmenter still returns this
        assert len(results["labels"]) == len(payload["feature_vectors"])
        assert len(results["cluster_centers"]) == payload["n_clusters"]

    def test_create_audience_segment_empty_feature_vectors(self, client: TestClient):
        """Test segmentation with empty feature_vectors."""
        payload = {
            "feature_vectors": [],
            "n_clusters": 2
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 400 # Segmenter error
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Input feature_vectors cannot be empty." # Error from segmenter

    def test_create_audience_segment_n_clusters_zero(self, client: TestClient):
        """Test segmentation with n_clusters = 0 (Pydantic validation)."""
        payload = {
            "feature_vectors": [[1.0, 2.0], [3.0, 4.0]],
            "n_clusters": 0
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert "detail" in data
        assert any("n_clusters" in err["loc"] and "Input should be greater than 0" in err["msg"] for err in data["detail"])

    def test_create_audience_segment_n_samples_lt_n_clusters(self, client: TestClient):
        """Test segmentation where number of samples is less than n_clusters."""
        payload = {
            "feature_vectors": [[1.0, 2.0]],
            "n_clusters": 2
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 400 # Segmenter error
        data = response.json()
        assert "detail" in data
        # Message from segmenter's KMeans: "Number of clusters 2 exceeds number of samples 1."
        assert "exceeds number of samples" in data["detail"]
        assert "Number of clusters 2" in data["detail"]
        assert "number of samples 1" in data["detail"]

    def test_create_audience_segment_missing_feature_vectors(self, client: TestClient):
        """Test request missing feature_vectors field."""
        payload = {
            "n_clusters": 2
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert "detail" in data
        assert any("feature_vectors" in err["loc"] and "Field required" in err["type"] for err in data["detail"]) 

    def test_create_audience_segment_missing_n_clusters(self, client: TestClient):
        """Test request missing n_clusters field."""
        payload = {
            "feature_vectors": [[1.0, 2.0]]
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert "detail" in data
        assert any("n_clusters" in err["loc"] and "Field required" in err["type"] for err in data["detail"])

    def test_create_audience_segment_inconsistent_vector_lengths(self, client: TestClient):
        """Test with feature_vectors that have inconsistent inner list lengths."""
        payload = {
            "feature_vectors": [[1.0, 2.0], [3.0]], 
            "n_clusters": 1
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 400 # Error from segmenter (numpy error caught)
        data = response.json()
        assert "detail" in data
        assert "setting an array element with a sequence" in data["detail"]

    def test_create_audience_segment_feature_vectors_not_list_of_lists(self, client: TestClient):
        """Test with feature_vectors that is not a list of lists (Pydantic should catch this)."""
        payload = {
            "feature_vectors": [1.0, 2.0, 3.0], # Not a list of lists
            "n_clusters": 1
        }
        response = client.post("/audience_segmentation", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert "detail" in data
        # Pydantic v2 error message for list of lists of floats
        assert any("feature_vectors" in err["loc"] and "Input should be a valid list" in item_err["msg"] for err in data["detail"] for item_err in err.get("ctx", {}).get("errors", []))

class TestForecastEndpoint:
    @pytest.fixture
    def sample_historical_data_api(self):
        return [
            {"ds": "2023-01-01", "y": 10.0},
            {"ds": "2023-01-02", "y": 12.5},
            {"ds": "2023-01-03", "y": 11.0},
            {"ds": "2023-01-04", "y": 15.0},
            {"ds": "2023-01-05", "y": 18.5},
            {"ds": "2023-01-06T12:00:00", "y": 16.0}, # Mixed date and datetime
            {"ds": "2023-01-07", "y": 20.0},
        ]

    def test_get_forecast_valid_request(self, client: TestClient, sample_historical_data_api):
        """Test successful forecast generation."""
        model_id = "test_sales_model"
        payload = {
            "historical_data": sample_historical_data_api,
            "periods": 5,
            "freq": "D",
            "prophet_kwargs": {"yearly_seasonality": False, "weekly_seasonality": False, "daily_seasonality": False} # Simplify for testing
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_id"] == model_id
        assert "forecast_results" in data
        results = data["forecast_results"]
        assert "forecast" in results
        assert "model_params" in results
        assert len(results["forecast"]) == payload["periods"]
        for item in results["forecast"]:
            assert "ds" in item
            assert "yhat" in item
            # Verify ds is a string and can be parsed as datetime
            datetime.datetime.fromisoformat(item["ds"])

    def test_get_forecast_missing_historical_data(self, client: TestClient):
        model_id = "test_model"
        payload = {
            # "historical_data": [], # Missing
            "periods": 5
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert any("historical_data" in err["loc"] and "field required" in err["msg"].lower() for err in data["detail"]) 

    def test_get_forecast_empty_historical_data(self, client: TestClient):
        model_id = "test_model"
        payload = {
            "historical_data": [], 
            "periods": 5
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation: min_items=2
        data = response.json()
        assert any("historical_data" in err["loc"] and "ensure this list has at least 2 items" in err["msg"].lower() for err in data["detail"]) 

    def test_get_forecast_insufficient_historical_data(self, client: TestClient):
        model_id = "test_model"
        payload = {
            "historical_data": [{"ds": "2023-01-01", "y": 10.0}], 
            "periods": 5
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation: min_items=2
        data = response.json()
        assert any("historical_data" in err["loc"] and "ensure this list has at least 2 items" in err["msg"].lower() for err in data["detail"]) 
        
    def test_get_forecast_missing_periods(self, client: TestClient, sample_historical_data_api):
        model_id = "test_model"
        payload = {
            "historical_data": sample_historical_data_api,
            # "periods": 5 # Missing
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert any("periods" in err["loc"] and "field required" in err["msg"].lower() for err in data["detail"]) 

    def test_get_forecast_invalid_periods(self, client: TestClient, sample_historical_data_api):
        model_id = "test_model"
        payload = {
            "historical_data": sample_historical_data_api,
            "periods": 0 # Invalid, must be > 0
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation error
        data = response.json()
        assert any("periods" in err["loc"] and "ensure this value is greater than 0" in err["msg"].lower() for err in data["detail"]) 

    def test_get_forecast_invalid_ds_format_in_historical_data(self, client: TestClient):
        model_id = "test_model"
        payload = {
            "historical_data": [
                {"ds": "2023-01-01", "y": 10.0},
                {"ds": "invalid-date-format", "y": 12.0}
            ],
            "periods": 5
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation error for ds format
        data = response.json()
        assert any("historical_data" in err["loc"] and "ds must be a valid ISO 8601 date or datetime string" in err_item["msg"] for err_item in data["detail"])

    def test_get_forecast_non_numeric_y_in_historical_data(self, client: TestClient):
        model_id = "test_model"
        payload = {
            "historical_data": [
                {"ds": "2023-01-01", "y": 10.0},
                {"ds": "2023-01-02", "y": "not-a-float"} # type: ignore
            ],
            "periods": 5
        }
        response = client.post(f"/predictive_model/{model_id}/forecast", json=payload)
        assert response.status_code == 422 # Pydantic validation error for y type
        data = response.json()
        assert any("historical_data" in err["loc"] and "value is not a valid float" in err_item["msg"].lower() for err_item in data["detail"])

class TestExperimentsEndpoint:
    @pytest.fixture
    def clean_experiment_manager(self):
        # Get the experiment manager and clear any existing experiments for clean testing
        manager = get_experiment_manager()
        # Store experiment IDs to delete after the test
        experiment_ids = list(manager.experiments.keys())
        
        yield manager
        
        # Clean up experiments created during the test
        for exp_id in list(manager.experiments.keys()):
            if exp_id not in experiment_ids:
                try:
                    manager.delete_experiment(exp_id)
                except ValueError:
                    pass

    def test_create_experiment(self, client: TestClient, clean_experiment_manager):
        payload = {
            "name": "Test A/B Experiment",
            "variants": [
                {"name": "control", "weight": 1.0, "config": {"feature_flag": False}},
                {"name": "treatment", "weight": 1.0, "config": {"feature_flag": True}}
            ],
            "experiment_type": "a_b_test",
            "description": "Testing the new feature"
        }
        
        response = client.post("/experiments", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Test A/B Experiment"
        assert data["type"] == "a_b_test"
        assert data["description"] == "Testing the new feature"
        assert data["is_active"] == True
        assert len(data["variants"]) == 2
        
        # Save experiment_id for future tests
        experiment_id = data["experiment_id"]
        
        # Test Get Experiment
        response = client.get(f"/experiments/{experiment_id}")
        assert response.status_code == 200
        assert response.json()["experiment_id"] == experiment_id
        
        return experiment_id
        
    def test_create_experiment_invalid_variants(self, client: TestClient):
        # Test with only one variant (minimum is 2)
        payload = {
            "name": "Invalid Experiment",
            "variants": [
                {"name": "control", "weight": 1.0}
            ]
        }
        
        response = client.post("/experiments", json=payload)
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert any("variants" in err["loc"] and "ensure this list has at least 2 items" in err["msg"].lower() for err in data["detail"])
        
    def test_create_experiment_invalid_type(self, client: TestClient):
        payload = {
            "name": "Invalid Type Experiment",
            "variants": [
                {"name": "control", "weight": 1.0},
                {"name": "treatment", "weight": 1.0}
            ],
            "experiment_type": "invalid_type"
        }
        
        response = client.post("/experiments", json=payload)
        assert response.status_code == 400
        assert "Invalid experiment type" in response.json()["detail"]
        
    def test_list_experiments(self, client: TestClient, clean_experiment_manager):
        # Create two experiments
        for i in range(2):
            payload = {
                "name": f"List Test Experiment {i}",
                "variants": [
                    {"name": "control", "weight": 1.0},
                    {"name": "treatment", "weight": 1.0}
                ],
                "experiment_id": f"list_test_{i}"
            }
            client.post("/experiments", json=payload)
        
        # End one experiment
        client.post("/experiments/list_test_1/end")
        
        # List all experiments
        response = client.get("/experiments")
        assert response.status_code == 200
        all_experiments = response.json()
        assert len(all_experiments) >= 2  # May have more from other tests
        
        # Check that our test experiments are in the list
        exp_ids = [e["experiment_id"] for e in all_experiments]
        assert "list_test_0" in exp_ids
        assert "list_test_1" in exp_ids
        
        # List only active experiments
        response = client.get("/experiments?active_only=true")
        assert response.status_code == 200
        active_experiments = response.json()
        
        # Check that only active experiments are in the list
        active_exp_ids = [e["experiment_id"] for e in active_experiments]
        assert "list_test_0" in active_exp_ids
        assert "list_test_1" not in active_exp_ids
        
    def test_experiment_assignment_and_results(self, client: TestClient, clean_experiment_manager):
        # Create an experiment
        create_payload = {
            "name": "Assignment Test",
            "variants": [
                {"name": "control", "weight": 1.0, "config": {"button_color": "blue"}},
                {"name": "treatment", "weight": 1.0, "config": {"button_color": "green"}}
            ],
            "experiment_id": "assignment_test"
        }
        
        client.post("/experiments", json=create_payload)
        
        # Assign a variant to a user
        assign_payload = {"user_id": "test_user_123"}
        response = client.post("/experiments/assignment_test/assign", json=assign_payload)
        
        assert response.status_code == 200
        assignment = response.json()
        assert assignment["experiment_id"] == "assignment_test"
        assert assignment["variant_name"] in ["control", "treatment"]
        assert "button_color" in assignment["variant_config"]
        
        # Record a conversion result
        variant_name = assignment["variant_name"]
        result_payload = {
            "experiment_id": "assignment_test",
            "variant_name": variant_name,
            "success": True
        }
        
        response = client.post("/experiments/assignment_test/result", json=result_payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Get experiment stats to verify the result was recorded
        response = client.get("/experiments/assignment_test")
        assert response.status_code == 200
        stats = response.json()
        
        # Find the variant we just recorded a success for
        variant = next((v for v in stats["variants"] if v["name"] == variant_name), None)
        assert variant is not None
        assert variant["successes"] == 1
        
    def test_get_nonexistent_experiment(self, client: TestClient):
        response = client.get("/experiments/nonexistent_exp_id")
        assert response.status_code == 404
        
    def test_end_and_delete_experiment(self, client: TestClient, clean_experiment_manager):
        # Create an experiment
        create_payload = {
            "name": "End and Delete Test",
            "variants": [
                {"name": "control", "weight": 1.0},
                {"name": "treatment", "weight": 1.0}
            ],
            "experiment_id": "end_delete_test"
        }
        
        client.post("/experiments", json=create_payload)
        
        # End the experiment
        response = client.post("/experiments/end_delete_test/end")
        assert response.status_code == 200
        
        # Verify it's marked as inactive
        response = client.get("/experiments/end_delete_test")
        assert response.status_code == 200
        stats = response.json()
        assert stats["is_active"] == False
        assert stats["end_date"] is not None
        
        # Delete the experiment
        response = client.delete("/experiments/end_delete_test")
        assert response.status_code == 200
        
        # Verify it's deleted
        response = client.get("/experiments/end_delete_test")
        assert response.status_code == 404
        
    def test_get_winner(self, client: TestClient, clean_experiment_manager):
        # This test can't easily simulate enough data to get a winner
        # since that would require a lot of API calls, but we can test the endpoint exists
        
        # Create an experiment
        create_payload = {
            "name": "Winner Test",
            "variants": [
                {"name": "control", "weight": 1.0},
                {"name": "treatment", "weight": 1.0}
            ],
            "experiment_id": "winner_test"
        }
        
        client.post("/experiments", json=create_payload)
        
        # Test getting winner (should be None with no data)
        response = client.get("/experiments/winner_test/winner")
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "winner_test"
        assert data["winner"] is None  # No data, so no winner

# Example test for another endpoint (if it existed and was more complex)
# def test_get_forecast_placeholder(client: TestClient):
#     response = client.get("/predictive_model/model123/forecast?horizon_days=10")
#     assert response.status_code == 200
#     data = response.json()
#     assert data["model_id"] == "model123"
#     assert data["horizon_days"] == 10
#     assert "forecast_placeholder" in data 