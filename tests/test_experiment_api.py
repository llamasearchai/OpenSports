import pytest
from fastapi.testclient import TestClient
import json
from OpenInsight.api.main import app
from OpenInsight.experiments.experiment_service import get_experiment_manager

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the FastAPI app."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def clean_manager():
    """Provides a clean experiment manager for tests."""
    manager = get_experiment_manager()
    # Store existing experiment IDs to avoid deleting them
    existing_experiments = set(manager.experiments.keys())
    
    yield manager
    
    # Clean up experiments created during tests
    for experiment_id in list(manager.experiments.keys()):
        if experiment_id not in existing_experiments:
            manager.delete_experiment(experiment_id)

def create_test_experiment(client, name="Test Experiment"):
    """Helper to create a test experiment."""
    response = client.post(
        "/experiments/",
        json={
            "name": name,
            "variants": [
                {"name": "Control"},
                {"name": "Variant A", "description": "Test variant"}
            ],
            "experiment_type": "ab_test",
            "traffic_allocation": 0.8,
            "description": "Test experiment description"
        }
    )
    return response

class TestExperimentAPI:
    def test_create_experiment(self, client, clean_manager):
        """Test creating an experiment."""
        response = create_test_experiment(client)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Experiment"
        assert data["experiment_type"] == "ab_test"
        assert data["traffic_allocation"] == 0.8
        assert data["description"] == "Test experiment description"
        assert data["is_active"] is True
        assert len(data["variants"]) == 2
        assert data["variants"][0]["name"] == "Control"
        assert data["variants"][1]["name"] == "Variant A"
        assert data["variants"][1]["description"] == "Test variant"
        
        # Store experiment_id for later tests
        return data["experiment_id"]
    
    def test_create_experiment_invalid(self, client, clean_manager):
        """Test creating an experiment with invalid data."""
        # Test with less than 2 variants
        response = client.post(
            "/experiments/",
            json={
                "name": "Invalid Experiment",
                "variants": [
                    {"name": "Single Variant"}
                ],
                "experiment_type": "ab_test"
            }
        )
        assert response.status_code == 400
        assert "at least 2 variants" in response.json()["detail"]
        
        # Test with invalid traffic allocation
        response = client.post(
            "/experiments/",
            json={
                "name": "Invalid Experiment",
                "variants": [
                    {"name": "Control"},
                    {"name": "Treatment"}
                ],
                "experiment_type": "ab_test",
                "traffic_allocation": 1.5
            }
        )
        assert response.status_code == 400
        assert "between 0 and 1" in response.json()["detail"]
    
    def test_list_experiments(self, client, clean_manager):
        """Test listing experiments."""
        # Create a couple of experiments
        exp1_id = self.test_create_experiment(client, "List Test 1")
        exp2_id = self.test_create_experiment(client, "List Test 2")
        
        # End one experiment
        client.post(f"/experiments/{exp2_id}/end")
        
        # List all experiments
        response = client.get("/experiments/")
        assert response.status_code == 200
        data = response.json()
        
        # Find our test experiments in the list
        test_experiments = [e for e in data if e["experiment_id"] in [exp1_id, exp2_id]]
        assert len(test_experiments) == 2
        
        # List only active experiments
        response = client.get("/experiments/?active_only=true")
        assert response.status_code == 200
        data = response.json()
        
        # Find our active test experiment
        active_experiments = [e for e in data if e["experiment_id"] in [exp1_id, exp2_id]]
        assert len(active_experiments) == 1
        assert active_experiments[0]["experiment_id"] == exp1_id
    
    def test_get_experiment(self, client, clean_manager):
        """Test getting a specific experiment."""
        # Create an experiment
        exp_id = self.test_create_experiment(client, "Get Test")
        
        # Get the experiment
        response = client.get(f"/experiments/{exp_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == exp_id
        assert data["name"] == "Get Test"
        
        # Try to get a non-existent experiment
        response = client.get("/experiments/nonexistent")
        assert response.status_code == 404
    
    def test_assign_variant(self, client, clean_manager):
        """Test assigning a variant to a user."""
        # Create an experiment
        exp_id = self.test_create_experiment(client, "Assign Test")
        
        # Assign a variant to a user
        response = client.post(
            f"/experiments/{exp_id}/assign",
            json={"user_id": "test_user_1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "variant_id" in data
        assert "name" in data
        
        # Same user should get same variant
        response = client.post(
            f"/experiments/{exp_id}/assign",
            json={"user_id": "test_user_1"}
        )
        assert response.status_code == 200
        assert response.json()["variant_id"] == data["variant_id"]
        
        # Check impressions count increased
        response = client.get(f"/experiments/{exp_id}")
        experiment_data = response.json()
        variant_id = data["variant_id"]
        variant = next(v for v in experiment_data["variants"] if v["variant_id"] == variant_id)
        assert variant["impressions"] >= 2  # At least 2 from our two calls
        
        # Try with non-existent experiment
        response = client.post(
            "/experiments/nonexistent/assign",
            json={"user_id": "test_user_1"}
        )
        assert response.status_code == 404
    
    def test_record_conversion(self, client, clean_manager):
        """Test recording a conversion."""
        # Create an experiment and get a variant
        exp_id = self.test_create_experiment(client, "Conversion Test")
        
        assign_response = client.post(
            f"/experiments/{exp_id}/assign",
            json={"user_id": "conversion_test_user"}
        )
        variant_id = assign_response.json()["variant_id"]
        
        # Record a conversion
        response = client.post(
            f"/experiments/{exp_id}/convert",
            json={"variant_id": variant_id, "value": 2.5}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Check that conversion was recorded
        response = client.get(f"/experiments/{exp_id}")
        experiment_data = response.json()
        variant = next(v for v in experiment_data["variants"] if v["variant_id"] == variant_id)
        assert variant["conversions"] == 1
        assert variant["conversion_value"] == 2.5
        
        # Try with non-existent experiment
        response = client.post(
            "/experiments/nonexistent/convert",
            json={"variant_id": variant_id}
        )
        assert response.status_code == 404
        
        # Try with non-existent variant
        response = client.post(
            f"/experiments/{exp_id}/convert",
            json={"variant_id": "nonexistent"}
        )
        assert response.status_code == 404
    
    def test_analyze_experiment(self, client, clean_manager):
        """Test analyzing an experiment."""
        # Create an experiment
        exp_id = self.test_create_experiment(client, "Analysis Test")
        
        # Get variants
        response = client.get(f"/experiments/{exp_id}")
        experiment_data = response.json()
        variant_ids = [v["variant_id"] for v in experiment_data["variants"]]
        control_id = variant_ids[0]
        treatment_id = variant_ids[1]
        
        # Record some impressions and conversions
        # For control: 1/5 conversion rate
        for i in range(5):
            client.post(
                f"/experiments/{exp_id}/assign",
                json={"user_id": f"control_user_{i}"}
            )
        
        client.post(
            f"/experiments/{exp_id}/convert",
            json={"variant_id": control_id}
        )
        
        # For treatment: 3/5 conversion rate
        for i in range(5):
            client.post(
                f"/experiments/{exp_id}/assign",
                json={"user_id": f"treatment_user_{i}"}
            )
        
        for i in range(3):
            client.post(
                f"/experiments/{exp_id}/convert",
                json={"variant_id": treatment_id}
            )
        
        # Analyze the experiment
        response = client.get(f"/experiments/{exp_id}/analyze")
        assert response.status_code == 200
        analysis = response.json()
        
        assert "variants" in analysis
        assert len(analysis["variants"]) == 2
        assert "total_impressions" in analysis
        assert analysis["total_impressions"] >= 10  # At least 10 from our setup
        assert "total_conversions" in analysis
        assert analysis["total_conversions"] >= 4  # At least 4 from our setup
        
        # Try with non-existent experiment
        response = client.get("/experiments/nonexistent/analyze")
        assert response.status_code == 404
    
    def test_end_experiment(self, client, clean_manager):
        """Test ending an experiment."""
        # Create an experiment
        exp_id = self.test_create_experiment(client, "End Test")
        
        # End the experiment
        response = client.post(f"/experiments/{exp_id}/end")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Verify the experiment is now inactive
        response = client.get(f"/experiments/{exp_id}")
        assert response.status_code == 200
        assert response.json()["is_active"] is False
        
        # Try with non-existent experiment
        response = client.post("/experiments/nonexistent/end")
        assert response.status_code == 404
    
    def test_delete_experiment(self, client, clean_manager):
        """Test deleting an experiment."""
        # Create an experiment
        exp_id = self.test_create_experiment(client, "Delete Test")
        
        # Delete the experiment
        response = client.delete(f"/experiments/{exp_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Verify the experiment is gone
        response = client.get(f"/experiments/{exp_id}")
        assert response.status_code == 404
        
        # Try with non-existent experiment
        response = client.delete("/experiments/nonexistent")
        assert response.status_code == 404 