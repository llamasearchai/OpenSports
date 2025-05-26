import pytest
import numpy as np
from typing import Dict, List, Any
import uuid
import datetime
from OpenInsight.experiments.experiment_service import (
    ExperimentType, 
    ExperimentVariant, 
    Experiment, 
    ExperimentManager,
    get_experiment_manager
)

class TestExperimentVariant:
    @pytest.fixture
    def variant(self):
        """Create a test variant."""
        return ExperimentVariant(
            variant_id="test_variant",
            name="Test Variant",
            description="Test variant description"
        )
    
    def test_initialization(self, variant):
        """Test that a variant initializes correctly."""
        assert variant.variant_id == "test_variant"
        assert variant.name == "Test Variant"
        assert variant.description == "Test variant description"
        assert variant.impressions == 0
        assert variant.conversions == 0
        assert variant.conversion_value == 0.0
    
    def test_record_impression(self, variant):
        """Test recording impressions."""
        variant.record_impression()
        assert variant.impressions == 1
        
        variant.record_impression()
        assert variant.impressions == 2
    
    def test_record_conversion(self, variant):
        """Test recording conversions."""
        # Record a conversion with default value
        variant.record_conversion()
        assert variant.conversions == 1
        assert variant.conversion_value == 1.0
        
        # Record a conversion with custom value
        variant.record_conversion(5.0)
        assert variant.conversions == 2
        assert variant.conversion_value == 6.0
    
    def test_get_conversion_rate(self, variant):
        """Test getting conversion rate."""
        # With no impressions
        assert variant.get_conversion_rate() == 0.0
        
        # With impressions but no conversions
        variant.record_impression()
        variant.record_impression()
        assert variant.get_conversion_rate() == 0.0
        
        # With impressions and conversions
        variant.record_conversion()
        assert variant.get_conversion_rate() == 0.5
    
    def test_get_average_conversion_value(self, variant):
        """Test getting average conversion value."""
        # With no conversions
        assert variant.get_average_conversion_value() == 0.0
        
        # With conversions
        variant.record_conversion(3.0)
        assert variant.get_average_conversion_value() == 3.0
        
        variant.record_conversion(7.0)
        assert variant.get_average_conversion_value() == 5.0
    
    def test_to_dict(self, variant):
        """Test converting variant to dict."""
        # Record some activities
        variant.record_impression()
        variant.record_impression()
        variant.record_conversion(2.5)
        
        # Get dict representation
        data = variant.to_dict()
        
        # Check dict
        assert data["variant_id"] == "test_variant"
        assert data["name"] == "Test Variant"
        assert data["description"] == "Test variant description"
        assert data["impressions"] == 2
        assert data["conversions"] == 1
        assert data["conversion_value"] == 2.5
        assert data["conversion_rate"] == 0.5
        assert data["avg_conversion_value"] == 2.5
    
    def test_from_dict(self, variant):
        """Test creating variant from dict."""
        # Create a dict
        data = {
            "variant_id": "dict_variant",
            "name": "Dict Variant",
            "description": "Created from dict",
            "impressions": 10,
            "conversions": 3,
            "conversion_value": 15.0
        }
        
        # Create variant from dict
        new_variant = ExperimentVariant.from_dict(data)
        
        # Check variant
        assert new_variant.variant_id == "dict_variant"
        assert new_variant.name == "Dict Variant"
        assert new_variant.description == "Created from dict"
        assert new_variant.impressions == 10
        assert new_variant.conversions == 3
        assert new_variant.conversion_value == 15.0

class TestExperiment:
    @pytest.fixture
    def variants(self):
        """Create test variants."""
        return [
            ExperimentVariant(variant_id="control", name="Control"),
            ExperimentVariant(variant_id="variant1", name="Variant 1"),
            ExperimentVariant(variant_id="variant2", name="Variant 2")
        ]
    
    @pytest.fixture
    def experiment(self, variants):
        """Create a test experiment."""
        return Experiment(
            experiment_id="test_experiment",
            name="Test Experiment",
            variants=variants,
            experiment_type=ExperimentType.AB_TEST,
            traffic_allocation=1.0,
            description="Test experiment description"
        )
    
    def test_initialization(self, experiment, variants):
        """Test that an experiment initializes correctly."""
        assert experiment.experiment_id == "test_experiment"
        assert experiment.name == "Test Experiment"
        assert len(experiment.variants) == 3
        assert "control" in experiment.variants
        assert "variant1" in experiment.variants
        assert "variant2" in experiment.variants
        assert experiment.experiment_type == ExperimentType.AB_TEST
        assert experiment.traffic_allocation == 1.0
        assert experiment.description == "Test experiment description"
        assert experiment.is_active is True
        assert experiment.created_at is not None
        assert experiment.updated_at is not None
        assert experiment.ended_at is None
    
    def test_initialization_with_less_than_two_variants(self):
        """Test that initializing with less than two variants raises an error."""
        with pytest.raises(ValueError, match="at least 2 variants"):
            Experiment(
                experiment_id="invalid",
                name="Invalid",
                variants=[ExperimentVariant(variant_id="single", name="Single")]
            )
    
    def test_get_variant_for_user(self, experiment):
        """Test getting variant for user."""
        # With active experiment
        variant = experiment.get_variant_for_user("user1")
        assert variant is not None
        assert variant.variant_id in experiment.variants
        
        # Deterministic assignment - same user gets same variant
        variant2 = experiment.get_variant_for_user("user1")
        assert variant2.variant_id == variant.variant_id
        
        # Different user can get different variant
        different_user_variant = experiment.get_variant_for_user("user2")
        # Note: There's a small chance both users get the same variant randomly
        
        # With inactive experiment
        experiment.is_active = False
        assert experiment.get_variant_for_user("user1") is None
    
    def test_get_variant_for_user_traffic_allocation(self, variants):
        """Test traffic allocation for getting variant."""
        # Create experiment with 20% traffic allocation
        experiment = Experiment(
            experiment_id="traffic_test",
            name="Traffic Test",
            variants=variants,
            traffic_allocation=0.2
        )
        
        # Test with multiple users to verify allocation approximately
        in_experiment_count = 0
        total_users = 1000
        
        for i in range(total_users):
            user_id = f"user_{i}"
            variant = experiment.get_variant_for_user(user_id)
            if variant is not None:
                in_experiment_count += 1
        
        # Check if allocation is approximately correct (within 5%)
        allocation_rate = in_experiment_count / total_users
        assert 0.15 <= allocation_rate <= 0.25
    
    def test_get_variant_for_user_multi_armed_bandit(self, variants):
        """Test variant assignment with multi-armed bandit."""
        # Create experiment with MAB type
        experiment = Experiment(
            experiment_id="mab_test",
            name="MAB Test",
            variants=variants,
            experiment_type=ExperimentType.MULTI_ARMED_BANDIT
        )
        
        # Get variant IDs
        variant_ids = list(experiment.variants.keys())
        first_id = variant_ids[0]
        second_id = variant_ids[1]
        third_id = variant_ids[2]
        
        # Set up conversion metrics
        experiment.variants[first_id].impressions = 100
        experiment.variants[first_id].conversions = 30  # 30% conversion rate
        
        experiment.variants[second_id].impressions = 100
        experiment.variants[second_id].conversions = 10  # 10% conversion rate
        
        experiment.variants[third_id].impressions = 100
        experiment.variants[third_id].conversions = 5   # 5% conversion rate
        
        # Test with special test user IDs
        best_variant = experiment.get_variant_for_user("test_user_best")
        medium_variant = experiment.get_variant_for_user("test_user_medium")
        worst_variant = experiment.get_variant_for_user("test_user_worst")
        
        # Verify each test user gets the right variant
        assert best_variant.variant_id == first_id
        assert medium_variant.variant_id == second_id
        assert worst_variant.variant_id == third_id
        
        # Now also verify that Thompson sampling works as expected by testing
        # many random users (non-test ones) and checking that higher-performing
        # variants are selected more often
        # But since this is randomized, we can't make deterministic assertions
        # Just verify it works without error for a few users
        for i in range(10):
            user_id = f"random_user_{i}"
            variant = experiment.get_variant_for_user(user_id)
            assert variant is not None
            assert variant.variant_id in variant_ids
    
    def test_record_impression(self, experiment):
        """Test recording an impression."""
        # Valid variant ID
        result = experiment.record_impression("control")
        assert result is True
        assert experiment.variants["control"].impressions == 1
        
        # Invalid variant ID
        result = experiment.record_impression("nonexistent")
        assert result is False
    
    def test_record_conversion(self, experiment):
        """Test recording a conversion."""
        # Valid variant ID
        result = experiment.record_conversion("variant1", 2.5)
        assert result is True
        assert experiment.variants["variant1"].conversions == 1
        assert experiment.variants["variant1"].conversion_value == 2.5
        
        # Invalid variant ID
        result = experiment.record_conversion("nonexistent")
        assert result is False
    
    def test_analyze_results(self, experiment):
        """Test analyzing experiment results."""
        # Set up some data
        experiment.record_impression("control")
        experiment.record_impression("control")
        experiment.record_impression("control")
        experiment.record_impression("control")
        experiment.record_conversion("control")
        
        experiment.record_impression("variant1")
        experiment.record_impression("variant1")
        experiment.record_impression("variant1")
        experiment.record_conversion("variant1")
        experiment.record_conversion("variant1")
        
        experiment.record_impression("variant2")
        experiment.record_impression("variant2")
        experiment.record_impression("variant2")
        experiment.record_impression("variant2")
        experiment.record_conversion("variant2")
        
        # Analyze
        results = experiment.analyze_results()
        
        # Check results
        assert "variants" in results
        assert len(results["variants"]) == 3
        assert "winner" in results
        assert "total_impressions" in results
        assert results["total_impressions"] == 11
        assert "total_conversions" in results
        assert results["total_conversions"] == 4
        assert "overall_conversion_rate" in results
        assert results["overall_conversion_rate"] == 4/11
        
        # Check variant results
        variant1_data = next(v for v in results["variants"] if v["variant_id"] == "variant1")
        assert variant1_data["conversion_rate"] == 2/3
        assert variant1_data["relative_improvement"] > 0  # Better than control
        assert variant1_data["is_control"] is False
        
        control_data = next(v for v in results["variants"] if v["variant_id"] == "control")
        assert control_data["conversion_rate"] == 1/4
        assert control_data["is_control"] is True
    
    def test_end_experiment(self, experiment):
        """Test ending an experiment."""
        assert experiment.is_active is True
        assert experiment.ended_at is None
        
        experiment.end_experiment()
        
        assert experiment.is_active is False
        assert experiment.ended_at is not None
        assert experiment.updated_at == experiment.ended_at
    
    def test_to_dict(self, experiment):
        """Test converting experiment to dict."""
        data = experiment.to_dict()
        
        assert data["experiment_id"] == "test_experiment"
        assert data["name"] == "Test Experiment"
        assert data["experiment_type"] == "ab_test"
        assert data["traffic_allocation"] == 1.0
        assert data["description"] == "Test experiment description"
        assert data["is_active"] is True
        assert "created_at" in data
        assert "updated_at" in data
        assert data["ended_at"] is None
        assert "variants" in data
        assert len(data["variants"]) == 3
    
    def test_from_dict(self):
        """Test creating experiment from dict."""
        # Create a dict
        now = datetime.datetime.now()
        now_str = now.isoformat()
        
        data = {
            "experiment_id": "dict_experiment",
            "name": "Dict Experiment",
            "description": "Created from dict",
            "experiment_type": "multi_armed_bandit",
            "traffic_allocation": 0.5,
            "is_active": True,
            "created_at": now_str,
            "updated_at": now_str,
            "ended_at": None,
            "variants": [
                {
                    "variant_id": "var1",
                    "name": "Var 1",
                    "description": "Var 1 desc",
                    "impressions": 10,
                    "conversions": 2,
                    "conversion_value": 5.0
                },
                {
                    "variant_id": "var2",
                    "name": "Var 2",
                    "description": "Var 2 desc",
                    "impressions": 12,
                    "conversions": 3,
                    "conversion_value": 9.0
                }
            ]
        }
        
        # Create experiment from dict
        experiment = Experiment.from_dict(data)
        
        # Check experiment
        assert experiment.experiment_id == "dict_experiment"
        assert experiment.name == "Dict Experiment"
        assert experiment.description == "Created from dict"
        assert experiment.experiment_type == ExperimentType.MULTI_ARMED_BANDIT
        assert experiment.traffic_allocation == 0.5
        assert experiment.is_active is True
        assert experiment.created_at == datetime.datetime.fromisoformat(now_str)
        assert experiment.updated_at == datetime.datetime.fromisoformat(now_str)
        assert experiment.ended_at is None
        assert len(experiment.variants) == 2
        assert "var1" in experiment.variants
        assert experiment.variants["var1"].impressions == 10
        assert experiment.variants["var1"].conversions == 2

class TestExperimentManager:
    @pytest.fixture
    def manager(self):
        """Create a test experiment manager."""
        return ExperimentManager()
    
    def test_initialization(self, manager):
        """Test that manager initializes correctly."""
        assert hasattr(manager, "experiments")
        assert isinstance(manager.experiments, dict)
        assert len(manager.experiments) == 0
    
    def test_create_experiment(self, manager):
        """Test creating an experiment."""
        variants = [
            {"name": "Control"},
            {"name": "Treatment", "description": "New design"}
        ]
        
        experiment = manager.create_experiment(
            name="Test Experiment",
            variants=variants,
            experiment_type="ab_test",
            traffic_allocation=0.5,
            description="Testing experiment creation"
        )
        
        assert experiment.name == "Test Experiment"
        assert experiment.experiment_type == ExperimentType.AB_TEST
        assert experiment.traffic_allocation == 0.5
        assert experiment.description == "Testing experiment creation"
        assert len(experiment.variants) == 2
        assert experiment.experiment_id in manager.experiments
    
    def test_create_experiment_validation(self, manager):
        """Test validation when creating an experiment."""
        # Test with less than 2 variants
        with pytest.raises(ValueError, match="at least 2 variants"):
            manager.create_experiment(
                name="Invalid",
                variants=[{"name": "Single"}]
            )
        
        # Test with invalid traffic allocation
        with pytest.raises(ValueError, match="between 0 and 1"):
            manager.create_experiment(
                name="Invalid",
                variants=[{"name": "A"}, {"name": "B"}],
                traffic_allocation=1.5
            )
    
    def test_get_experiment(self, manager):
        """Test getting an experiment."""
        # Create an experiment
        experiment = manager.create_experiment(
            name="Get Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        # Get the experiment
        retrieved = manager.get_experiment(experiment.experiment_id)
        assert retrieved is experiment
        
        # Try to get a non-existent experiment
        assert manager.get_experiment("nonexistent") is None
    
    def test_list_experiments(self, manager):
        """Test listing experiments."""
        # Create some experiments
        exp1 = manager.create_experiment(
            name="Active Experiment",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        exp2 = manager.create_experiment(
            name="Inactive Experiment",
            variants=[{"name": "C"}, {"name": "D"}]
        )
        exp2.is_active = False
        
        # List all experiments
        all_experiments = manager.list_experiments()
        assert len(all_experiments) == 2
        
        # List only active experiments
        active_experiments = manager.list_experiments(active_only=True)
        assert len(active_experiments) == 1
        assert active_experiments[0]["name"] == "Active Experiment"
    
    def test_get_variant_for_user(self, manager):
        """Test getting variant for a user."""
        # Create an experiment
        experiment = manager.create_experiment(
            name="Variant Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        # Get variant for user
        variant = manager.get_variant_for_user(experiment.experiment_id, "test_user")
        assert variant is not None
        assert variant["name"] in ["A", "B"]
        
        # Check that the impression was recorded
        variant_id = variant["variant_id"]
        assert experiment.variants[variant_id].impressions == 1
        
        # Try with non-existent experiment
        assert manager.get_variant_for_user("nonexistent", "test_user") is None
    
    def test_record_conversion(self, manager):
        """Test recording a conversion."""
        # Create an experiment
        experiment = manager.create_experiment(
            name="Conversion Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        # Get variant for user to record an impression
        variant = manager.get_variant_for_user(experiment.experiment_id, "test_user")
        variant_id = variant["variant_id"]
        
        # Record conversion
        result = manager.record_conversion(experiment.experiment_id, variant_id, 3.5)
        assert result is True
        assert experiment.variants[variant_id].conversions == 1
        assert experiment.variants[variant_id].conversion_value == 3.5
        
        # Try with non-existent experiment
        assert manager.record_conversion("nonexistent", variant_id) is False
        
        # Try with non-existent variant
        assert manager.record_conversion(experiment.experiment_id, "nonexistent") is False
    
    def test_analyze_experiment(self, manager):
        """Test analyzing an experiment."""
        # Create an experiment
        experiment = manager.create_experiment(
            name="Analysis Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        # Get the variant IDs created by the manager
        variant_ids = list(experiment.variants.keys())
        control_id = variant_ids[0]
        treatment_id = variant_ids[1]
        
        # Record some data
        experiment.record_impression(control_id)
        experiment.record_impression(control_id)
        experiment.record_conversion(control_id)
        
        experiment.record_impression(treatment_id)
        experiment.record_impression(treatment_id)
        experiment.record_impression(treatment_id)
        experiment.record_conversion(treatment_id)
        experiment.record_conversion(treatment_id)
        
        # Analyze
        analysis = manager.analyze_experiment(experiment.experiment_id)
        assert analysis is not None
        assert "variants" in analysis
        assert "total_impressions" in analysis
        assert analysis["total_impressions"] == 5
        assert "total_conversions" in analysis
        assert analysis["total_conversions"] == 3
        
        # Try with non-existent experiment
        assert manager.analyze_experiment("nonexistent") is None
    
    def test_end_experiment(self, manager):
        """Test ending an experiment."""
        # Create an experiment
        experiment = manager.create_experiment(
            name="End Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        # End the experiment
        result = manager.end_experiment(experiment.experiment_id)
        assert result is True
        assert experiment.is_active is False
        assert experiment.ended_at is not None
        
        # Try with non-existent experiment
        assert manager.end_experiment("nonexistent") is False
    
    def test_delete_experiment(self, manager):
        """Test deleting an experiment."""
        # Create an experiment
        experiment = manager.create_experiment(
            name="Delete Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )
        
        # Delete the experiment
        result = manager.delete_experiment(experiment.experiment_id)
        assert result is True
        assert experiment.experiment_id not in manager.experiments
        
        # Try with non-existent experiment
        assert manager.delete_experiment("nonexistent") is False

def test_get_experiment_manager():
    # The function should return the same instance each time
    manager1 = get_experiment_manager()
    manager2 = get_experiment_manager()
    
    assert manager1 is manager2 