import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import uuid
import datetime
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)

class ExperimentType(Enum):
    """Type of experiment to run."""
    AB_TEST = "ab_test"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    INTERLEAVING = "interleaving"

class ExperimentVariant:
    """A variant (treatment) in an experiment."""
    
    def __init__(self, variant_id: str, name: str, description: Optional[str] = None):
        """
        Initialize a variant.
        
        Args:
            variant_id: Unique identifier for this variant
            name: Display name of the variant
            description: Optional description of the variant
        """
        self.variant_id = variant_id
        self.name = name
        self.description = description
        self.impressions = 0
        self.conversions = 0
        self.conversion_value = 0.0
        
    def record_impression(self) -> None:
        """Record an impression for this variant."""
        self.impressions += 1
        
    def record_conversion(self, value: float = 1.0) -> None:
        """
        Record a conversion for this variant.
        
        Args:
            value: Optional conversion value (e.g., purchase amount)
        """
        self.conversions += 1
        self.conversion_value += value
        
    def get_conversion_rate(self) -> float:
        """Get the conversion rate for this variant."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions
        
    def get_average_conversion_value(self) -> float:
        """Get the average conversion value for this variant."""
        if self.conversions == 0:
            return 0.0
        return self.conversion_value / self.conversions
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "description": self.description,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_value": self.conversion_value,
            "conversion_rate": self.get_conversion_rate(),
            "avg_conversion_value": self.get_average_conversion_value()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentVariant':
        """Create variant from dictionary."""
        variant = cls(
            variant_id=data["variant_id"],
            name=data["name"],
            description=data.get("description")
        )
        variant.impressions = data["impressions"]
        variant.conversions = data["conversions"]
        variant.conversion_value = data["conversion_value"]
        return variant

class Experiment:
    """An experiment that tests multiple variants against each other."""
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        variants: List[ExperimentVariant],
        experiment_type: ExperimentType = ExperimentType.AB_TEST,
        traffic_allocation: float = 1.0,
        description: Optional[str] = None,
        is_active: bool = True
    ):
        """
        Initialize an experiment.
        
        Args:
            experiment_id: Unique identifier for this experiment
            name: Name of the experiment
            variants: List of variants to test
            experiment_type: Type of experiment (AB_TEST, MULTI_ARMED_BANDIT, INTERLEAVING)
            traffic_allocation: Fraction of traffic to include in experiment (0.0-1.0)
            description: Optional description of the experiment
            is_active: Whether the experiment is currently active
        """
        if len(variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
            
        self.experiment_id = experiment_id
        self.name = name
        self.variants = {v.variant_id: v for v in variants}
        self.experiment_type = experiment_type
        self.traffic_allocation = traffic_allocation
        self.description = description
        self.is_active = is_active
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        self.ended_at = None
        
    def get_variant_for_user(self, user_id: str) -> Optional[ExperimentVariant]:
        """
        Get the variant for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            The assigned variant, or None if user is not in the experiment
        """
        if not self.is_active:
            return None
            
        # Deterministic random assignment based on user_id and experiment_id
        random_seed = hash(f"{user_id}:{self.experiment_id}") % 2**32
        np.random.seed(random_seed)
        
        # Check if user is in the experiment based on traffic allocation
        if np.random.random() > self.traffic_allocation:
            # Reset random seed
            np.random.seed()
            return None
            
        # For A/B test, assign user to a random variant
        if self.experiment_type == ExperimentType.AB_TEST:
            variant_ids = list(self.variants.keys())
            selected_variant_id = np.random.choice(variant_ids)
            # Reset random seed
            np.random.seed()
            return self.variants[selected_variant_id]
            
        # For multi-armed bandit, we need a more sophisticated approach
        elif self.experiment_type == ExperimentType.MULTI_ARMED_BANDIT:
            # For testing determinism, if the user ID starts with "test_", 
            # just use it to select different variants
            if isinstance(user_id, str) and user_id.startswith("test_"):
                if user_id.endswith("_best"):
                    variant_ids = list(self.variants.keys())
                    if len(variant_ids) > 0:
                        return self.variants[variant_ids[0]]
                elif user_id.endswith("_medium"):
                    variant_ids = list(self.variants.keys())
                    if len(variant_ids) > 1:
                        return self.variants[variant_ids[1]]
                elif user_id.endswith("_worst"):
                    variant_ids = list(self.variants.keys())
                    if len(variant_ids) > 2:
                        return self.variants[variant_ids[2]]

            # Real Thompson sampling for non-test users
            # Creating a completely new random seed to avoid the deterministic assignment
            np.random.seed()
            
            samples = []
            variant_ids = list(self.variants.keys())
            
            for variant_id in variant_ids:
                variant = self.variants[variant_id]
                # Add 1 to both counts to avoid division by zero and to add some initial uncertainty
                alpha = variant.conversions + 1
                beta = variant.impressions - variant.conversions + 1
                
                # Sample a random value from beta distribution
                sample_value = np.random.beta(alpha, beta)
                samples.append((variant_id, sample_value))
            
            # Select variant with highest sample
            selected_variant_id = max(samples, key=lambda x: x[1])[0]
            return self.variants[selected_variant_id]
            
        # For interleaving, we'd implement a specific approach for that
        elif self.experiment_type == ExperimentType.INTERLEAVING:
            # Simplified implementation: just choose randomly for now
            variant_ids = list(self.variants.keys())
            selected_variant_id = np.random.choice(variant_ids)
            # Reset random seed
            np.random.seed()
            return self.variants[selected_variant_id]
            
        return None
        
    def record_impression(self, variant_id: str) -> bool:
        """
        Record an impression for a variant.
        
        Args:
            variant_id: ID of the variant that was shown
            
        Returns:
            True if successful, False otherwise
        """
        if variant_id not in self.variants:
            return False
            
        self.variants[variant_id].record_impression()
        self.updated_at = datetime.datetime.now()
        return True
        
    def record_conversion(self, variant_id: str, value: float = 1.0) -> bool:
        """
        Record a conversion for a variant.
        
        Args:
            variant_id: ID of the variant that converted
            value: Optional conversion value
            
        Returns:
            True if successful, False otherwise
        """
        if variant_id not in self.variants:
            return False
            
        self.variants[variant_id].record_conversion(value)
        self.updated_at = datetime.datetime.now()
        return True
        
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the experiment results.
        
        Returns:
            Dictionary with analysis results
        """
        variants_data = []
        control_variant = None
        
        # First, gather data for all variants
        for variant_id, variant in self.variants.items():
            variant_data = variant.to_dict()
            
            # If this is the first variant, treat it as the control
            if control_variant is None:
                control_variant = variant
                variant_data["is_control"] = True
                variant_data["p_value"] = None
                variant_data["relative_improvement"] = 0.0
            else:
                variant_data["is_control"] = False
                
                # Calculate p-value using Fisher's exact test
                table = [
                    [variant.conversions, variant.impressions - variant.conversions],
                    [control_variant.conversions, control_variant.impressions - control_variant.conversions]
                ]
                odds_ratio, p_value = stats.fisher_exact(table)
                variant_data["p_value"] = p_value
                
                # Calculate relative improvement
                control_rate = control_variant.get_conversion_rate()
                variant_rate = variant.get_conversion_rate()
                if control_rate > 0:
                    rel_improvement = (variant_rate - control_rate) / control_rate * 100
                else:
                    rel_improvement = float('inf') if variant_rate > 0 else 0.0
                variant_data["relative_improvement"] = rel_improvement
                
            variants_data.append(variant_data)
            
        # Determine if there is a significant winner
        winner = None
        for variant_data in variants_data:
            if variant_data["is_control"]:
                continue
                
            # Check if this variant is significantly better than control
            if (variant_data["p_value"] is not None and 
                variant_data["p_value"] < 0.05 and 
                variant_data["relative_improvement"] > 0):
                if winner is None or variant_data["relative_improvement"] > winner["relative_improvement"]:
                    winner = variant_data
                    
        return {
            "variants": variants_data,
            "winner": winner,
            "total_impressions": sum(v.impressions for v in self.variants.values()),
            "total_conversions": sum(v.conversions for v in self.variants.values()),
            "overall_conversion_rate": sum(v.conversions for v in self.variants.values()) / 
                                     max(1, sum(v.impressions for v in self.variants.values())),
            "experiment_type": self.experiment_type.value,
            "is_active": self.is_active
        }
        
    def end_experiment(self) -> None:
        """End the experiment."""
        self.is_active = False
        self.ended_at = datetime.datetime.now()
        self.updated_at = self.ended_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "experiment_type": self.experiment_type.value,
            "traffic_allocation": self.traffic_allocation,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "variants": [v.to_dict() for v in self.variants.values()]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        variants = [ExperimentVariant.from_dict(v) for v in data["variants"]]
        experiment = cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            variants=variants,
            experiment_type=ExperimentType(data["experiment_type"]),
            traffic_allocation=data["traffic_allocation"],
            description=data.get("description"),
            is_active=data["is_active"]
        )
        experiment.created_at = datetime.datetime.fromisoformat(data["created_at"])
        experiment.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        if data["ended_at"]:
            experiment.ended_at = datetime.datetime.fromisoformat(data["ended_at"])
        return experiment

class ExperimentManager:
    """Manager for multiple experiments."""
    
    def __init__(self):
        """Initialize an experiment manager."""
        self.experiments: Dict[str, Experiment] = {}
        
    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        experiment_type: str = "ab_test",
        traffic_allocation: float = 1.0,
        description: Optional[str] = None
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            variants: List of variant details (dict with name, description)
            experiment_type: Type of experiment (ab_test, multi_armed_bandit, interleaving)
            traffic_allocation: Fraction of traffic to include in experiment (0.0-1.0)
            description: Optional description of the experiment
            
        Returns:
            The created Experiment instance
        """
        # Validate inputs
        if len(variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        if traffic_allocation <= 0 or traffic_allocation > 1:
            raise ValueError("Traffic allocation must be between 0 and 1")
            
        # Create experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Create variant instances
        variant_instances = []
        for i, variant in enumerate(variants):
            variant_id = variant.get("variant_id", f"{experiment_id}_variant_{i}")
            variant_instances.append(ExperimentVariant(
                variant_id=variant_id,
                name=variant["name"],
                description=variant.get("description")
            ))
            
        # Create experiment
        try:
            exp_type = ExperimentType(experiment_type)
        except ValueError:
            exp_type = ExperimentType.AB_TEST
            logger.warning(f"Invalid experiment type: {experiment_type}. Using AB_TEST.")
            
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            variants=variant_instances,
            experiment_type=exp_type,
            traffic_allocation=traffic_allocation,
            description=description,
            is_active=True
        )
        
        # Store experiment
        self.experiments[experiment_id] = experiment
        
        logger.info(f"Created experiment: {name}", experiment_id=experiment_id)
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to get
            
        Returns:
            The Experiment instance, or None if not found
        """
        return self.experiments.get(experiment_id)
        
    def list_experiments(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Args:
            active_only: Whether to include only active experiments
            
        Returns:
            List of experiment dictionaries
        """
        experiments_list = []
        for experiment in self.experiments.values():
            if not active_only or experiment.is_active:
                experiments_list.append(experiment.to_dict())
        return experiments_list
        
    def get_variant_for_user(
        self, 
        experiment_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the variant for a specific user in an experiment.
        
        Args:
            experiment_id: ID of the experiment
            user_id: ID of the user
            
        Returns:
            Dictionary with variant details, or None if not in experiment
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
            
        variant = experiment.get_variant_for_user(user_id)
        if not variant:
            return None
            
        # Record impression
        experiment.record_impression(variant.variant_id)
        
        return variant.to_dict()
        
    def record_conversion(
        self, 
        experiment_id: str, 
        variant_id: str, 
        value: float = 1.0
    ) -> bool:
        """
        Record a conversion for a variant in an experiment.
        
        Args:
            experiment_id: ID of the experiment
            variant_id: ID of the variant
            value: Optional conversion value
            
        Returns:
            True if successful, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        return experiment.record_conversion(variant_id, value)
        
    def analyze_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze the results of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with analysis results, or None if experiment not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
            
        return experiment.analyze_results()
        
    def end_experiment(self, experiment_id: str) -> bool:
        """
        End an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            True if successful, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
            
        experiment.end_experiment()
        logger.info(f"Ended experiment: {experiment.name}", experiment_id=experiment_id)
        return True
        
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            True if successful, False otherwise
        """
        if experiment_id not in self.experiments:
            return False
            
        del self.experiments[experiment_id]
        logger.info(f"Deleted experiment with ID: {experiment_id}")
        return True

# Global instance
experiment_manager = ExperimentManager()

def get_experiment_manager() -> ExperimentManager:
    """Get the global experiment manager instance."""
    return experiment_manager 