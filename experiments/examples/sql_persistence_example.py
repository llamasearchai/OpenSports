#!/usr/bin/env python3
"""
Example of using the SQL persistence backend for the OpenInsight Experiment Service.

This example demonstrates how to:
1. Set up a SQL database for storing experiments
2. Run experiments with automatic persistence
3. Restart the application and retrieve the experiment data

Run this example with:
```
python sql_persistence_example.py
```
"""

import sys
import os
import time
from datetime import datetime
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from OpenInsight.experiments import (
    ExperimentType,
    get_sql_persistent_manager,
    SQLDatabaseBackend,
    AutoSavingManager,
    calculate_required_sample_size
)

def create_experiment(manager):
    """Create a new A/B test experiment."""
    print("Creating new experiment...")
    
    # Create a simple A/B test
    experiment = manager.create_experiment(
        name="SQL Persistence Test Experiment",
        variants=[
            {"name": "Control", "description": "Original version"},
            {"name": "Treatment", "description": "New version"}
        ],
        experiment_type="ab_test",
        traffic_allocation=1.0,
        description="Example experiment showing SQL persistence"
    )
    
    print(f"Created experiment with ID: {experiment.experiment_id}")
    return experiment

def simulate_traffic(manager, experiment_id, num_users=100):
    """Simulate user traffic and conversions."""
    print(f"Simulating traffic for {num_users} users...")
    
    # Control: 20% conversion rate
    # Treatment: 30% conversion rate
    
    for i in range(num_users):
        user_id = f"user_{i}"
        
        # Get variant
        variant = manager.get_variant_for_user(experiment_id, user_id)
        if not variant:
            continue
            
        # Record conversion based on variant
        if variant["name"] == "Control" and i % 5 == 0:  # 20% conversion
            manager.record_conversion(experiment_id, variant["variant_id"], 1.0)
        elif variant["name"] == "Treatment" and i % 3 == 0:  # 33% conversion
            manager.record_conversion(experiment_id, variant["variant_id"], 1.5)
    
    print("Traffic simulation complete.")

def print_experiment_details(manager, experiment_id):
    """Print experiment details and analysis."""
    print("\n" + "="*60)
    print("EXPERIMENT DETAILS")
    print("="*60)
    
    experiment = manager.get_experiment(experiment_id)
    if not experiment:
        print(f"Experiment {experiment_id} not found!")
        return
        
    exp_dict = experiment.to_dict()
    print(f"ID: {exp_dict['experiment_id']}")
    print(f"Name: {exp_dict['name']}")
    print(f"Type: {exp_dict['experiment_type']}")
    print(f"Is Active: {exp_dict['is_active']}")
    print(f"Created: {exp_dict['created_at']}")
    print(f"Last Updated: {exp_dict['updated_at']}")
    
    print("\nVARIANTS:")
    for i, variant in enumerate(exp_dict['variants']):
        print(f"  {i+1}. {variant['name']} (ID: {variant['variant_id']})")
        print(f"     Impressions: {variant['impressions']}")
        print(f"     Conversions: {variant['conversions']}")
        print(f"     Conversion Rate: {variant['conversion_rate']:.2%}")
        print(f"     Conversion Value: {variant['conversion_value']:.2f}")
    
    # Analyze results
    print("\nANALYSIS:")
    analysis = manager.analyze_experiment(experiment_id)
    
    print(f"Total Impressions: {analysis['total_impressions']}")
    print(f"Total Conversions: {analysis['total_conversions']}")
    print(f"Overall Conversion Rate: {analysis['overall_conversion_rate']:.2%}")
    
    # Calculate required sample size for 20% MDE
    baseline_rate = experiment.variants[list(experiment.variants.keys())[0]].get_conversion_rate()
    if baseline_rate > 0:
        sample_size = calculate_required_sample_size(
            baseline_conversion_rate=baseline_rate,
            minimum_detectable_effect=20
        )
        print(f"\nRequired sample size for 20% MDE: {sample_size} per variant")
        print(f"Current progress: {analysis['total_impressions']/2}/{sample_size} per variant ({analysis['total_impressions']/2/sample_size:.1%})")
    
    if analysis['winner']:
        winner = analysis['winner']
        print("\nWINNER DETECTED:")
        print(f"  {winner['name']} is better!")
        print(f"  Conversion Rate: {winner['conversion_rate']:.2%}")
        print(f"  Relative Improvement: {winner['relative_improvement']:.2f}%")
        print(f"  p-value: {winner['p_value']:.4f}")
    else:
        print("\nNo winner detected yet.")
    
    print("="*60)

def demo_sql_persistence():
    """
    Demonstrate SQL persistence by simulating creating and retrieving experiments.
    """
    # Use an in-memory SQLite database for this example
    # For a real application, use a persistent database file or server
    db_url = "sqlite:///example_experiments.db"
    
    parser = argparse.ArgumentParser(description="SQL Persistence Example")
    parser.add_argument("--create", action="store_true", help="Create a new experiment")
    parser.add_argument("--users", type=int, default=100, help="Number of users to simulate")
    args = parser.parse_args()
    
    # Get a manager with SQL persistence
    manager = get_sql_persistent_manager(connection_string=db_url, autosave_interval=5)
    
    # List existing experiments
    experiments = manager.list_experiments()
    
    if args.create or not experiments:
        # Create a new experiment
        experiment = create_experiment(manager)
        experiment_id = experiment.experiment_id
        print(f"Created experiment with ID: {experiment_id}")
        
        # Simulate some initial traffic
        simulate_traffic(manager, experiment_id, args.users)
        
        # Print experiment details
        print_experiment_details(manager, experiment_id)
        
        print("\nExperiment data is now persisted in the SQL database.")
        print(f"Run this script again without --create to see the persisted data.")
        print(f"Or run with --users <number> to simulate more traffic.")
    elif experiments:
        print("Found existing experiments in the database:")
        for i, exp in enumerate(experiments):
            print(f"{i+1}. {exp['name']} (ID: {exp['experiment_id']})")
        
        # Use the first experiment
        experiment_id = experiments[0]["experiment_id"]
        
        # If --users specified, simulate more traffic
        if args.users > 0:
            print(f"\nSimulating additional traffic for experiment {experiment_id}...")
            simulate_traffic(manager, experiment_id, args.users)
        
        # Print details
        print_experiment_details(manager, experiment_id)

if __name__ == "__main__":
    print("OpenInsight Experiment SQL Persistence Example")
    print("-" * 50)
    print(f"Current Time: {datetime.now()}")
    print("-" * 50)
    
    demo_sql_persistence() 