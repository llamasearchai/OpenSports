#!/usr/bin/env python3
"""
Example of using the Redis persistence backend for the OpenInsight Experiment Service.

This example demonstrates how to:
1. Set up a Redis server for storing experiments
2. Run experiments with automatic Redis persistence
3. Utilize Redis for high-performance and distributed applications

Run this example with:
```
python redis_persistence_example.py
```

Note: This example requires a Redis server running. 
You can start one with Docker using:
```
docker run --name redis-server -p 6379:6379 -d redis
```
"""

import sys
import os
import time
import random
from datetime import datetime
import argparse
import threading

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from OpenInsight.experiments import (
    ExperimentType,
    get_redis_persistent_manager,
    RedisBackend,
    AutoSavingManager
)

def create_experiment(manager):
    """Create a new A/B test experiment."""
    print("Creating new experiment...")
    
    # Create a simple A/B test
    experiment = manager.create_experiment(
        name="Redis High-Performance Test",
        variants=[
            {"name": "Control", "description": "Original version"},
            {"name": "Treatment", "description": "New version"}
        ],
        experiment_type="ab_test",
        traffic_allocation=1.0,
        description="Example experiment showing Redis persistence for high-performance applications"
    )
    
    print(f"Created experiment with ID: {experiment.experiment_id}")
    return experiment

def simulate_high_traffic(manager, experiment_id, num_users=1000, threads=4):
    """
    Simulate high traffic with multi-threading to demonstrate Redis performance.
    
    Args:
        manager: Experiment manager
        experiment_id: ID of the experiment
        num_users: Number of users to simulate
        threads: Number of threads to use
    """
    print(f"Simulating high traffic with {num_users} users across {threads} threads...")
    
    # Split users among threads
    users_per_thread = num_users // threads
    
    def worker(thread_id, start_user, end_user):
        """Worker function for each thread."""
        total_impressions = 0
        total_conversions = 0
        
        for i in range(start_user, end_user):
            # Create a unique user ID
            user_id = f"high_volume_user_{i}"
            
            # Get variant
            variant = manager.get_variant_for_user(experiment_id, user_id)
            if not variant:
                continue
                
            total_impressions += 1
                
            # Record conversion with probability dependent on variant
            # Control: 10% conversion rate
            # Treatment: 15% conversion rate
            roll = random.random()
            if variant["name"] == "Control" and roll < 0.10:
                manager.record_conversion(experiment_id, variant["variant_id"], 1.0)
                total_conversions += 1
            elif variant["name"] == "Treatment" and roll < 0.15:
                manager.record_conversion(experiment_id, variant["variant_id"], 1.5)
                total_conversions += 1
        
        print(f"Thread {thread_id} processed {total_impressions} impressions and {total_conversions} conversions")
    
    # Create and start threads
    threads_list = []
    for i in range(threads):
        start_user = i * users_per_thread
        end_user = start_user + users_per_thread
        thread = threading.Thread(
            target=worker, 
            args=(i, start_user, end_user)
        )
        threads_list.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads_list:
        thread.join()
    
    print(f"High traffic simulation complete.")

def benchmark_access(manager, experiment_id, iterations=100):
    """
    Benchmark access performance to Redis.
    
    Args:
        manager: Experiment manager
        experiment_id: ID of the experiment
        iterations: Number of iterations for the benchmark
    """
    print(f"Running access performance benchmark ({iterations} iterations)...")
    
    # Benchmark variant assignment
    start_time = time.time()
    for i in range(iterations):
        user_id = f"benchmark_user_{i}"
        variant = manager.get_variant_for_user(experiment_id, user_id)
    
    variant_time = time.time() - start_time
    variant_rate = iterations / variant_time if variant_time > 0 else 0
    
    print(f"Variant assignment: {variant_rate:.1f} requests/second")
    
    # Benchmark analysis
    start_time = time.time()
    for i in range(10):  # Analysis is more computationally intensive, so fewer iterations
        analysis = manager.analyze_experiment(experiment_id)
    
    analysis_time = time.time() - start_time
    analysis_rate = 10 / analysis_time if analysis_time > 0 else 0
    
    print(f"Experiment analysis: {analysis_rate:.1f} requests/second")
    
    return {
        "variant_rate": variant_rate,
        "analysis_rate": analysis_rate
    }

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

def demo_redis_persistence():
    """
    Demonstrate Redis persistence with high performance scenarios.
    """
    parser = argparse.ArgumentParser(description="Redis Persistence Example")
    parser.add_argument("--create", action="store_true", help="Create a new experiment")
    parser.add_argument("--users", type=int, default=1000, help="Number of users to simulate")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for simulation")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0", help="Redis server URL")
    parser.add_argument("--benchmark", action="store_true", help="Run access performance benchmark")
    args = parser.parse_args()
    
    # Get a manager with Redis persistence
    # Use a short autosave interval for this example to demonstrate Redis performance
    manager = get_redis_persistent_manager(
        redis_url=args.redis_url, 
        key_prefix="example_experiments:", 
        autosave_interval=2
    )
    
    # List existing experiments
    experiments = manager.list_experiments()
    
    if args.create or not experiments:
        # Create a new experiment
        experiment = create_experiment(manager)
        experiment_id = experiment.experiment_id
        print(f"Created experiment with ID: {experiment_id}")
        
        # Simulate high traffic
        simulate_high_traffic(manager, experiment_id, args.users, args.threads)
        
        # Force a save
        manager.save()
        
        # Print experiment details
        print_experiment_details(manager, experiment_id)
        
        if args.benchmark:
            benchmark_access(manager, experiment_id)
        
        print("\nExperiment data is now persisted in Redis.")
        print(f"Run this script again without --create to see the persisted data.")
    elif experiments:
        print("Found existing experiments in Redis:")
        for i, exp in enumerate(experiments):
            print(f"{i+1}. {exp['name']} (ID: {exp['experiment_id']})")
        
        # Use the first experiment
        experiment_id = experiments[0]["experiment_id"]
        
        # If users specified, simulate more traffic
        if args.users > 0:
            print(f"\nSimulating additional traffic for experiment {experiment_id}...")
            simulate_high_traffic(manager, experiment_id, args.users, args.threads)
            manager.save()  # Force a save
        
        # Print details
        print_experiment_details(manager, experiment_id)
        
        if args.benchmark:
            benchmark_access(manager, experiment_id)
    
    return manager

if __name__ == "__main__":
    print("OpenInsight Experiment Redis Persistence Example")
    print("-" * 50)
    print(f"Current Time: {datetime.now()}")
    print("-" * 50)
    
    manager = demo_redis_persistence()
    
    # Keep the script running if requested (to demonstrate the autosave)
    if "--keep-running" in sys.argv:
        print("\nKeeping the manager alive to demonstrate autosave...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nSaving before exit...")
            manager.save()
            print("Done!") 