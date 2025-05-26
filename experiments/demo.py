#!/usr/bin/env python3
"""
Demo script for the OpenInsight Experiment Service.

This script demonstrates how to use the experiment service to:
1. Create experiments
2. Assign variants to users
3. Record conversions
4. Analyze experiment results
"""

import sys
import os
import time
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from OpenInsight.experiments.experiment_service import (
    ExperimentManager,
    get_experiment_manager,
    ExperimentType
)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)

def print_experiment_info(experiment):
    """Print experiment information."""
    exp_dict = experiment.to_dict()
    print(f"Experiment ID: {exp_dict['experiment_id']}")
    print(f"Name: {exp_dict['name']}")
    print(f"Type: {exp_dict['experiment_type']}")
    print(f"Is Active: {exp_dict['is_active']}")
    print(f"Variants:")
    for i, variant in enumerate(exp_dict['variants']):
        print(f"  {i+1}. {variant['name']} (ID: {variant['variant_id']})")
        print(f"     Impressions: {variant['impressions']}")
        print(f"     Conversions: {variant['conversions']}")
        print(f"     Conversion Rate: {variant['conversion_rate']:.2%}")
        print(f"     Conversion Value: {variant['conversion_value']:.2f}")

def print_analysis(analysis):
    """Print experiment analysis."""
    print("Analysis Results:")
    print(f"Total Impressions: {analysis['total_impressions']}")
    print(f"Total Conversions: {analysis['total_conversions']}")
    print(f"Overall Conversion Rate: {analysis['overall_conversion_rate']:.2%}")
    
    print("\nVariant Performance:")
    for i, variant in enumerate(analysis['variants']):
        is_control = variant.get('is_control', False)
        control_text = " (Control)" if is_control else ""
        p_value = variant.get('p_value')
        p_value_text = f"p-value: {p_value:.4f}" if p_value is not None else ""
        rel_improvement = variant.get('relative_improvement', 0)
        rel_improvement_text = f"Relative Improvement: {rel_improvement:.2f}%" if not is_control else ""
        
        print(f"  {i+1}. {variant['name']}{control_text}")
        print(f"     Impressions: {variant['impressions']}")
        print(f"     Conversions: {variant['conversions']}")
        print(f"     Conversion Rate: {variant['conversion_rate']:.2%}")
        if p_value_text:
            print(f"     {p_value_text}")
        if rel_improvement_text:
            print(f"     {rel_improvement_text}")
    
    if analysis['winner']:
        winner = analysis['winner']
        print("\nWinner:")
        print(f"  {winner['name']} (ID: {winner['variant_id']})")
        print(f"  Conversion Rate: {winner['conversion_rate']:.2%}")
        print(f"  p-value: {winner['p_value']:.4f}")
        print(f"  Relative Improvement: {winner['relative_improvement']:.2f}%")
    else:
        print("\nNo clear winner yet.")

def demo_ab_test():
    """Demonstrate a simple A/B test."""
    print_header("Simple A/B Test")
    
    # Get the experiment manager
    manager = get_experiment_manager()
    
    # Create an experiment
    experiment = manager.create_experiment(
        name="Homepage Button Color Test",
        variants=[
            {"name": "Blue Button", "description": "Standard blue button"},
            {"name": "Green Button", "description": "New green button"}
        ],
        experiment_type="ab_test",
        traffic_allocation=1.0,
        description="Testing whether a green CTA button performs better than our standard blue"
    )
    
    print("Experiment created!")
    print_experiment_info(experiment)
    
    # Simulate user traffic and conversions
    print("\nSimulating user traffic and conversions...")
    
    # Get variant IDs
    variant_ids = list(experiment.variants.keys())
    blue_button_id = variant_ids[0]
    green_button_id = variant_ids[1]
    
    # Blue button: 20% conversion rate (baseline)
    # Simulate 100 users, 20 conversions
    for i in range(100):
        user_id = f"user_blue_{i}"
        variant = experiment.get_variant_for_user(user_id)
        # Record an impression manually to ensure it's counted
        experiment.record_impression(blue_button_id)
        # Every 5th user converts
        if i % 5 == 0:
            experiment.record_conversion(blue_button_id, 1.0)
    
    # Green button: 30% conversion rate (improved)
    # Simulate 100 users, 30 conversions
    for i in range(100):
        user_id = f"user_green_{i}"
        variant = experiment.get_variant_for_user(user_id)
        # Record an impression manually to ensure it's counted
        experiment.record_impression(green_button_id)
        # Every 3.33rd user converts
        if i % 3 == 0:
            experiment.record_conversion(green_button_id, 1.5)  # Higher value per conversion
    
    print("Traffic simulation complete.")
    print_experiment_info(experiment)
    
    # Analyze the results
    print("\nAnalyzing experiment results...")
    analysis = experiment.analyze_results()
    print_analysis(analysis)
    
    return experiment

def demo_multi_armed_bandit():
    """Demonstrate a multi-armed bandit experiment."""
    print_header("Multi-Armed Bandit Experiment")
    
    # Get the experiment manager
    manager = get_experiment_manager()
    
    # Create an experiment
    experiment = manager.create_experiment(
        name="Pricing Strategy Test",
        variants=[
            {"name": "$19.99", "description": "Standard price"},
            {"name": "$24.99", "description": "Premium price"},
            {"name": "$14.99", "description": "Discount price"}
        ],
        experiment_type="multi_armed_bandit",
        traffic_allocation=1.0,
        description="Testing different price points to maximize revenue"
    )
    
    print("Experiment created!")
    print_experiment_info(experiment)
    
    # Simulate user traffic and conversions
    print("\nSimulating user traffic and conversions...")
    
    # Get variant IDs
    variant_ids = list(experiment.variants.keys())
    standard_id = variant_ids[0]
    premium_id = variant_ids[1]
    discount_id = variant_ids[2]
    
    # Standard price: 10% conversion rate, $19.99 value
    for i in range(100):
        user_id = f"user_standard_{i}"
        variant = experiment.get_variant_for_user(user_id)
        # Record an impression
        experiment.record_impression(standard_id)
        if i % 10 == 0:
            experiment.record_conversion(standard_id, 19.99)
    
    # Premium price: 5% conversion rate, $24.99 value
    for i in range(100):
        user_id = f"user_premium_{i}"
        variant = experiment.get_variant_for_user(user_id)
        # Record an impression
        experiment.record_impression(premium_id)
        if i % 20 == 0:
            experiment.record_conversion(premium_id, 24.99)
    
    # Discount price: 15% conversion rate, $14.99 value
    for i in range(100):
        user_id = f"user_discount_{i}"
        variant = experiment.get_variant_for_user(user_id)
        # Record an impression
        experiment.record_impression(discount_id)
        if i % 7 == 0:
            experiment.record_conversion(discount_id, 14.99)
    
    print("Traffic simulation complete.")
    print_experiment_info(experiment)
    
    # Analyze the results
    print("\nAnalyzing experiment results...")
    analysis = experiment.analyze_results()
    print_analysis(analysis)
    
    # Simulate additional traffic to see MAB adaptively allocate traffic
    print("\nSimulating additional traffic with MAB optimization...")
    for i in range(300):
        user_id = f"adaptive_user_{i}"
        variant = experiment.get_variant_for_user(user_id)
        
        # Record the impression
        experiment.record_impression(variant.variant_id)
        
        # The MAB should favor the better-performing variants
        if variant.variant_id == standard_id and i % 10 == 0:
            experiment.record_conversion(standard_id, 19.99)
        elif variant.variant_id == premium_id and i % 20 == 0:
            experiment.record_conversion(premium_id, 24.99)
        elif variant.variant_id == discount_id and i % 7 == 0:
            experiment.record_conversion(discount_id, 14.99)
    
    print("Additional traffic simulation complete.")
    print_experiment_info(experiment)
    
    # Final analysis
    print("\nFinal analysis results...")
    analysis = experiment.analyze_results()
    print_analysis(analysis)
    
    return experiment

def main():
    """Run the demo."""
    print("OpenInsight Experiment Service Demo")
    print("=" * 50)
    print(f"Current Time: {datetime.now()}")
    print("-" * 50)
    
    # Run the A/B test demo
    ab_experiment = demo_ab_test()
    
    # Run the multi-armed bandit demo
    mab_experiment = demo_multi_armed_bandit()
    
    # End the experiments
    print_header("Ending Experiments")
    
    ab_experiment.end_experiment()
    print(f"A/B Test Experiment '{ab_experiment.name}' ended.")
    
    mab_experiment.end_experiment()
    print(f"Multi-Armed Bandit Experiment '{mab_experiment.name}' ended.")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main() 