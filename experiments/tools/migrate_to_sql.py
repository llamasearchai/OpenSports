#!/usr/bin/env python3
"""
Migration utility to transfer experiments from JSON storage to SQL database storage.

This script helps transition from the default JSON storage to a more robust SQL 
database backend for production environments.

Usage:
    python migrate_to_sql.py --source data/experiments.json --target sqlite:///data/experiments.db

"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from OpenInsight.experiments import (
    Experiment,
    SQLDatabaseBackend,
    JSONFileBackend
)

def migrate_json_to_sql(source_path, target_connection_string, create_tables=True):
    """
    Migrate experiments from a JSON file to a SQL database.
    
    Args:
        source_path: Path to the source JSON file
        target_connection_string: SQLAlchemy connection string for the target database
        create_tables: Whether to create tables in the target database if they don't exist
    
    Returns:
        Tuple of (success status, number of experiments migrated)
    """
    print(f"Starting migration from {source_path} to {target_connection_string}")
    
    # Create the source and target backends
    try:
        source_backend = JSONFileBackend(source_path)
        target_backend = SQLDatabaseBackend(target_connection_string, create_tables)
    except Exception as e:
        print(f"Failed to initialize backends: {str(e)}")
        return False, 0
    
    # Load experiments from JSON
    try:
        experiments = source_backend.load_experiments()
        if not experiments:
            print(f"No experiments found in source file: {source_path}")
            return True, 0
            
        num_experiments = len(experiments)
        print(f"Loaded {num_experiments} experiments from JSON")
    except Exception as e:
        print(f"Failed to load experiments from JSON: {str(e)}")
        return False, 0
    
    # Save experiments to SQL
    try:
        success = target_backend.save_experiments(experiments)
        if success:
            print(f"Successfully migrated {num_experiments} experiments to SQL database")
            return True, num_experiments
        else:
            print("Failed to save experiments to SQL database")
            return False, 0
    except Exception as e:
        print(f"Error saving experiments to SQL database: {str(e)}")
        return False, 0

def validate_migration(source_path, target_connection_string):
    """
    Validate that the experiments were migrated correctly.
    
    Args:
        source_path: Path to the source JSON file
        target_connection_string: SQLAlchemy connection string for the target database
    
    Returns:
        Tuple of (success status, validation results dictionary)
    """
    print(f"Validating migration...")
    
    # Create the source and target backends
    try:
        source_backend = JSONFileBackend(source_path)
        target_backend = SQLDatabaseBackend(target_connection_string, create_tables=False)
    except Exception as e:
        print(f"Failed to initialize backends for validation: {str(e)}")
        return False, {}
    
    # Load experiments from both sources
    try:
        source_experiments = source_backend.load_experiments()
        target_experiments = target_backend.load_experiments()
        
        source_count = len(source_experiments)
        target_count = len(target_experiments)
        
        results = {
            "source_count": source_count,
            "target_count": target_count,
            "count_match": source_count == target_count,
            "missing_experiments": [],
            "data_differences": []
        }
        
        # Check for missing experiments
        for exp_id in source_experiments:
            if exp_id not in target_experiments:
                results["missing_experiments"].append(exp_id)
        
        # Check for data differences
        for exp_id, source_exp in source_experiments.items():
            if exp_id in target_experiments:
                target_exp = target_experiments[exp_id]
                
                # Compare key metrics
                source_dict = source_exp.to_dict()
                target_dict = target_exp.to_dict()
                
                # Check variant counts match
                if len(source_dict["variants"]) != len(target_dict["variants"]):
                    results["data_differences"].append({
                        "experiment_id": exp_id,
                        "issue": "variant_count_mismatch",
                        "source": len(source_dict["variants"]),
                        "target": len(target_dict["variants"])
                    })
                    continue
                
                # Check basic properties
                for prop in ["name", "experiment_type", "traffic_allocation", "is_active"]:
                    if source_dict.get(prop) != target_dict.get(prop):
                        results["data_differences"].append({
                            "experiment_id": exp_id,
                            "issue": f"{prop}_mismatch",
                            "source": source_dict.get(prop),
                            "target": target_dict.get(prop)
                        })
                
                # Check variant data
                for source_variant in source_dict["variants"]:
                    variant_id = source_variant["variant_id"]
                    target_variant = next((v for v in target_dict["variants"] 
                                          if v["variant_id"] == variant_id), None)
                    
                    if not target_variant:
                        results["data_differences"].append({
                            "experiment_id": exp_id,
                            "issue": "missing_variant",
                            "variant_id": variant_id
                        })
                        continue
                    
                    # Check variant metrics
                    for metric in ["impressions", "conversions", "conversion_value"]:
                        if source_variant.get(metric) != target_variant.get(metric):
                            results["data_differences"].append({
                                "experiment_id": exp_id,
                                "variant_id": variant_id,
                                "issue": f"{metric}_mismatch",
                                "source": source_variant.get(metric),
                                "target": target_variant.get(metric)
                            })
        
        # Determine overall success
        validation_success = (
            results["count_match"] and
            not results["missing_experiments"] and
            not results["data_differences"]
        )
        
        if validation_success:
            print("Validation successful! All experiments migrated correctly.")
        else:
            print("Validation found issues with the migration:")
            print(f"  - Source experiments: {source_count}")
            print(f"  - Target experiments: {target_count}")
            print(f"  - Missing experiments: {len(results['missing_experiments'])}")
            print(f"  - Data differences: {len(results['data_differences'])}")
        
        return validation_success, results
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return False, {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Migrate experiments from JSON to SQL database")
    parser.add_argument("--source", required=True, help="Path to source JSON file")
    parser.add_argument("--target", required=True, help="SQLAlchemy connection string for target database")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    parser.add_argument("--backup", action="store_true", help="Create a backup of the source JSON file")
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup:
        backup_path = f"{args.source}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            import shutil
            shutil.copy2(args.source, backup_path)
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Failed to create backup: {str(e)}")
            return
    
    # Perform migration
    success, count = migrate_json_to_sql(args.source, args.target)
    
    if not success:
        print("Migration failed. Exiting.")
        return
    
    # Validate migration
    if not args.skip_validation and count > 0:
        validation_success, results = validate_migration(args.source, args.target)
        
        if validation_success:
            print("\nMigration completed successfully!")
            print(f"Migrated {count} experiments from JSON to SQL database.")
        else:
            print("\nMigration completed with validation issues.")
            print("Please check the validation results and resolve any issues.")
    elif count > 0:
        print("\nMigration completed but validation was skipped.")
        print(f"Migrated {count} experiments from JSON to SQL database.")
        print("We recommend running validation separately to ensure data integrity.")
    else:
        print("\nNo experiments were migrated. Source file may be empty.")

if __name__ == "__main__":
    main() 