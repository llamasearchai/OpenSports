#!/usr/bin/env python3
"""
Migration utility to transfer experiments between different storage backends.

This script helps migrate experiment data between JSON, SQL, and Redis backends.
It supports:
1. JSON to SQL migration
2. JSON to Redis migration
3. SQL to JSON migration
4. SQL to Redis migration
5. Redis to JSON migration
6. Redis to SQL migration

Usage:
    python migrate_between_backends.py --source-type json --source-path data/experiments.json --target-type sql --target-path sqlite:///data/experiments.db

"""

import os
import sys
import argparse
import time
import datetime
from typing import Dict, Optional, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from OpenInsight.experiments import (
    StorageBackend,
    JSONFileBackend,
    SQLDatabaseBackend, 
    RedisBackend,
    Experiment
)

def create_backend(backend_type: str, path: str, **kwargs) -> StorageBackend:
    """
    Create a storage backend of the specified type.
    
    Args:
        backend_type: Type of backend ('json', 'sql', or 'redis')
        path: Path or connection string for the backend
        **kwargs: Additional arguments for the backend
        
    Returns:
        StorageBackend instance
    """
    if backend_type.lower() == 'json':
        return JSONFileBackend(path)
    elif backend_type.lower() == 'sql':
        create_tables = kwargs.get('create_tables', True)
        return SQLDatabaseBackend(path, create_tables=create_tables)
    elif backend_type.lower() == 'redis':
        key_prefix = kwargs.get('key_prefix', 'experiments:')
        expire_time = kwargs.get('expire_time', None)
        return RedisBackend(path, key_prefix=key_prefix, expire_time=expire_time)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

def migrate_experiments(
    source_backend: StorageBackend,
    target_backend: StorageBackend,
    verbose: bool = True,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Migrate experiments from source backend to target backend.
    
    Args:
        source_backend: Source backend to load from
        target_backend: Target backend to save to
        verbose: Whether to print progress information
        validate: Whether to validate the migration
        
    Returns:
        Dictionary with migration results
    """
    if verbose:
        print(f"Starting migration between backends...")
    
    start_time = time.time()
    
    # Load experiments from source
    try:
        if verbose:
            print("Loading experiments from source backend...")
        
        experiments = source_backend.load_experiments()
        
        if not experiments:
            if verbose:
                print("No experiments found in source backend.")
            return {
                "success": True,
                "count": 0,
                "elapsed_time": time.time() - start_time,
                "validation": {"success": True}
            }
            
        if verbose:
            print(f"Loaded {len(experiments)} experiments from source.")
            
    except Exception as e:
        print(f"Failed to load experiments from source: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to load from source: {str(e)}",
            "elapsed_time": time.time() - start_time
        }
    
    # Save experiments to target
    try:
        if verbose:
            print(f"Saving {len(experiments)} experiments to target backend...")
            
        success = target_backend.save_experiments(experiments)
        
        if not success:
            return {
                "success": False,
                "error": "Failed to save to target backend",
                "elapsed_time": time.time() - start_time
            }
            
        if verbose:
            print(f"Successfully saved {len(experiments)} experiments to target.")
            
    except Exception as e:
        print(f"Failed to save experiments to target: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to save to target: {str(e)}",
            "elapsed_time": time.time() - start_time
        }
    
    # Validate migration if requested
    validation_result = {"success": True}
    
    if validate:
        if verbose:
            print("Validating migration...")
            
        try:
            # Load experiments from target
            target_experiments = target_backend.load_experiments()
            
            # Compare experiment counts
            source_count = len(experiments)
            target_count = len(target_experiments)
            
            validation_result["source_count"] = source_count
            validation_result["target_count"] = target_count
            validation_result["count_match"] = source_count == target_count
            
            if source_count != target_count:
                validation_result["success"] = False
                if verbose:
                    print(f"Validation failed: Source has {source_count} experiments, target has {target_count}")
            
            # Check for missing experiments
            missing_experiments = []
            for exp_id in experiments:
                if exp_id not in target_experiments:
                    missing_experiments.append(exp_id)
            
            validation_result["missing_experiments"] = missing_experiments
            
            if missing_experiments:
                validation_result["success"] = False
                if verbose:
                    print(f"Validation failed: {len(missing_experiments)} experiments are missing in target")
            
            if validation_result["success"] and verbose:
                print("Validation successful: All experiments migrated correctly.")
                
        except Exception as e:
            validation_result["success"] = False
            validation_result["error"] = str(e)
            if verbose:
                print(f"Validation error: {str(e)}")
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"Migration completed in {elapsed_time:.2f} seconds")
    
    return {
        "success": True,
        "count": len(experiments),
        "elapsed_time": elapsed_time,
        "validation": validation_result
    }

def main():
    """Parse arguments and run migration."""
    parser = argparse.ArgumentParser(description="Migrate experiments between storage backends")
    
    # Source backend options
    parser.add_argument("--source-type", required=True, choices=["json", "sql", "redis"],
                       help="Source backend type")
    parser.add_argument("--source-path", required=True,
                       help="Source path (file path for JSON, connection string for SQL/Redis)")
    parser.add_argument("--source-key-prefix", default="experiments:",
                       help="Key prefix for Redis source (only for Redis)")
    
    # Target backend options
    parser.add_argument("--target-type", required=True, choices=["json", "sql", "redis"],
                       help="Target backend type")
    parser.add_argument("--target-path", required=True,
                       help="Target path (file path for JSON, connection string for SQL/Redis)")
    parser.add_argument("--target-key-prefix", default="experiments:",
                       help="Key prefix for Redis target (only for Redis)")
    parser.add_argument("--create-tables", action="store_true",
                       help="Create tables in SQL target if they don't exist (only for SQL)")
    parser.add_argument("--expire-time", type=int, default=None,
                       help="Expiration time in seconds for Redis keys (only for Redis)")
    
    # General options
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip validation step")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup before migration (only for JSON source)")
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup and args.source_type == "json":
        backup_path = f"{args.source_path}.backup-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            import shutil
            shutil.copy2(args.source_path, backup_path)
            if not args.quiet:
                print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Failed to create backup: {str(e)}")
            return
    
    # Create source backend
    try:
        source_kwargs = {}
        if args.source_type == "redis":
            source_kwargs["key_prefix"] = args.source_key_prefix
            
        source_backend = create_backend(args.source_type, args.source_path, **source_kwargs)
        if not args.quiet:
            print(f"Created {args.source_type.upper()} source backend")
    except Exception as e:
        print(f"Failed to create source backend: {str(e)}")
        return
    
    # Create target backend
    try:
        target_kwargs = {}
        if args.target_type == "sql":
            target_kwargs["create_tables"] = args.create_tables
        elif args.target_type == "redis":
            target_kwargs["key_prefix"] = args.target_key_prefix
            target_kwargs["expire_time"] = args.expire_time
            
        target_backend = create_backend(args.target_type, args.target_path, **target_kwargs)
        if not args.quiet:
            print(f"Created {args.target_type.upper()} target backend")
    except Exception as e:
        print(f"Failed to create target backend: {str(e)}")
        return
    
    # Perform migration
    result = migrate_experiments(
        source_backend=source_backend,
        target_backend=target_backend,
        verbose=not args.quiet,
        validate=not args.no_validate
    )
    
    # Print summary
    if not args.quiet:
        if result["success"]:
            print(f"\nMigration summary:")
            print(f"- Migrated {result['count']} experiments")
            print(f"- Elapsed time: {result['elapsed_time']:.2f} seconds")
            
            if "validation" in result and not args.no_validate:
                if result["validation"]["success"]:
                    print("- Validation: Success")
                else:
                    print("- Validation: Failed")
                    if "missing_experiments" in result["validation"] and result["validation"]["missing_experiments"]:
                        print(f"  - Missing experiments: {len(result['validation']['missing_experiments'])}")
                    if "error" in result["validation"]:
                        print(f"  - Error: {result['validation']['error']}")
        else:
            print(f"\nMigration failed:")
            print(f"- Error: {result.get('error', 'Unknown error')}")
    
    # Return exit code
    return 0 if result["success"] and (args.no_validate or result["validation"]["success"]) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 