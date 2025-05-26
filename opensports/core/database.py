"""
Database management for OpenSports platform.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import sqlite3
import asyncio
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager, contextmanager
import pandas as pd
import duckdb
import sqlite_utils
from opensports.core.config import settings
from opensports.core.logging import get_logger, LoggerMixin

logger = get_logger(__name__)


class Database(LoggerMixin):
    """Main database interface supporting SQLite and DuckDB."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.sqlite_db = None
        self.duckdb_conn = None
        self._setup_databases()
    
    def _setup_databases(self) -> None:
        """Initialize database connections."""
        try:
            # SQLite setup
            if self.database_url.startswith("sqlite:///"):
                db_path = self.database_url.replace("sqlite:///", "")
                self.sqlite_db = sqlite_utils.Database(db_path)
                self.logger.info("SQLite database initialized", path=db_path)
            
            # DuckDB setup for analytics
            self.duckdb_conn = duckdb.connect(":memory:")
            self.logger.info("DuckDB connection established")
            
            # Create initial tables
            self._create_tables()
            
        except Exception as e:
            self.logger.error("Failed to setup databases", error=str(e))
            raise
    
    def _create_tables(self) -> None:
        """Create initial database tables."""
        # Sports data tables
        tables = {
            "games": {
                "id": str,
                "sport": str,
                "home_team": str,
                "away_team": str,
                "game_date": str,
                "season": str,
                "home_score": int,
                "away_score": int,
                "status": str,
                "created_at": str,
                "updated_at": str,
            },
            "players": {
                "id": str,
                "name": str,
                "team": str,
                "position": str,
                "sport": str,
                "age": int,
                "height": float,
                "weight": float,
                "created_at": str,
                "updated_at": str,
            },
            "player_stats": {
                "id": str,
                "player_id": str,
                "game_id": str,
                "points": float,
                "assists": float,
                "rebounds": float,
                "minutes_played": float,
                "field_goal_percentage": float,
                "created_at": str,
            },
            "teams": {
                "id": str,
                "name": str,
                "city": str,
                "sport": str,
                "conference": str,
                "division": str,
                "founded": int,
                "created_at": str,
                "updated_at": str,
            },
            "predictions": {
                "id": str,
                "model_id": str,
                "prediction_type": str,
                "target_id": str,
                "prediction_value": float,
                "confidence": float,
                "features": str,  # JSON
                "created_at": str,
                "actual_value": float,
            },
            "experiments": {
                "id": str,
                "name": str,
                "experiment_type": str,
                "status": str,
                "traffic_allocation": float,
                "created_at": str,
                "updated_at": str,
                "ended_at": str,
            },
            "experiment_variants": {
                "id": str,
                "experiment_id": str,
                "name": str,
                "description": str,
                "impressions": int,
                "conversions": int,
                "conversion_value": float,
                "created_at": str,
            },
            "user_assignments": {
                "id": str,
                "user_id": str,
                "experiment_id": str,
                "variant_id": str,
                "assigned_at": str,
            },
            "model_performance": {
                "id": str,
                "model_id": str,
                "metric_name": str,
                "metric_value": float,
                "evaluation_date": str,
                "dataset_size": int,
                "created_at": str,
            },
        }
        
        if self.sqlite_db:
            for table_name, schema in tables.items():
                try:
                    self.sqlite_db[table_name].create(schema, if_not_exists=True)
                    self.logger.debug("Created table", table=table_name)
                except Exception as e:
                    self.logger.warning("Failed to create table", table=table_name, error=str(e))
    
    @contextmanager
    def get_sqlite_connection(self):
        """Get SQLite connection context manager."""
        if not self.sqlite_db:
            raise ValueError("SQLite database not initialized")
        
        try:
            yield self.sqlite_db
        except Exception as e:
            self.logger.error("SQLite operation failed", error=str(e))
            raise
    
    @contextmanager
    def get_duckdb_connection(self):
        """Get DuckDB connection context manager."""
        if not self.duckdb_conn:
            raise ValueError("DuckDB connection not initialized")
        
        try:
            yield self.duckdb_conn
        except Exception as e:
            self.logger.error("DuckDB operation failed", error=str(e))
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        try:
            with self.get_sqlite_connection() as db:
                if params:
                    result = db.execute(query, params).fetchall()
                else:
                    result = db.execute(query).fetchall()
                
                # Convert to list of dicts
                return [dict(row) for row in result]
                
        except Exception as e:
            self.logger.error("Query execution failed", query=query, error=str(e))
            raise
    
    def insert_data(self, table: str, data: Union[Dict, List[Dict]]) -> None:
        """Insert data into table."""
        try:
            with self.get_sqlite_connection() as db:
                if isinstance(data, dict):
                    db[table].insert(data)
                else:
                    db[table].insert_all(data)
                
                self.logger.debug("Data inserted", table=table, count=len(data) if isinstance(data, list) else 1)
                
        except Exception as e:
            self.logger.error("Data insertion failed", table=table, error=str(e))
            raise
    
    def update_data(self, table: str, data: Dict, where_clause: Dict) -> None:
        """Update data in table."""
        try:
            with self.get_sqlite_connection() as db:
                db[table].update(where_clause, data)
                self.logger.debug("Data updated", table=table)
                
        except Exception as e:
            self.logger.error("Data update failed", table=table, error=str(e))
            raise
    
    def delete_data(self, table: str, where_clause: Dict) -> None:
        """Delete data from table."""
        try:
            with self.get_sqlite_connection() as db:
                db[table].delete_where(where_clause)
                self.logger.debug("Data deleted", table=table)
                
        except Exception as e:
            self.logger.error("Data deletion failed", table=table, error=str(e))
            raise
    
    def get_table_info(self, table: str) -> Dict[str, Any]:
        """Get table information."""
        try:
            with self.get_sqlite_connection() as db:
                table_obj = db[table]
                return {
                    "name": table,
                    "count": table_obj.count,
                    "columns": [col.name for col in table_obj.columns],
                    "schema": {col.name: col.type for col in table_obj.columns},
                }
                
        except Exception as e:
            self.logger.error("Failed to get table info", table=table, error=str(e))
            raise
    
    def execute_analytics_query(self, query: str) -> pd.DataFrame:
        """Execute analytics query using DuckDB and return DataFrame."""
        try:
            with self.get_duckdb_connection() as conn:
                # Load SQLite data into DuckDB for analytics
                if self.sqlite_db:
                    db_path = self.database_url.replace("sqlite:///", "")
                    conn.execute(f"ATTACH '{db_path}' AS sqlite_db")
                
                result = conn.execute(query).fetchdf()
                self.logger.debug("Analytics query executed", rows=len(result))
                return result
                
        except Exception as e:
            self.logger.error("Analytics query failed", query=query, error=str(e))
            raise
    
    def backup_database(self, backup_path: str) -> None:
        """Create database backup."""
        try:
            if self.sqlite_db:
                import shutil
                db_path = self.database_url.replace("sqlite:///", "")
                shutil.copy2(db_path, backup_path)
                self.logger.info("Database backup created", backup_path=backup_path)
                
        except Exception as e:
            self.logger.error("Database backup failed", error=str(e))
            raise
    
    def get_sports_data_summary(self) -> Dict[str, Any]:
        """Get summary of sports data in the database."""
        try:
            summary = {}
            
            # Get counts for main tables
            tables = ["games", "players", "teams", "predictions", "experiments"]
            for table in tables:
                try:
                    info = self.get_table_info(table)
                    summary[table] = info["count"]
                except Exception:
                    summary[table] = 0
            
            # Get recent activity
            recent_games = self.execute_query(
                "SELECT COUNT(*) as count FROM games WHERE game_date >= date('now', '-7 days')"
            )
            summary["recent_games"] = recent_games[0]["count"] if recent_games else 0
            
            # Get prediction accuracy
            accuracy_query = """
                SELECT 
                    AVG(CASE WHEN ABS(prediction_value - actual_value) < 0.1 THEN 1 ELSE 0 END) as accuracy
                FROM predictions 
                WHERE actual_value IS NOT NULL
            """
            accuracy_result = self.execute_query(accuracy_query)
            summary["prediction_accuracy"] = accuracy_result[0]["accuracy"] if accuracy_result else 0
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to get sports data summary", error=str(e))
            return {}


class DatasetteManager(LoggerMixin):
    """Manager for Datasette integration."""
    
    def __init__(self, database: Database):
        self.database = database
        self.datasette_process = None
    
    async def start_datasette_server(self, port: int = 8001) -> None:
        """Start Datasette server for data exploration."""
        try:
            import subprocess
            
            db_path = self.database.database_url.replace("sqlite:///", "")
            cmd = [
                "datasette",
                db_path,
                "--port", str(port),
                "--host", "0.0.0.0",
                "--cors",
            ]
            
            self.datasette_process = subprocess.Popen(cmd)
            self.logger.info("Datasette server started", port=port)
            
        except Exception as e:
            self.logger.error("Failed to start Datasette server", error=str(e))
            raise
    
    def stop_datasette_server(self) -> None:
        """Stop Datasette server."""
        if self.datasette_process:
            self.datasette_process.terminate()
            self.datasette_process = None
            self.logger.info("Datasette server stopped")


# Global database instance
_database_instance: Optional[Database] = None


def get_database() -> Database:
    """Get global database instance."""
    global _database_instance
    
    if _database_instance is None:
        _database_instance = Database()
    
    return _database_instance


def init_database(database_url: Optional[str] = None) -> Database:
    """Initialize database with custom URL."""
    global _database_instance
    _database_instance = Database(database_url)
    return _database_instance 