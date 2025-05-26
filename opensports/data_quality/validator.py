"""
Advanced Data Validation System

Comprehensive data validation with custom rules, schema validation,
and detailed reporting for sports analytics data.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError
import jsonschema
from sqlalchemy import create_engine, text
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from opensports.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Validation status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'rule_name': self.rule_name,
            'status': self.status.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'execution_time': self.execution_time
        }


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
    
    @abstractmethod
    async def validate(self, data: Any) -> ValidationResult:
        """Execute validation rule"""
        pass


class SchemaValidationRule(ValidationRule):
    """JSON Schema validation rule"""
    
    def __init__(self, name: str, schema: Dict[str, Any], 
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.schema = schema
    
    async def validate(self, data: Any) -> ValidationResult:
        """Validate data against JSON schema"""
        start_time = datetime.utcnow()
        
        try:
            jsonschema.validate(data, self.schema)
            return ValidationResult(
                rule_name=self.name,
                status=ValidationStatus.PASSED,
                severity=self.severity,
                message="Schema validation passed",
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
        except jsonschema.ValidationError as e:
            return ValidationResult(
                rule_name=self.name,
                status=ValidationStatus.FAILED,
                severity=self.severity,
                message=f"Schema validation failed: {e.message}",
                details={'error_path': list(e.path), 'failed_value': e.instance},
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )


class DataFrameValidationRule(ValidationRule):
    """DataFrame validation rule"""
    
    def __init__(self, name: str, validator_func: Callable[[pd.DataFrame], bool],
                 error_message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.validator_func = validator_func
        self.error_message = error_message
    
    async def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame"""
        start_time = datetime.utcnow()
        
        try:
            is_valid = self.validator_func(data)
            status = ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED
            message = "Validation passed" if is_valid else self.error_message
            
            return ValidationResult(
                rule_name=self.name,
                status=status,
                severity=self.severity,
                message=message,
                details={'row_count': len(data), 'column_count': len(data.columns)},
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation error: {str(e)}",
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )


class SportsDataValidationRule(ValidationRule):
    """Sports-specific data validation rules"""
    
    def __init__(self, name: str, sport: str, data_type: str,
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, severity)
        self.sport = sport
        self.data_type = data_type
    
    async def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate sports data"""
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Sport-specific validations
            if self.sport.lower() == 'nba':
                errors.extend(self._validate_nba_data(data))
            elif self.sport.lower() == 'nfl':
                errors.extend(self._validate_nfl_data(data))
            elif self.sport.lower() == 'soccer':
                errors.extend(self._validate_soccer_data(data))
            elif self.sport.lower() == 'formula1':
                errors.extend(self._validate_f1_data(data))
            
            status = ValidationStatus.PASSED if not errors else ValidationStatus.FAILED
            message = "Sports data validation passed" if not errors else f"Found {len(errors)} validation errors"
            
            return ValidationResult(
                rule_name=self.name,
                status=status,
                severity=self.severity,
                message=message,
                details={'errors': errors, 'sport': self.sport, 'data_type': self.data_type},
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Sports validation error: {str(e)}",
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _validate_nba_data(self, data: pd.DataFrame) -> List[str]:
        """Validate NBA-specific data"""
        errors = []
        
        if 'points' in data.columns:
            if (data['points'] < 0).any():
                errors.append("Negative points values found")
            if (data['points'] > 100).any():
                errors.append("Unrealistic points values (>100) found")
        
        if 'minutes_played' in data.columns:
            if (data['minutes_played'] < 0).any():
                errors.append("Negative minutes played found")
            if (data['minutes_played'] > 48).any():
                errors.append("Minutes played exceeds game length")
        
        if 'field_goal_percentage' in data.columns:
            if (data['field_goal_percentage'] < 0).any() or (data['field_goal_percentage'] > 1).any():
                errors.append("Invalid field goal percentage values")
        
        return errors
    
    def _validate_nfl_data(self, data: pd.DataFrame) -> List[str]:
        """Validate NFL-specific data"""
        errors = []
        
        if 'yards' in data.columns:
            if (data['yards'] < -50).any():
                errors.append("Unrealistic negative yards found")
        
        if 'touchdowns' in data.columns:
            if (data['touchdowns'] < 0).any():
                errors.append("Negative touchdown values found")
        
        return errors
    
    def _validate_soccer_data(self, data: pd.DataFrame) -> List[str]:
        """Validate Soccer-specific data"""
        errors = []
        
        if 'goals' in data.columns:
            if (data['goals'] < 0).any():
                errors.append("Negative goals found")
        
        if 'minutes_played' in data.columns:
            if (data['minutes_played'] > 120).any():
                errors.append("Minutes played exceeds maximum game time")
        
        return errors
    
    def _validate_f1_data(self, data: pd.DataFrame) -> List[str]:
        """Validate Formula 1-specific data"""
        errors = []
        
        if 'lap_time' in data.columns:
            if (data['lap_time'] <= 0).any():
                errors.append("Invalid lap times found")
        
        if 'position' in data.columns:
            if (data['position'] < 1).any() or (data['position'] > 20).any():
                errors.append("Invalid race positions found")
        
        return errors


class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.results: List[ValidationResult] = []
        self.ge_context = None
        self._setup_great_expectations()
    
    def _setup_great_expectations(self):
        """Setup Great Expectations context"""
        try:
            self.ge_context = ge.get_context()
        except Exception as e:
            logger.warning(f"Could not setup Great Expectations: {e}")
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.rules.append(rule)
    
    def add_schema_rule(self, name: str, schema: Dict[str, Any], 
                       severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add JSON schema validation rule"""
        rule = SchemaValidationRule(name, schema, severity)
        self.add_rule(rule)
    
    def add_dataframe_rule(self, name: str, validator_func: Callable[[pd.DataFrame], bool],
                          error_message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add DataFrame validation rule"""
        rule = DataFrameValidationRule(name, validator_func, error_message, severity)
        self.add_rule(rule)
    
    def add_sports_rule(self, name: str, sport: str, data_type: str,
                       severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add sports-specific validation rule"""
        rule = SportsDataValidationRule(name, sport, data_type, severity)
        self.add_rule(rule)
    
    async def validate(self, data: Any, rules: Optional[List[str]] = None) -> List[ValidationResult]:
        """Execute validation rules"""
        self.results = []
        
        # Filter rules if specified
        rules_to_run = self.rules
        if rules:
            rules_to_run = [rule for rule in self.rules if rule.name in rules]
        
        # Run validations concurrently
        tasks = [rule.validate(data) for rule in rules_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ValidationResult):
                self.results.append(result)
            else:
                # Handle exceptions
                error_result = ValidationResult(
                    rule_name="unknown",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation exception: {str(result)}"
                )
                self.results.append(error_result)
        
        return self.results
    
    def validate_with_great_expectations(self, data: pd.DataFrame, 
                                       suite_name: str) -> ValidationResult:
        """Validate using Great Expectations"""
        if not self.ge_context:
            return ValidationResult(
                rule_name="great_expectations",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.WARNING,
                message="Great Expectations not available"
            )
        
        try:
            # Convert DataFrame to Great Expectations dataset
            dataset = ge.from_pandas(data)
            
            # Get expectation suite
            suite = self.ge_context.get_expectation_suite(suite_name)
            
            # Run validation
            validation_result = dataset.validate(expectation_suite=suite)
            
            status = ValidationStatus.PASSED if validation_result.success else ValidationStatus.FAILED
            
            return ValidationResult(
                rule_name="great_expectations",
                status=status,
                severity=ValidationSeverity.ERROR,
                message=f"Great Expectations validation: {validation_result.success}",
                details={
                    'statistics': validation_result.statistics,
                    'results': [r.to_json_dict() for r in validation_result.results]
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_name="great_expectations",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Great Expectations error: {str(e)}"
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        if not self.results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        summary = {
            'total': len(self.results),
            'passed': len([r for r in self.results if r.status == ValidationStatus.PASSED]),
            'failed': len([r for r in self.results if r.status == ValidationStatus.FAILED]),
            'skipped': len([r for r in self.results if r.status == ValidationStatus.SKIPPED]),
            'by_severity': {}
        }
        
        # Group by severity
        for severity in ValidationSeverity:
            summary['by_severity'][severity.value] = len([
                r for r in self.results if r.severity == severity
            ])
        
        return summary
    
    def get_failed_results(self) -> List[ValidationResult]:
        """Get failed validation results"""
        return [r for r in self.results if r.status == ValidationStatus.FAILED]
    
    def export_results(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export validation results"""
        results_data = {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if format == 'json':
            import json
            return json.dumps(results_data, indent=2)
        else:
            return results_data


# Predefined validation rules for common sports data
class CommonSportsValidationRules:
    """Common validation rules for sports data"""
    
    @staticmethod
    def get_player_stats_rules() -> List[ValidationRule]:
        """Get validation rules for player statistics"""
        rules = []
        
        # Basic data quality rules
        rules.append(DataFrameValidationRule(
            "non_empty_data",
            lambda df: len(df) > 0,
            "Dataset is empty"
        ))
        
        rules.append(DataFrameValidationRule(
            "no_duplicate_records",
            lambda df: not df.duplicated().any(),
            "Duplicate records found"
        ))
        
        rules.append(DataFrameValidationRule(
            "required_columns",
            lambda df: all(col in df.columns for col in ['player_id', 'game_id', 'date']),
            "Required columns missing"
        ))
        
        return rules
    
    @staticmethod
    def get_game_data_rules() -> List[ValidationRule]:
        """Get validation rules for game data"""
        rules = []
        
        rules.append(DataFrameValidationRule(
            "valid_scores",
            lambda df: (df['home_score'] >= 0).all() and (df['away_score'] >= 0).all() if 'home_score' in df.columns and 'away_score' in df.columns else True,
            "Invalid game scores found"
        ))
        
        rules.append(DataFrameValidationRule(
            "valid_dates",
            lambda df: pd.to_datetime(df['date'], errors='coerce').notna().all() if 'date' in df.columns else True,
            "Invalid date values found"
        ))
        
        return rules 