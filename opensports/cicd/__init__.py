"""
OpenSports CI/CD Module

Comprehensive continuous integration and deployment system
with automated testing, quality gates, and deployment pipelines.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

from .pipeline import CICDPipeline, PipelineStage, PipelineResult
from .testing import TestRunner, TestSuite, TestResult
from .quality import QualityGate, CodeQualityAnalyzer, SecurityScanner
from .deployment import DeploymentManager, EnvironmentManager, RollbackManager
from .monitoring import PipelineMonitor, DeploymentMonitor, AlertManager
from .artifacts import ArtifactManager, BuildArtifact, DeploymentArtifact

__all__ = [
    'CICDPipeline',
    'PipelineStage',
    'PipelineResult',
    'TestRunner',
    'TestSuite',
    'TestResult',
    'QualityGate',
    'CodeQualityAnalyzer',
    'SecurityScanner',
    'DeploymentManager',
    'EnvironmentManager',
    'RollbackManager',
    'PipelineMonitor',
    'DeploymentMonitor',
    'AlertManager',
    'ArtifactManager',
    'BuildArtifact',
    'DeploymentArtifact'
]

__version__ = "1.0.0" 