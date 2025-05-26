"""
Test Suite Runners

Comprehensive test execution framework for the OpenSports platform with
parallel execution, coverage reporting, and continuous testing capabilities.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import pytest
import coverage
import unittest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from opensports.core.config import settings
from opensports.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Result of a test execution."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    message: Optional[str] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Result of a test suite execution."""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    coverage_percentage: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class TestSuiteRunner:
    """
    Comprehensive test suite runner.
    
    Features:
    - Parallel test execution
    - Coverage reporting
    - Performance metrics
    - Multiple test frameworks support
    - Detailed reporting
    - Test discovery and filtering
    """
    
    def __init__(self):
        self.test_results = []
        self.coverage_data = None
        self.parallel_workers = 4
        self.test_timeout = 300  # 5 minutes
        
    async def discover_tests(self, test_paths: List[str] = None) -> List[str]:
        """Discover all test files in the project."""
        if test_paths is None:
            test_paths = ["tests/", "opensports/"]
        
        test_files = []
        
        for path in test_paths:
            path_obj = Path(path)
            if path_obj.exists():
                # Find Python test files
                test_files.extend(path_obj.rglob("test_*.py"))
                test_files.extend(path_obj.rglob("*_test.py"))
        
        logger.info(f"Discovered {len(test_files)} test files")
        return [str(f) for f in test_files]
    
    async def run_unit_tests(self, test_pattern: str = "test_*.py") -> TestSuiteResult:
        """Run unit tests using pytest."""
        logger.info("Running unit tests")
        start_time = datetime.now()
        
        try:
            # Configure pytest arguments
            pytest_args = [
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                f"--maxfail=10",  # Stop after 10 failures
                "--durations=10",  # Show 10 slowest tests
                "--junit-xml=test-results/unit-tests.xml",
                "--cov=opensports",
                "--cov-report=xml:test-results/coverage.xml",
                "--cov-report=html:test-results/htmlcov",
                "--cov-report=term-missing",
                f"-k {test_pattern}" if test_pattern != "test_*.py" else "",
                "tests/unit/"
            ]
            
            # Remove empty arguments
            pytest_args = [arg for arg in pytest_args if arg]
            
            # Run pytest
            result = pytest.main(pytest_args)
            
            end_time = datetime.now()
            
            # Parse results
            suite_result = await self._parse_junit_results(
                "test-results/unit-tests.xml",
                "Unit Tests",
                start_time,
                end_time
            )
            
            # Add coverage data
            suite_result.coverage_percentage = await self._get_coverage_percentage()
            
            logger.info(f"Unit tests completed: {suite_result.passed}/{suite_result.total_tests} passed")
            return suite_result
            
        except Exception as e:
            logger.error(f"Failed to run unit tests: {e}")
            return TestSuiteResult(
                suite_name="Unit Tests",
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1
            )
    
    async def run_integration_tests(self) -> TestSuiteResult:
        """Run integration tests."""
        logger.info("Running integration tests")
        start_time = datetime.now()
        
        try:
            pytest_args = [
                "-v",
                "--tb=short",
                "--maxfail=5",
                "--junit-xml=test-results/integration-tests.xml",
                "tests/integration/"
            ]
            
            result = pytest.main(pytest_args)
            end_time = datetime.now()
            
            suite_result = await self._parse_junit_results(
                "test-results/integration-tests.xml",
                "Integration Tests",
                start_time,
                end_time
            )
            
            logger.info(f"Integration tests completed: {suite_result.passed}/{suite_result.total_tests} passed")
            return suite_result
            
        except Exception as e:
            logger.error(f"Failed to run integration tests: {e}")
            return TestSuiteResult(
                suite_name="Integration Tests",
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1
            )
    
    async def run_performance_tests(self) -> TestSuiteResult:
        """Run performance tests."""
        logger.info("Running performance tests")
        start_time = datetime.now()
        
        try:
            # Use pytest-benchmark for performance tests
            pytest_args = [
                "-v",
                "--benchmark-only",
                "--benchmark-json=test-results/benchmark.json",
                "--junit-xml=test-results/performance-tests.xml",
                "tests/performance/"
            ]
            
            result = pytest.main(pytest_args)
            end_time = datetime.now()
            
            suite_result = await self._parse_junit_results(
                "test-results/performance-tests.xml",
                "Performance Tests",
                start_time,
                end_time
            )
            
            # Add benchmark data
            suite_result.performance_metrics = await self._parse_benchmark_results()
            
            logger.info(f"Performance tests completed: {suite_result.passed}/{suite_result.total_tests} passed")
            return suite_result
            
        except Exception as e:
            logger.error(f"Failed to run performance tests: {e}")
            return TestSuiteResult(
                suite_name="Performance Tests",
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1
            )
    
    async def run_security_tests(self) -> TestSuiteResult:
        """Run security tests."""
        logger.info("Running security tests")
        start_time = datetime.now()
        
        try:
            # Run security-specific tests
            pytest_args = [
                "-v",
                "--tb=short",
                "--junit-xml=test-results/security-tests.xml",
                "tests/security/"
            ]
            
            result = pytest.main(pytest_args)
            
            # Also run bandit for security linting
            await self._run_bandit_scan()
            
            end_time = datetime.now()
            
            suite_result = await self._parse_junit_results(
                "test-results/security-tests.xml",
                "Security Tests",
                start_time,
                end_time
            )
            
            logger.info(f"Security tests completed: {suite_result.passed}/{suite_result.total_tests} passed")
            return suite_result
            
        except Exception as e:
            logger.error(f"Failed to run security tests: {e}")
            return TestSuiteResult(
                suite_name="Security Tests",
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1
            )
    
    async def run_all_tests(self, parallel: bool = True) -> List[TestSuiteResult]:
        """Run all test suites."""
        logger.info("Running all test suites")
        
        # Ensure test results directory exists
        Path("test-results").mkdir(exist_ok=True)
        
        if parallel:
            # Run test suites in parallel
            tasks = [
                self.run_unit_tests(),
                self.run_integration_tests(),
                self.run_performance_tests(),
                self.run_security_tests()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            suite_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test suite {i} failed: {result}")
                    suite_results.append(TestSuiteResult(
                        suite_name=f"Test Suite {i}",
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        total_tests=0,
                        passed=0,
                        failed=1,
                        skipped=0,
                        errors=1
                    ))
                else:
                    suite_results.append(result)
        else:
            # Run test suites sequentially
            suite_results = [
                await self.run_unit_tests(),
                await self.run_integration_tests(),
                await self.run_performance_tests(),
                await self.run_security_tests()
            ]
        
        # Generate summary report
        await self._generate_summary_report(suite_results)
        
        return suite_results
    
    async def run_custom_test_suite(self, test_files: List[str], suite_name: str) -> TestSuiteResult:
        """Run a custom test suite with specified files."""
        logger.info(f"Running custom test suite: {suite_name}")
        start_time = datetime.now()
        
        try:
            pytest_args = [
                "-v",
                "--tb=short",
                f"--junit-xml=test-results/{suite_name.lower().replace(' ', '-')}.xml"
            ] + test_files
            
            result = pytest.main(pytest_args)
            end_time = datetime.now()
            
            suite_result = await self._parse_junit_results(
                f"test-results/{suite_name.lower().replace(' ', '-')}.xml",
                suite_name,
                start_time,
                end_time
            )
            
            logger.info(f"{suite_name} completed: {suite_result.passed}/{suite_result.total_tests} passed")
            return suite_result
            
        except Exception as e:
            logger.error(f"Failed to run {suite_name}: {e}")
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1
            )
    
    async def _parse_junit_results(self, xml_file: str, suite_name: str, start_time: datetime, end_time: datetime) -> TestSuiteResult:
        """Parse JUnit XML results."""
        try:
            if not Path(xml_file).exists():
                logger.warning(f"JUnit XML file not found: {xml_file}")
                return TestSuiteResult(
                    suite_name=suite_name,
                    start_time=start_time,
                    end_time=end_time,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=0
                )
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Parse testsuite attributes
            total_tests = int(root.get('tests', 0))
            failures = int(root.get('failures', 0))
            errors = int(root.get('errors', 0))
            skipped = int(root.get('skipped', 0))
            passed = total_tests - failures - errors - skipped
            
            # Parse individual test results
            test_results = []
            for testcase in root.findall('.//testcase'):
                test_name = f"{testcase.get('classname', '')}.{testcase.get('name', '')}"
                duration = float(testcase.get('time', 0))
                
                # Determine test status
                if testcase.find('failure') is not None:
                    status = 'failed'
                    failure_elem = testcase.find('failure')
                    message = failure_elem.get('message', '')
                    traceback = failure_elem.text
                elif testcase.find('error') is not None:
                    status = 'error'
                    error_elem = testcase.find('error')
                    message = error_elem.get('message', '')
                    traceback = error_elem.text
                elif testcase.find('skipped') is not None:
                    status = 'skipped'
                    skipped_elem = testcase.find('skipped')
                    message = skipped_elem.get('message', '')
                    traceback = None
                else:
                    status = 'passed'
                    message = None
                    traceback = None
                
                test_results.append(TestResult(
                    test_name=test_name,
                    status=status,
                    duration=duration,
                    message=message,
                    traceback=traceback
                ))
            
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                total_tests=total_tests,
                passed=passed,
                failed=failures,
                skipped=skipped,
                errors=errors,
                test_results=test_results
            )
            
        except Exception as e:
            logger.error(f"Failed to parse JUnit results: {e}")
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=0
            )
    
    async def _get_coverage_percentage(self) -> float:
        """Get code coverage percentage."""
        try:
            coverage_file = Path("test-results/coverage.xml")
            if not coverage_file.exists():
                return 0.0
            
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Find coverage percentage
            coverage_elem = root.find('.//coverage')
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get('line-rate', 0))
                return line_rate * 100
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get coverage percentage: {e}")
            return 0.0
    
    async def _parse_benchmark_results(self) -> Dict[str, Any]:
        """Parse pytest-benchmark results."""
        try:
            benchmark_file = Path("test-results/benchmark.json")
            if not benchmark_file.exists():
                return {}
            
            with open(benchmark_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            benchmarks = data.get('benchmarks', [])
            metrics = {
                'total_benchmarks': len(benchmarks),
                'fastest_test': None,
                'slowest_test': None,
                'average_time': 0
            }
            
            if benchmarks:
                times = [b['stats']['mean'] for b in benchmarks]
                metrics['average_time'] = sum(times) / len(times)
                
                fastest = min(benchmarks, key=lambda x: x['stats']['mean'])
                slowest = max(benchmarks, key=lambda x: x['stats']['mean'])
                
                metrics['fastest_test'] = {
                    'name': fastest['name'],
                    'time': fastest['stats']['mean']
                }
                metrics['slowest_test'] = {
                    'name': slowest['name'],
                    'time': slowest['stats']['mean']
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to parse benchmark results: {e}")
            return {}
    
    async def _run_bandit_scan(self):
        """Run Bandit security scan."""
        try:
            result = subprocess.run([
                'bandit', '-r', 'opensports/', '-f', 'json', '-o', 'test-results/bandit.json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Bandit security scan completed successfully")
            else:
                logger.warning(f"Bandit scan found issues: {result.stderr}")
                
        except FileNotFoundError:
            logger.warning("Bandit not found, skipping security scan")
        except Exception as e:
            logger.error(f"Failed to run Bandit scan: {e}")
    
    async def _generate_summary_report(self, suite_results: List[TestSuiteResult]):
        """Generate a comprehensive test summary report."""
        try:
            total_tests = sum(r.total_tests for r in suite_results)
            total_passed = sum(r.passed for r in suite_results)
            total_failed = sum(r.failed for r in suite_results)
            total_skipped = sum(r.skipped for r in suite_results)
            total_errors = sum(r.errors for r in suite_results)
            
            # Calculate overall success rate
            success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            
            # Generate HTML report
            html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OpenSports Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 10px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ background-color: #d4edda; }}
        .failed {{ background-color: #f8d7da; }}
        .skipped {{ background-color: #fff3cd; }}
        .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .suite-header {{ font-weight: bold; font-size: 18px; margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OpenSports Test Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="metric passed">
            <h3>{total_passed}</h3>
            <p>Passed</p>
        </div>
        <div class="metric failed">
            <h3>{total_failed}</h3>
            <p>Failed</p>
        </div>
        <div class="metric skipped">
            <h3>{total_skipped}</h3>
            <p>Skipped</p>
        </div>
        <div class="metric">
            <h3>{success_rate:.1f}%</h3>
            <p>Success Rate</p>
        </div>
    </div>
"""
            
            # Add suite details
            for suite in suite_results:
                duration = (suite.end_time - suite.start_time).total_seconds()
                html_report += f"""
    <div class="suite">
        <div class="suite-header">{suite.suite_name}</div>
        <p>Duration: {duration:.2f}s | Tests: {suite.total_tests} | Passed: {suite.passed} | Failed: {suite.failed}</p>
        
        <table>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Message</th>
            </tr>
"""
                
                for test in suite.test_results[:10]:  # Show first 10 tests
                    status_class = test.status
                    html_report += f"""
            <tr class="{status_class}">
                <td>{test.test_name}</td>
                <td>{test.status.upper()}</td>
                <td>{test.duration:.3f}s</td>
                <td>{test.message or ''}</td>
            </tr>
"""
                
                html_report += """
        </table>
    </div>
"""
            
            html_report += """
</body>
</html>
"""
            
            # Write HTML report
            with open("test-results/test-report.html", 'w') as f:
                f.write(html_report)
            
            # Generate JSON summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'skipped': total_skipped,
                'errors': total_errors,
                'success_rate': success_rate,
                'suites': [
                    {
                        'name': suite.suite_name,
                        'total_tests': suite.total_tests,
                        'passed': suite.passed,
                        'failed': suite.failed,
                        'skipped': suite.skipped,
                        'errors': suite.errors,
                        'duration': (suite.end_time - suite.start_time).total_seconds(),
                        'coverage': suite.coverage_percentage
                    }
                    for suite in suite_results
                ]
            }
            
            with open("test-results/test-summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Test summary report generated: {total_passed}/{total_tests} tests passed ({success_rate:.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")


class ContinuousTestRunner:
    """
    Continuous testing runner for development and CI/CD.
    
    Features:
    - File watching for automatic test execution
    - Incremental testing
    - Test result caching
    - Integration with CI/CD pipelines
    """
    
    def __init__(self):
        self.test_runner = TestSuiteRunner()
        self.is_running = False
        self.last_run_time = None
        self.file_watcher = None
        
    async def start_continuous_testing(self, watch_paths: List[str] = None):
        """Start continuous testing with file watching."""
        if watch_paths is None:
            watch_paths = ["opensports/", "tests/"]
        
        self.is_running = True
        logger.info("Starting continuous testing")
        
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class TestEventHandler(FileSystemEventHandler):
                def __init__(self, runner):
                    self.runner = runner
                
                def on_modified(self, event):
                    if event.is_directory:
                        return
                    
                    if event.src_path.endswith('.py'):
                        asyncio.create_task(self.runner._run_incremental_tests(event.src_path))
            
            # Set up file watcher
            event_handler = TestEventHandler(self)
            observer = Observer()
            
            for path in watch_paths:
                if Path(path).exists():
                    observer.schedule(event_handler, path, recursive=True)
            
            observer.start()
            self.file_watcher = observer
            
            # Keep running
            while self.is_running:
                await asyncio.sleep(1)
                
        except ImportError:
            logger.warning("Watchdog not available, falling back to polling")
            await self._polling_continuous_testing(watch_paths)
        except Exception as e:
            logger.error(f"Continuous testing error: {e}")
        finally:
            if self.file_watcher:
                self.file_watcher.stop()
                self.file_watcher.join()
    
    async def stop_continuous_testing(self):
        """Stop continuous testing."""
        self.is_running = False
        if self.file_watcher:
            self.file_watcher.stop()
        logger.info("Stopped continuous testing")
    
    async def _run_incremental_tests(self, changed_file: str):
        """Run tests related to the changed file."""
        logger.info(f"File changed: {changed_file}, running related tests")
        
        try:
            # Determine which tests to run based on the changed file
            if "test" in changed_file:
                # If a test file changed, run that specific test
                await self.test_runner.run_custom_test_suite([changed_file], "Incremental Tests")
            else:
                # If source code changed, run related unit tests
                await self.test_runner.run_unit_tests(f"*{Path(changed_file).stem}*")
                
        except Exception as e:
            logger.error(f"Failed to run incremental tests: {e}")
    
    async def _polling_continuous_testing(self, watch_paths: List[str]):
        """Fallback polling-based continuous testing."""
        last_modified_times = {}
        
        while self.is_running:
            try:
                for path in watch_paths:
                    path_obj = Path(path)
                    if path_obj.exists():
                        for py_file in path_obj.rglob("*.py"):
                            mtime = py_file.stat().st_mtime
                            
                            if str(py_file) not in last_modified_times:
                                last_modified_times[str(py_file)] = mtime
                            elif mtime > last_modified_times[str(py_file)]:
                                last_modified_times[str(py_file)] = mtime
                                await self._run_incremental_tests(str(py_file))
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(10) 