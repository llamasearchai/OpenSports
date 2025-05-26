"""
Advanced Performance Profiler

Comprehensive performance profiling and optimization tools for the OpenSports platform
with CPU, memory, I/O profiling, and performance bottleneck detection.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import time
import cProfile
import pstats
import tracemalloc
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import numpy as np
import pandas as pd
from opensports.core.config import settings
from opensports.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileResult:
    """Result of a performance profiling session."""
    name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_stats: Dict[str, Any] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    io_stats: Dict[str, Any] = field(default_factory=dict)
    function_stats: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetric:
    """A single performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Advanced performance profiler.
    
    Features:
    - CPU profiling with function-level analysis
    - Memory profiling and leak detection
    - I/O performance monitoring
    - Async operation profiling
    - Performance bottleneck identification
    - Optimization recommendations
    - Real-time performance monitoring
    """
    
    def __init__(self):
        self.active_profiles = {}
        self.profile_history = []
        self.performance_metrics = []
        self.monitoring_enabled = False
        self.monitoring_interval = 1.0  # seconds
        self.baseline_metrics = {}
        
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        logger.info("Starting performance monitoring")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self.monitoring_enabled = False
        logger.info("Stopping performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Store metrics
            metrics = [
                PerformanceMetric("cpu_percent", cpu_percent, "%", timestamp),
                PerformanceMetric("cpu_count", cpu_count, "cores", timestamp),
                PerformanceMetric("load_avg_1m", load_avg[0], "load", timestamp),
                PerformanceMetric("memory_percent", memory.percent, "%", timestamp),
                PerformanceMetric("memory_available", memory.available / (1024**3), "GB", timestamp),
                PerformanceMetric("swap_percent", swap.percent, "%", timestamp),
                PerformanceMetric("process_memory_rss", process_memory.rss / (1024**2), "MB", timestamp),
                PerformanceMetric("process_memory_vms", process_memory.vms / (1024**2), "MB", timestamp),
                PerformanceMetric("process_cpu_percent", process_cpu, "%", timestamp),
            ]
            
            if disk_io:
                metrics.extend([
                    PerformanceMetric("disk_read_bytes", disk_io.read_bytes, "bytes", timestamp),
                    PerformanceMetric("disk_write_bytes", disk_io.write_bytes, "bytes", timestamp),
                    PerformanceMetric("disk_read_count", disk_io.read_count, "ops", timestamp),
                    PerformanceMetric("disk_write_count", disk_io.write_count, "ops", timestamp),
                ])
            
            if network_io:
                metrics.extend([
                    PerformanceMetric("network_bytes_sent", network_io.bytes_sent, "bytes", timestamp),
                    PerformanceMetric("network_bytes_recv", network_io.bytes_recv, "bytes", timestamp),
                    PerformanceMetric("network_packets_sent", network_io.packets_sent, "packets", timestamp),
                    PerformanceMetric("network_packets_recv", network_io.packets_recv, "packets", timestamp),
                ])
            
            self.performance_metrics.extend(metrics)
            
            # Keep only recent metrics (last hour)
            cutoff_time = timestamp - timedelta(hours=1)
            self.performance_metrics = [
                m for m in self.performance_metrics if m.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    @contextmanager
    def profile_cpu(self, name: str):
        """Context manager for CPU profiling."""
        profiler = cProfile.Profile()
        start_time = datetime.now()
        
        logger.info(f"Starting CPU profiling: {name}")
        profiler.enable()
        
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = datetime.now()
            
            # Analyze results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # Extract function statistics
            function_stats = []
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_number, function_name = func_info
                function_stats.append({
                    'function': function_name,
                    'filename': filename,
                    'line_number': line_number,
                    'call_count': cc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_per_call': tt / cc if cc > 0 else 0
                })
            
            # Sort by cumulative time
            function_stats.sort(key=lambda x: x['cumulative_time'], reverse=True)
            
            # Create profile result
            result = ProfileResult(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                cpu_stats={
                    'total_calls': stats.total_calls,
                    'primitive_calls': stats.prim_calls,
                    'total_time': stats.total_tt
                },
                function_stats=function_stats[:50]  # Top 50 functions
            )
            
            # Identify bottlenecks
            result.bottlenecks = self._identify_cpu_bottlenecks(function_stats)
            result.recommendations = self._generate_cpu_recommendations(result)
            
            self.profile_history.append(result)
            logger.info(f"CPU profiling completed: {name} ({result.duration_seconds:.3f}s)")
    
    @contextmanager
    def profile_memory(self, name: str):
        """Context manager for memory profiling."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            started_tracing = True
        else:
            started_tracing = False
        
        start_time = datetime.now()
        start_snapshot = tracemalloc.take_snapshot()
        
        logger.info(f"Starting memory profiling: {name}")
        
        try:
            yield
        finally:
            end_time = datetime.now()
            end_snapshot = tracemalloc.take_snapshot()
            
            if started_tracing:
                tracemalloc.stop()
            
            # Analyze memory usage
            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            
            memory_stats = []
            total_size_diff = 0
            
            for stat in top_stats[:20]:  # Top 20 memory allocations
                memory_stats.append({
                    'filename': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                    'size_diff': stat.size_diff,
                    'size_diff_mb': stat.size_diff / (1024 * 1024),
                    'count_diff': stat.count_diff
                })
                total_size_diff += stat.size_diff
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Create profile result
            result = ProfileResult(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                memory_stats={
                    'total_size_diff': total_size_diff,
                    'total_size_diff_mb': total_size_diff / (1024 * 1024),
                    'current_rss': memory_info.rss,
                    'current_rss_mb': memory_info.rss / (1024 * 1024),
                    'current_vms': memory_info.vms,
                    'current_vms_mb': memory_info.vms / (1024 * 1024),
                    'allocations': memory_stats
                }
            )
            
            # Identify memory bottlenecks
            result.bottlenecks = self._identify_memory_bottlenecks(memory_stats)
            result.recommendations = self._generate_memory_recommendations(result)
            
            self.profile_history.append(result)
            logger.info(f"Memory profiling completed: {name} ({result.memory_stats['total_size_diff_mb']:.2f}MB diff)")
    
    async def profile_async_function(self, func: Callable, *args, **kwargs) -> ProfileResult:
        """Profile an async function."""
        name = f"{func.__module__}.{func.__name__}"
        start_time = datetime.now()
        
        # Start monitoring
        start_metrics = self._get_current_metrics()
        
        logger.info(f"Starting async function profiling: {name}")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            end_time = datetime.now()
            end_metrics = self._get_current_metrics()
            
            # Calculate differences
            metric_diffs = {}
            for key in start_metrics:
                if key in end_metrics:
                    metric_diffs[key] = end_metrics[key] - start_metrics[key]
            
            # Create profile result
            profile_result = ProfileResult(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                cpu_stats=metric_diffs,
                function_stats=[{
                    'function': name,
                    'duration': (end_time - start_time).total_seconds(),
                    'result_type': type(result).__name__ if result is not None else 'None'
                }]
            )
            
            # Generate recommendations
            profile_result.recommendations = self._generate_async_recommendations(profile_result)
            
            self.profile_history.append(profile_result)
            logger.info(f"Async function profiling completed: {name} ({profile_result.duration_seconds:.3f}s)")
            
            return profile_result
            
        except Exception as e:
            end_time = datetime.now()
            logger.error(f"Error during async function profiling: {e}")
            
            # Create error result
            profile_result = ProfileResult(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                function_stats=[{
                    'function': name,
                    'error': str(e),
                    'duration': (end_time - start_time).total_seconds()
                }]
            )
            
            self.profile_history.append(profile_result)
            raise
    
    def profile_decorator(self, profile_type: str = 'cpu'):
        """Decorator for automatic function profiling."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                name = f"{func.__module__}.{func.__name__}"
                
                if profile_type == 'cpu':
                    with self.profile_cpu(name):
                        return func(*args, **kwargs)
                elif profile_type == 'memory':
                    with self.profile_memory(name):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            async def async_wrapper(*args, **kwargs):
                if asyncio.iscoroutinefunction(func):
                    return await self.profile_async_function(func, *args, **kwargs)
                else:
                    return wrapper(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        
        return decorator
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_rss': memory_info.rss,
                'memory_vms': memory_info.vms,
                'process_cpu': process.cpu_percent(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def _identify_cpu_bottlenecks(self, function_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify CPU performance bottlenecks."""
        bottlenecks = []
        
        if not function_stats:
            return bottlenecks
        
        # Find functions with high cumulative time
        total_time = sum(stat['cumulative_time'] for stat in function_stats)
        
        for stat in function_stats[:10]:  # Top 10 functions
            if stat['cumulative_time'] > total_time * 0.05:  # More than 5% of total time
                bottlenecks.append({
                    'type': 'cpu_hotspot',
                    'function': stat['function'],
                    'cumulative_time': stat['cumulative_time'],
                    'percentage': (stat['cumulative_time'] / total_time) * 100,
                    'severity': 'high' if stat['cumulative_time'] > total_time * 0.2 else 'medium'
                })
        
        # Find functions with many calls
        for stat in function_stats:
            if stat['call_count'] > 10000:  # More than 10k calls
                bottlenecks.append({
                    'type': 'high_call_count',
                    'function': stat['function'],
                    'call_count': stat['call_count'],
                    'time_per_call': stat['time_per_call'],
                    'severity': 'medium'
                })
        
        return bottlenecks
    
    def _identify_memory_bottlenecks(self, memory_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify memory performance bottlenecks."""
        bottlenecks = []
        
        for stat in memory_stats:
            if stat['size_diff_mb'] > 10:  # More than 10MB allocation
                bottlenecks.append({
                    'type': 'large_allocation',
                    'filename': stat['filename'],
                    'size_diff_mb': stat['size_diff_mb'],
                    'count_diff': stat['count_diff'],
                    'severity': 'high' if stat['size_diff_mb'] > 50 else 'medium'
                })
        
        return bottlenecks
    
    def _generate_cpu_recommendations(self, result: ProfileResult) -> List[str]:
        """Generate CPU optimization recommendations."""
        recommendations = []
        
        for bottleneck in result.bottlenecks:
            if bottleneck['type'] == 'cpu_hotspot':
                recommendations.append(
                    f"Optimize function '{bottleneck['function']}' which consumes "
                    f"{bottleneck['percentage']:.1f}% of total CPU time"
                )
            elif bottleneck['type'] == 'high_call_count':
                recommendations.append(
                    f"Consider caching or reducing calls to '{bottleneck['function']}' "
                    f"({bottleneck['call_count']} calls)"
                )
        
        # General recommendations
        if result.duration_seconds > 5:
            recommendations.append("Consider breaking down long-running operations into smaller chunks")
        
        if len(result.function_stats) > 100:
            recommendations.append("High function call overhead detected - consider code optimization")
        
        return recommendations
    
    def _generate_memory_recommendations(self, result: ProfileResult) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        total_diff_mb = result.memory_stats.get('total_size_diff_mb', 0)
        
        if total_diff_mb > 100:
            recommendations.append(f"High memory allocation detected ({total_diff_mb:.1f}MB)")
        
        for bottleneck in result.bottlenecks:
            if bottleneck['type'] == 'large_allocation':
                recommendations.append(
                    f"Large memory allocation in {bottleneck['filename']} "
                    f"({bottleneck['size_diff_mb']:.1f}MB)"
                )
        
        # Check for potential memory leaks
        if total_diff_mb > 0 and result.duration_seconds < 10:
            recommendations.append("Potential memory leak detected - monitor for memory growth")
        
        return recommendations
    
    def _generate_async_recommendations(self, result: ProfileResult) -> List[str]:
        """Generate async function optimization recommendations."""
        recommendations = []
        
        if result.duration_seconds > 1:
            recommendations.append("Long-running async function - consider adding progress tracking")
        
        if result.duration_seconds > 10:
            recommendations.append("Very long async operation - consider breaking into smaller tasks")
        
        return recommendations
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_metrics = [m for m in self.performance_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No performance data available"}
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in recent_metrics:
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for name, values in metrics_by_name.items():
            summary[name] = {
                'current': values[-1] if values else 0,
                'average': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'count': len(values)
            }
        
        # Add profile summary
        recent_profiles = [p for p in self.profile_history if p.start_time >= cutoff_time]
        summary['profiling'] = {
            'total_profiles': len(recent_profiles),
            'avg_duration': np.mean([p.duration_seconds for p in recent_profiles]) if recent_profiles else 0,
            'total_bottlenecks': sum(len(p.bottlenecks) for p in recent_profiles)
        }
        
        return summary
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Get comprehensive bottleneck analysis."""
        all_bottlenecks = []
        for profile in self.profile_history:
            all_bottlenecks.extend(profile.bottlenecks)
        
        if not all_bottlenecks:
            return {"message": "No bottlenecks detected"}
        
        # Group by type
        bottlenecks_by_type = {}
        for bottleneck in all_bottlenecks:
            btype = bottleneck['type']
            if btype not in bottlenecks_by_type:
                bottlenecks_by_type[btype] = []
            bottlenecks_by_type[btype].append(bottleneck)
        
        # Analyze patterns
        analysis = {}
        for btype, bottlenecks in bottlenecks_by_type.items():
            analysis[btype] = {
                'count': len(bottlenecks),
                'severity_distribution': {
                    'high': sum(1 for b in bottlenecks if b.get('severity') == 'high'),
                    'medium': sum(1 for b in bottlenecks if b.get('severity') == 'medium'),
                    'low': sum(1 for b in bottlenecks if b.get('severity') == 'low')
                }
            }
            
            # Add type-specific analysis
            if btype == 'cpu_hotspot':
                functions = [b['function'] for b in bottlenecks]
                analysis[btype]['most_common_functions'] = list(set(functions))
            elif btype == 'large_allocation':
                files = [b['filename'] for b in bottlenecks]
                analysis[btype]['most_common_files'] = list(set(files))
        
        return analysis
    
    def export_profile_data(self, filename: str = None) -> str:
        """Export profile data to JSON file."""
        if filename is None:
            filename = f"opensports_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for export
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'bottleneck_analysis': self.get_bottleneck_analysis(),
            'profile_history': []
        }
        
        # Add profile history (last 100 profiles)
        for profile in self.profile_history[-100:]:
            export_data['profile_history'].append({
                'name': profile.name,
                'start_time': profile.start_time.isoformat(),
                'end_time': profile.end_time.isoformat(),
                'duration_seconds': profile.duration_seconds,
                'cpu_stats': profile.cpu_stats,
                'memory_stats': profile.memory_stats,
                'bottlenecks': profile.bottlenecks,
                'recommendations': profile.recommendations
            })
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Profile data exported to {filename}")
        return filename
    
    def clear_history(self, older_than_hours: int = 24):
        """Clear old profile history."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        initial_count = len(self.profile_history)
        self.profile_history = [p for p in self.profile_history if p.start_time >= cutoff_time]
        
        cleared_count = initial_count - len(self.profile_history)
        logger.info(f"Cleared {cleared_count} old profile records")
        
        # Also clear old metrics
        initial_metrics_count = len(self.performance_metrics)
        self.performance_metrics = [m for m in self.performance_metrics if m.timestamp >= cutoff_time]
        
        cleared_metrics_count = initial_metrics_count - len(self.performance_metrics)
        logger.info(f"Cleared {cleared_metrics_count} old performance metrics") 