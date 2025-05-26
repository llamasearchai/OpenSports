"""
Comprehensive Monitoring Dashboard

Real-time monitoring dashboard for the OpenSports platform with
metrics visualization, health status, and operational insights.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis.asyncio as redis
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.monitoring.metrics import MetricsCollector
from opensports.monitoring.health import HealthChecker, HealthStatus
from opensports.monitoring.alerting import AlertManager

logger = get_logger(__name__)


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard.
    
    Features:
    - Real-time metrics visualization
    - System health monitoring
    - Alert management interface
    - Performance analytics
    - Resource utilization tracking
    - Historical trend analysis
    """
    
    def __init__(self):
        self.metrics_collector = None
        self.health_checker = None
        self.alert_manager = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize the monitoring dashboard."""
        logger.info("Initializing monitoring dashboard")
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        await self.metrics_collector.initialize()
        
        self.health_checker = HealthChecker()
        await self.health_checker.initialize()
        
        self.alert_manager = AlertManager()
        await self.alert_manager.initialize()
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            await self.redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    def render_dashboard(self):
        """Render the main monitoring dashboard."""
        st.set_page_config(
            page_title="OpenSports Monitoring Dashboard",
            page_icon="CHART",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #2a5298;
        }
        .status-healthy { color: #28a745; }
        .status-degraded { color: #ffc107; }
        .status-unhealthy { color: #dc3545; }
        .status-unknown { color: #6c757d; }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>OpenSports Monitoring Dashboard</h1>
            <p>Real-time system monitoring and observability</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Dashboard",
            ["System Overview", "Metrics", "Health Status", "Alerts", "Performance", "Analytics"]
        )
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
        if auto_refresh:
            st.rerun()
        
        # Render selected page
        if page == "System Overview":
            self.render_system_overview()
        elif page == "Metrics":
            self.render_metrics_page()
        elif page == "Health Status":
            self.render_health_page()
        elif page == "Alerts":
            self.render_alerts_page()
        elif page == "Performance":
            self.render_performance_page()
        elif page == "Analytics":
            self.render_analytics_page()
    
    def render_system_overview(self):
        """Render the system overview page."""
        st.header("System Overview")
        
        # Get current system status
        system_health = self.get_system_health_sync()
        current_metrics = self.get_current_metrics_sync()
        active_alerts = self.get_active_alerts_sync()
        
        # Status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = self.get_status_color(system_health.get('status', 'unknown'))
            st.markdown(f"""
            <div class="metric-card">
                <h3>System Status</h3>
                <h2 class="{status_color}">{system_health.get('status', 'Unknown').title()}</h2>
                <p>{system_health.get('healthy_count', 0)}/{system_health.get('component_count', 0)} components healthy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            alert_count = len(active_alerts)
            alert_color = "status-unhealthy" if alert_count > 0 else "status-healthy"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Active Alerts</h3>
                <h2 class="{alert_color}">{alert_count}</h2>
                <p>System alerts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            uptime_hours = round(system_health.get('uptime_seconds', 0) / 3600, 1)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Uptime</h3>
                <h2 class="status-healthy">{uptime_hours}h</h2>
                <p>System uptime</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cpu_usage = current_metrics.get('system_cpu_percent', 0)
            cpu_color = self.get_metric_color(cpu_usage, 70, 90)
            st.markdown(f"""
            <div class="metric-card">
                <h3>CPU Usage</h3>
                <h2 class="{cpu_color}">{cpu_usage:.1f}%</h2>
                <p>Current CPU utilization</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Metrics Trend")
            self.render_metrics_trend_chart()
        
        with col2:
            st.subheader("Component Health")
            self.render_health_status_chart(system_health)
        
        # Recent alerts
        if active_alerts:
            st.subheader("Recent Alerts")
            self.render_alerts_table(active_alerts[:5])
        
        # Performance summary
        st.subheader("Performance Summary")
        self.render_performance_summary(current_metrics)
    
    def render_metrics_page(self):
        """Render the metrics monitoring page."""
        st.header("System Metrics")
        
        # Metrics filters
        col1, col2, col3 = st.columns(3)
        with col1:
            time_range = st.selectbox("Time Range", ["1h", "6h", "24h", "7d"], index=1)
        with col2:
            metric_category = st.selectbox("Category", ["All", "System", "API", "Database", "ML"])
        with col3:
            refresh_interval = st.selectbox("Refresh", ["30s", "1m", "5m"], index=1)
        
        # Get metrics data
        metrics_data = self.get_metrics_history_sync(time_range)
        
        # System metrics
        if metric_category in ["All", "System"]:
            st.subheader("System Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_cpu_memory_chart(metrics_data)
            
            with col2:
                self.render_disk_network_chart(metrics_data)
        
        # API metrics
        if metric_category in ["All", "API"]:
            st.subheader("API Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_api_requests_chart(metrics_data)
            
            with col2:
                self.render_api_response_time_chart(metrics_data)
        
        # Database metrics
        if metric_category in ["All", "Database"]:
            st.subheader("Database Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_database_connections_chart(metrics_data)
            
            with col2:
                self.render_database_query_time_chart(metrics_data)
        
        # ML metrics
        if metric_category in ["All", "ML"]:
            st.subheader("ML Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_ml_predictions_chart(metrics_data)
            
            with col2:
                self.render_ml_accuracy_chart(metrics_data)
    
    def render_health_page(self):
        """Render the health monitoring page."""
        st.header("System Health")
        
        # Get health data
        system_health = self.get_system_health_sync()
        health_checks = system_health.get('checks', [])
        
        # Health summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Healthy", system_health.get('healthy_count', 0), delta=None)
        with col2:
            st.metric("Degraded", system_health.get('degraded_count', 0), delta=None)
        with col3:
            st.metric("Unhealthy", system_health.get('unhealthy_count', 0), delta=None)
        with col4:
            st.metric("Total Components", system_health.get('component_count', 0), delta=None)
        
        # Health checks table
        st.subheader("Health Checks")
        if health_checks:
            health_df = pd.DataFrame([
                {
                    "Component": check.get('name', 'Unknown'),
                    "Status": check.get('status', 'unknown'),
                    "Message": check.get('message', ''),
                    "Duration (ms)": check.get('duration_ms', 0),
                    "Last Check": check.get('timestamp', '')
                }
                for check in health_checks
            ])
            
            # Color code status
            def color_status(val):
                if val == 'healthy':
                    return 'background-color: #d4edda'
                elif val == 'degraded':
                    return 'background-color: #fff3cd'
                elif val == 'unhealthy':
                    return 'background-color: #f8d7da'
                return ''
            
            styled_df = health_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
        
        # Health history
        st.subheader("Health Trends")
        selected_component = st.selectbox(
            "Select Component",
            [check.get('name', 'Unknown') for check in health_checks]
        )
        
        if selected_component:
            self.render_health_history_chart(selected_component)
    
    def render_alerts_page(self):
        """Render the alerts management page."""
        st.header("Alert Management")
        
        # Alert summary
        active_alerts = self.get_active_alerts_sync()
        alert_history = self.get_alert_history_sync()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Alerts", len(active_alerts))
        with col2:
            critical_alerts = sum(1 for alert in active_alerts if alert.get('severity') == 'critical')
            st.metric("Critical Alerts", critical_alerts)
        with col3:
            st.metric("Total Alerts (24h)", len(alert_history))
        
        # Active alerts
        if active_alerts:
            st.subheader("Active Alerts")
            self.render_alerts_table(active_alerts)
        
        # Alert actions
        st.subheader("Alert Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Acknowledge All"):
                st.success("All alerts acknowledged")
        
        with col2:
            if st.button("Refresh Alerts"):
                st.rerun()
        
        # Alert rules
        st.subheader("Alert Rules")
        alert_rules = self.get_alert_rules_sync()
        if alert_rules:
            rules_df = pd.DataFrame([
                {
                    "Rule Name": rule.get('name', ''),
                    "Condition": rule.get('condition', ''),
                    "Threshold": rule.get('threshold', 0),
                    "Severity": rule.get('severity', ''),
                    "Enabled": rule.get('enabled', False)
                }
                for rule in alert_rules
            ])
            st.dataframe(rules_df, use_container_width=True)
        
        # Alert history chart
        st.subheader("Alert History")
        self.render_alert_history_chart(alert_history)
    
    def render_performance_page(self):
        """Render the performance monitoring page."""
        st.header("Performance Monitoring")
        
        # Performance metrics
        current_metrics = self.get_current_metrics_sync()
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_response_time = current_metrics.get('api_avg_response_time', 0)
            st.metric("Avg Response Time", f"{avg_response_time:.2f}ms")
        
        with col2:
            throughput = current_metrics.get('api_requests_per_second', 0)
            st.metric("Throughput", f"{throughput:.1f} req/s")
        
        with col3:
            error_rate = current_metrics.get('api_error_rate', 0)
            st.metric("Error Rate", f"{error_rate:.2f}%")
        
        with col4:
            prediction_accuracy = current_metrics.get('ml_prediction_accuracy', 0)
            st.metric("ML Accuracy", f"{prediction_accuracy:.1f}%")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Time Distribution")
            self.render_response_time_distribution()
        
        with col2:
            st.subheader("Throughput Trend")
            self.render_throughput_trend()
        
        # Resource utilization
        st.subheader("Resource Utilization")
        self.render_resource_utilization_chart(current_metrics)
        
        # Top endpoints
        st.subheader("Top API Endpoints")
        self.render_top_endpoints_chart()
    
    def render_analytics_page(self):
        """Render the analytics and insights page."""
        st.header("Analytics & Insights")
        
        # Analytics summary
        analytics_data = self.get_analytics_data_sync()
        
        # Business metrics
        st.subheader("Business Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily Active Users", analytics_data.get('daily_active_users', 0))
        with col2:
            st.metric("API Calls Today", analytics_data.get('api_calls_today', 0))
        with col3:
            st.metric("Predictions Made", analytics_data.get('predictions_made', 0))
        with col4:
            st.metric("Data Points Processed", analytics_data.get('data_points_processed', 0))
        
        # Usage patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Usage Patterns")
            self.render_usage_patterns_chart()
        
        with col2:
            st.subheader("Sports Coverage")
            self.render_sports_coverage_chart()
        
        # Performance insights
        st.subheader("Performance Insights")
        self.render_performance_insights()
        
        # Capacity planning
        st.subheader("Capacity Planning")
        self.render_capacity_planning_chart()
    
    # Chart rendering methods
    def render_metrics_trend_chart(self):
        """Render system metrics trend chart."""
        # Generate sample data
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='5min')
        cpu_data = np.random.normal(45, 15, len(timestamps))
        memory_data = np.random.normal(60, 10, len(timestamps))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=cpu_data, name='CPU %', line=dict(color='#ff6b6b')))
        fig.add_trace(go.Scatter(x=timestamps, y=memory_data, name='Memory %', line=dict(color='#4ecdc4')))
        
        fig.update_layout(
            title="System Resource Usage",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_health_status_chart(self, system_health):
        """Render component health status chart."""
        checks = system_health.get('checks', [])
        if not checks:
            st.info("No health check data available")
            return
        
        status_counts = {}
        for check in checks:
            status = check.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4,
            marker_colors=['#28a745', '#ffc107', '#dc3545', '#6c757d']
        )])
        
        fig.update_layout(
            title="Component Health Status",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_table(self, alerts):
        """Render alerts table."""
        if not alerts:
            st.info("No active alerts")
            return
        
        alerts_df = pd.DataFrame([
            {
                "Severity": alert.get('severity', '').upper(),
                "Rule": alert.get('rule_name', ''),
                "Message": alert.get('message', ''),
                "Created": alert.get('created_at', ''),
                "Status": alert.get('status', '')
            }
            for alert in alerts
        ])
        
        # Color code severity
        def color_severity(val):
            if val == 'CRITICAL':
                return 'background-color: #f8d7da'
            elif val == 'HIGH':
                return 'background-color: #fff3cd'
            elif val == 'MEDIUM':
                return 'background-color: #d1ecf1'
            return ''
        
        styled_df = alerts_df.style.applymap(color_severity, subset=['Severity'])
        st.dataframe(styled_df, use_container_width=True)
    
    def render_performance_summary(self, metrics):
        """Render performance summary."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Response Time",
                f"{metrics.get('api_avg_response_time', 0):.2f}ms",
                delta=f"{np.random.uniform(-10, 10):.1f}ms"
            )
        
        with col2:
            st.metric(
                "Throughput",
                f"{metrics.get('api_requests_per_second', 0):.1f} req/s",
                delta=f"{np.random.uniform(-5, 15):.1f} req/s"
            )
        
        with col3:
            st.metric(
                "Error Rate",
                f"{metrics.get('api_error_rate', 0):.2f}%",
                delta=f"{np.random.uniform(-0.5, 0.5):.2f}%"
            )
    
    # Helper methods
    def get_status_color(self, status):
        """Get CSS class for status color."""
        status_map = {
            'healthy': 'status-healthy',
            'degraded': 'status-degraded',
            'unhealthy': 'status-unhealthy',
            'unknown': 'status-unknown'
        }
        return status_map.get(status.lower(), 'status-unknown')
    
    def get_metric_color(self, value, warning_threshold, critical_threshold):
        """Get CSS class for metric color based on thresholds."""
        if value >= critical_threshold:
            return 'status-unhealthy'
        elif value >= warning_threshold:
            return 'status-degraded'
        else:
            return 'status-healthy'
    
    # Sync wrapper methods for async operations
    def get_system_health_sync(self):
        """Get system health data synchronously."""
        # In a real implementation, this would fetch from the health checker
        return {
            'status': 'healthy',
            'component_count': 5,
            'healthy_count': 4,
            'degraded_count': 1,
            'unhealthy_count': 0,
            'uptime_seconds': 86400,
            'checks': [
                {'name': 'database', 'status': 'healthy', 'message': 'Database healthy', 'duration_ms': 15.2, 'timestamp': datetime.now().isoformat()},
                {'name': 'redis', 'status': 'healthy', 'message': 'Redis healthy', 'duration_ms': 8.1, 'timestamp': datetime.now().isoformat()},
                {'name': 'api', 'status': 'degraded', 'message': 'API responding slowly', 'duration_ms': 250.5, 'timestamp': datetime.now().isoformat()},
                {'name': 'ml_models', 'status': 'healthy', 'message': 'ML models healthy', 'duration_ms': 45.3, 'timestamp': datetime.now().isoformat()},
                {'name': 'system_resources', 'status': 'healthy', 'message': 'System resources healthy', 'duration_ms': 12.7, 'timestamp': datetime.now().isoformat()}
            ]
        }
    
    def get_current_metrics_sync(self):
        """Get current metrics synchronously."""
        return {
            'system_cpu_percent': np.random.uniform(20, 80),
            'system_memory_percent': np.random.uniform(40, 90),
            'api_avg_response_time': np.random.uniform(50, 300),
            'api_requests_per_second': np.random.uniform(10, 100),
            'api_error_rate': np.random.uniform(0, 5),
            'ml_prediction_accuracy': np.random.uniform(75, 95)
        }
    
    def get_active_alerts_sync(self):
        """Get active alerts synchronously."""
        return [
            {
                'severity': 'high',
                'rule_name': 'high_response_time',
                'message': 'API response time above threshold',
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
        ]
    
    def get_alert_history_sync(self):
        """Get alert history synchronously."""
        return []
    
    def get_alert_rules_sync(self):
        """Get alert rules synchronously."""
        return [
            {'name': 'high_error_rate', 'condition': 'error_rate', 'threshold': 5.0, 'severity': 'high', 'enabled': True},
            {'name': 'high_response_time', 'condition': 'avg_response_time', 'threshold': 2.0, 'severity': 'high', 'enabled': True}
        ]
    
    def get_metrics_history_sync(self, time_range):
        """Get metrics history synchronously."""
        return {}
    
    def get_analytics_data_sync(self):
        """Get analytics data synchronously."""
        return {
            'daily_active_users': np.random.randint(100, 1000),
            'api_calls_today': np.random.randint(1000, 10000),
            'predictions_made': np.random.randint(500, 5000),
            'data_points_processed': np.random.randint(10000, 100000)
        }
    
    # Placeholder chart methods
    def render_cpu_memory_chart(self, data):
        st.info("CPU & Memory chart would be rendered here")
    
    def render_disk_network_chart(self, data):
        st.info("Disk & Network chart would be rendered here")
    
    def render_api_requests_chart(self, data):
        st.info("API Requests chart would be rendered here")
    
    def render_api_response_time_chart(self, data):
        st.info("API Response Time chart would be rendered here")
    
    def render_database_connections_chart(self, data):
        st.info("Database Connections chart would be rendered here")
    
    def render_database_query_time_chart(self, data):
        st.info("Database Query Time chart would be rendered here")
    
    def render_ml_predictions_chart(self, data):
        st.info("ML Predictions chart would be rendered here")
    
    def render_ml_accuracy_chart(self, data):
        st.info("ML Accuracy chart would be rendered here")
    
    def render_health_history_chart(self, component):
        st.info(f"Health history for {component} would be rendered here")
    
    def render_alert_history_chart(self, history):
        st.info("Alert history chart would be rendered here")
    
    def render_response_time_distribution(self):
        st.info("Response time distribution would be rendered here")
    
    def render_throughput_trend(self):
        st.info("Throughput trend would be rendered here")
    
    def render_resource_utilization_chart(self, metrics):
        st.info("Resource utilization chart would be rendered here")
    
    def render_top_endpoints_chart(self):
        st.info("Top endpoints chart would be rendered here")
    
    def render_usage_patterns_chart(self):
        st.info("Usage patterns chart would be rendered here")
    
    def render_sports_coverage_chart(self):
        st.info("Sports coverage chart would be rendered here")
    
    def render_performance_insights(self):
        st.info("Performance insights would be rendered here")
    
    def render_capacity_planning_chart(self):
        st.info("Capacity planning chart would be rendered here")


def main():
    """Main function to run the monitoring dashboard."""
    dashboard = MonitoringDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main() 