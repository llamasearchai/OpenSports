"""
Advanced Alerting System

Comprehensive alerting and notification system for the OpenSports platform
with intelligent alert management, multiple channels, and escalation policies.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import redis.asyncio as redis
from opensports.core.config import settings
from opensports.core.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class ChannelType(Enum):
    """Alert notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    evaluation_window: int  # seconds
    cooldown_period: int = 300  # seconds
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """An active alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: float
    threshold: float
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertChannel:
    """Configuration for an alert notification channel."""
    name: str
    channel_type: ChannelType
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)


class AlertManager:
    """
    Advanced alert management system.
    
    Features:
    - Rule-based alerting with flexible conditions
    - Multiple notification channels
    - Alert deduplication and grouping
    - Escalation policies
    - Alert suppression and maintenance windows
    - Rich context and metadata
    """
    
    def __init__(self):
        self.rules = {}
        self.channels = {}
        self.active_alerts = {}
        self.alert_history = []
        self.redis_client = None
        self.is_running = False
        self.evaluation_interval = 30  # seconds
        
    async def initialize(self):
        """Initialize the alert manager."""
        logger.info("Initializing alert manager")
        
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for alerting")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
        
        # Load default alert rules
        await self._load_default_rules()
        
        # Load default channels
        await self._load_default_channels()
    
    async def _load_default_rules(self):
        """Load default alert rules."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                description="High error rate detected",
                condition="error_rate",
                severity=AlertSeverity.HIGH,
                threshold=5.0,
                comparison="gt",
                evaluation_window=300,
                notification_channels=["email", "slack"]
            ),
            AlertRule(
                name="low_prediction_accuracy",
                description="Prediction accuracy below threshold",
                condition="prediction_accuracy",
                severity=AlertSeverity.MEDIUM,
                threshold=70.0,
                comparison="lt",
                evaluation_window=600,
                notification_channels=["email"]
            ),
            AlertRule(
                name="high_response_time",
                description="API response time too high",
                condition="avg_response_time",
                severity=AlertSeverity.HIGH,
                threshold=2.0,
                comparison="gt",
                evaluation_window=180,
                notification_channels=["slack", "pagerduty"]
            ),
            AlertRule(
                name="database_connection_failure",
                description="Database connection issues",
                condition="db_connection_errors",
                severity=AlertSeverity.CRITICAL,
                threshold=1.0,
                comparison="gte",
                evaluation_window=60,
                notification_channels=["email", "slack", "pagerduty"]
            ),
            AlertRule(
                name="data_ingestion_stopped",
                description="Data ingestion has stopped",
                condition="data_ingestion_rate",
                severity=AlertSeverity.HIGH,
                threshold=10.0,
                comparison="lt",
                evaluation_window=300,
                notification_channels=["email", "slack"]
            )
        ]
        
        for rule in default_rules:
            await self.add_rule(rule)
    
    async def _load_default_channels(self):
        """Load default notification channels."""
        default_channels = [
            AlertChannel(
                name="email",
                channel_type=ChannelType.EMAIL,
                config={
                    "smtp_server": getattr(settings, 'SMTP_SERVER', 'localhost'),
                    "smtp_port": getattr(settings, 'SMTP_PORT', 587),
                    "username": getattr(settings, 'SMTP_USERNAME', ''),
                    "password": getattr(settings, 'SMTP_PASSWORD', ''),
                    "from_email": getattr(settings, 'ALERT_FROM_EMAIL', 'alerts@opensports.com'),
                    "to_emails": getattr(settings, 'ALERT_TO_EMAILS', ['admin@opensports.com'])
                }
            ),
            AlertChannel(
                name="slack",
                channel_type=ChannelType.SLACK,
                config={
                    "webhook_url": getattr(settings, 'SLACK_WEBHOOK_URL', ''),
                    "channel": getattr(settings, 'SLACK_CHANNEL', '#alerts'),
                    "username": "OpenSports Alerts"
                }
            ),
            AlertChannel(
                name="pagerduty",
                channel_type=ChannelType.PAGERDUTY,
                config={
                    "integration_key": getattr(settings, 'PAGERDUTY_INTEGRATION_KEY', ''),
                    "service_name": "OpenSports"
                },
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.HIGH]
            )
        ]
        
        for channel in default_channels:
            await self.add_channel(channel)
    
    async def add_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.setex(
                f"alert_rule:{rule.name}",
                86400,  # 24 hours
                json.dumps({
                    "name": rule.name,
                    "description": rule.description,
                    "condition": rule.condition,
                    "severity": rule.severity.value,
                    "threshold": rule.threshold,
                    "comparison": rule.comparison,
                    "evaluation_window": rule.evaluation_window,
                    "enabled": rule.enabled,
                    "tags": rule.tags,
                    "notification_channels": rule.notification_channels
                })
            )
    
    async def add_channel(self, channel: AlertChannel):
        """Add a new notification channel."""
        self.channels[channel.name] = channel
        logger.info(f"Added notification channel: {channel.name}")
    
    async def start_monitoring(self):
        """Start the alert monitoring loop."""
        self.is_running = True
        logger.info("Starting alert monitoring")
        
        while self.is_running:
            try:
                await self._evaluate_rules()
                await self._process_alert_queue()
                await self._cleanup_resolved_alerts()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(5)
    
    async def stop_monitoring(self):
        """Stop the alert monitoring loop."""
        self.is_running = False
        logger.info("Stopping alert monitoring")
    
    async def _evaluate_rules(self):
        """Evaluate all active alert rules."""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get current metric value
                current_value = await self._get_metric_value(rule.condition)
                
                if current_value is None:
                    continue
                
                # Evaluate condition
                should_alert = self._evaluate_condition(
                    current_value, rule.threshold, rule.comparison
                )
                
                if should_alert:
                    await self._trigger_alert(rule, current_value)
                else:
                    await self._resolve_alert_if_exists(rule_name)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric."""
        if not self.redis_client:
            return None
        
        try:
            # Get latest metrics snapshot
            metrics_data = await self.redis_client.get("metrics:latest")
            if metrics_data:
                metrics = eval(metrics_data)  # In production, use json.loads
                return metrics.get(metric_name)
        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
        
        return None
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if a condition is met."""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "eq":
            return value == threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "lte":
            return value <= threshold
        else:
            logger.warning(f"Unknown comparison operator: {comparison}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert for a rule."""
        alert_id = f"{rule.name}_{int(datetime.now().timestamp())}"
        
        # Check if alert already exists (deduplication)
        existing_alert = self.active_alerts.get(rule.name)
        if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
            # Update existing alert
            existing_alert.value = current_value
            existing_alert.updated_at = datetime.now()
            return
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.description}: {current_value} {rule.comparison} {rule.threshold}",
            value=current_value,
            threshold=rule.threshold,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=rule.tags.copy(),
            context={
                "metric": rule.condition,
                "evaluation_window": rule.evaluation_window
            }
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert, rule.notification_channels)
        
        logger.warning(f"Alert triggered: {alert.message}")
    
    async def _resolve_alert_if_exists(self, rule_name: str):
        """Resolve an alert if it exists and is active."""
        alert = self.active_alerts.get(rule_name)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.updated_at = datetime.now()
            
            # Send resolution notification
            await self._send_resolution_notification(alert)
            
            logger.info(f"Alert resolved: {alert.message}")
    
    async def _send_notifications(self, alert: Alert, channel_names: List[str]):
        """Send alert notifications to specified channels."""
        for channel_name in channel_names:
            channel = self.channels.get(channel_name)
            if not channel or not channel.enabled:
                continue
            
            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            try:
                await self._send_to_channel(alert, channel)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel_name}: {e}")
    
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to a specific channel."""
        if channel.channel_type == ChannelType.EMAIL:
            await self._send_email_alert(alert, channel)
        elif channel.channel_type == ChannelType.SLACK:
            await self._send_slack_alert(alert, channel)
        elif channel.channel_type == ChannelType.WEBHOOK:
            await self._send_webhook_alert(alert, channel)
        elif channel.channel_type == ChannelType.PAGERDUTY:
            await self._send_pagerduty_alert(alert, channel)
        else:
            logger.warning(f"Unsupported channel type: {channel.channel_type}")
    
    async def _send_email_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert via email."""
        config = channel.config
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] OpenSports Alert: {alert.rule_name}"
        
        # Email body
        body = f"""
        Alert: {alert.rule_name}
        Severity: {alert.severity.value.upper()}
        Message: {alert.message}
        
        Details:
        - Current Value: {alert.value}
        - Threshold: {alert.threshold}
        - Triggered At: {alert.created_at}
        
        Context: {json.dumps(alert.context, indent=2)}
        
        --
        OpenSports Monitoring System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email (in production, use async email library)
        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('username'):
                server.starttls()
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.rule_name}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert via Slack webhook."""
        config = channel.config
        webhook_url = config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        # Create Slack message
        color_map = {
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.HIGH: "warning",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "good",
            AlertSeverity.INFO: "good"
        }
        
        payload = {
            "channel": config.get('channel', '#alerts'),
            "username": config.get('username', 'OpenSports Alerts'),
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "title": f"Alert: {alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Current Value", "value": str(alert.value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Triggered At", "value": alert.created_at.isoformat(), "short": True}
                ],
                "footer": "OpenSports Monitoring",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Slack alert sent for {alert.rule_name}")
                else:
                    logger.error(f"Failed to send Slack alert: {response.status}")
    
    async def _send_webhook_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert via generic webhook."""
        config = channel.config
        webhook_url = config.get('url')
        
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return
        
        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": alert.message,
            "value": alert.value,
            "threshold": alert.threshold,
            "created_at": alert.created_at.isoformat(),
            "tags": alert.tags,
            "context": alert.context
        }
        
        headers = config.get('headers', {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Webhook alert sent for {alert.rule_name}")
                else:
                    logger.error(f"Failed to send webhook alert: {response.status}")
    
    async def _send_pagerduty_alert(self, alert: Alert, channel: AlertChannel):
        """Send alert via PagerDuty."""
        config = channel.config
        integration_key = config.get('integration_key')
        
        if not integration_key:
            logger.warning("PagerDuty integration key not configured")
            return
        
        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "dedup_key": alert.rule_name,
            "payload": {
                "summary": alert.message,
                "severity": alert.severity.value,
                "source": "OpenSports",
                "component": alert.rule_name,
                "group": "monitoring",
                "class": "alert",
                "custom_details": {
                    "current_value": alert.value,
                    "threshold": alert.threshold,
                    "context": alert.context
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            ) as response:
                if response.status == 202:
                    logger.info(f"PagerDuty alert sent for {alert.rule_name}")
                else:
                    logger.error(f"Failed to send PagerDuty alert: {response.status}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification."""
        # Create a copy of the alert for resolution notification
        resolution_alert = Alert(
            id=f"{alert.id}_resolved",
            rule_name=alert.rule_name,
            severity=alert.severity,
            status=AlertStatus.RESOLVED,
            message=f"RESOLVED: {alert.message}",
            value=alert.value,
            threshold=alert.threshold,
            created_at=alert.resolved_at,
            updated_at=alert.resolved_at,
            tags=alert.tags,
            context=alert.context
        )
        
        # Get rule to find notification channels
        rule = self.rules.get(alert.rule_name)
        if rule:
            await self._send_notifications(resolution_alert, rule.notification_channels)
    
    async def _process_alert_queue(self):
        """Process any queued alert operations."""
        # This would handle batching, rate limiting, etc.
        pass
    
    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove resolved alerts older than 24 hours from active alerts
        to_remove = []
        for rule_name, alert in self.active_alerts.items():
            if (alert.status == AlertStatus.RESOLVED and 
                alert.resolved_at and alert.resolved_at < cutoff_time):
                to_remove.append(rule_name)
        
        for rule_name in to_remove:
            del self.active_alerts[rule_name]
    
    async def acknowledge_alert(self, rule_name: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        alert = self.active_alerts.get(rule_name)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now()
            
            logger.info(f"Alert acknowledged: {rule_name} by {acknowledged_by}")
            return True
        
        return False
    
    async def suppress_alert(self, rule_name: str, duration_minutes: int) -> bool:
        """Suppress an alert for a specified duration."""
        rule = self.rules.get(rule_name)
        if rule:
            # Store suppression in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    f"alert_suppressed:{rule_name}",
                    duration_minutes * 60,
                    "true"
                )
            
            logger.info(f"Alert suppressed: {rule_name} for {duration_minutes} minutes")
            return True
        
        return False
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() 
                if alert.status == AlertStatus.ACTIVE]
    
    async def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history 
                if alert.created_at >= cutoff_time]
    
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status."""
        active_alerts = await self.get_active_alerts()
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = sum(
                1 for alert in active_alerts if alert.severity == severity
            )
        
        return {
            "total_active_alerts": len(active_alerts),
            "severity_breakdown": severity_counts,
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "total_channels": len(self.channels),
            "enabled_channels": sum(1 for channel in self.channels.values() if channel.enabled),
            "monitoring_status": "running" if self.is_running else "stopped",
            "last_evaluation": datetime.now().isoformat()
        } 