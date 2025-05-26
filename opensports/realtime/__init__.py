"""
OpenSports Real-time Analytics Module

Real-time data processing and analytics for live sports events including:
- Live game streaming and processing
- Real-time alerts and notifications
- Performance monitoring
- Event detection and analysis
"""

from opensports.realtime.stream_processor import StreamProcessor
from opensports.realtime.live_analytics import LiveAnalytics
from opensports.realtime.alerts import AlertManager
from opensports.realtime.event_detector import EventDetector

__all__ = [
    "StreamProcessor",
    "LiveAnalytics",
    "AlertManager", 
    "EventDetector",
] 