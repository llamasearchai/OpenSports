"""
OpenSports Segmentation Module

Advanced audience segmentation and clustering for sports analytics including:
- Fan behavior segmentation
- Player performance clustering
- Team strategy grouping
- Market segmentation
"""

from opensports.segmentation.audience import AudienceSegmenter
from opensports.segmentation.player_clustering import PlayerClusterer
from opensports.segmentation.team_analysis import TeamAnalyzer

__all__ = [
    "AudienceSegmenter",
    "PlayerClusterer",
    "TeamAnalyzer",
] 