"""
Reusable UI Components for Sports Analytics Dashboard

Professional dashboard components including cards, tables, and indicators
for displaying sports data in an intuitive and visually appealing way.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ComponentStyle:
    """Styling configuration for dashboard components."""
    primary_color: str = "#FF6B6B"
    secondary_color: str = "#4ECDC4"
    accent_color: str = "#45B7D1"
    background_color: str = "rgba(255, 255, 255, 0.1)"
    text_color: str = "#FFFFFF"
    border_radius: str = "10px"
    padding: str = "1rem"
    margin: str = "0.5rem 0"

class BaseComponent:
    """Base class for all dashboard components."""
    
    def __init__(self, style: Optional[ComponentStyle] = None):
        self.style = style or ComponentStyle()
    
    def apply_custom_css(self, css_class: str, styles: Dict[str, str]) -> str:
        """Generate custom CSS for component styling."""
        css_rules = []
        for property_name, value in styles.items():
            css_rules.append(f"{property_name}: {value}")
        
        return f"""
        <style>
        .{css_class} {{
            {'; '.join(css_rules)};
        }}
        </style>
        """

class MetricsCard(BaseComponent):
    """Professional metrics display card."""
    
    def render(self, 
               title: str, 
               value: Union[int, float, str], 
               delta: Optional[Union[int, float]] = None,
               delta_color: str = "normal",
               description: Optional[str] = None,
               icon: Optional[str] = None) -> None:
        """
        Render a metrics card with value, delta, and styling.
        
        Args:
            title: Card title
            value: Main metric value
            delta: Change from previous period
            delta_color: Color for delta (normal, inverse, off)
            description: Additional description text
            icon: Optional icon emoji
        """
        try:
            # Custom CSS for the card
            card_css = self.apply_custom_css("metric-card", {
                "background": f"linear-gradient(135deg, {self.style.primary_color} 0%, {self.style.secondary_color} 100%)",
                "padding": self.style.padding,
                "border-radius": self.style.border_radius,
                "color": self.style.text_color,
                "margin": self.style.margin,
                "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                "backdrop-filter": "blur(10px)"
            })
            
            st.markdown(card_css, unsafe_allow_html=True)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if icon:
                        st.markdown(f"### {icon} {title}")
                    else:
                        st.markdown(f"### {title}")
                    
                    # Format value based on type
                    if isinstance(value, (int, float)):
                        if value >= 1000000:
                            formatted_value = f"{value/1000000:.1f}M"
                        elif value >= 1000:
                            formatted_value = f"{value/1000:.1f}K"
                        else:
                            formatted_value = f"{value:.1f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)
                    
                    st.markdown(f"## {formatted_value}")
                    
                    if description:
                        st.caption(description)
                
                with col2:
                    if delta is not None:
                        delta_symbol = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
                        delta_text = f"{delta_symbol} {abs(delta):.1f}"
                        
                        if delta > 0:
                            st.success(delta_text)
                        elif delta < 0:
                            st.error(delta_text)
                        else:
                            st.info(delta_text)
                            
        except Exception as e:
            logger.error(f"Error rendering metrics card: {e}")
            st.error("Failed to render metrics card")

class PlayerCard(BaseComponent):
    """Player information and stats card."""
    
    def render(self, 
               player_data: Dict[str, Any],
               show_photo: bool = True,
               show_stats: bool = True,
               compact: bool = False) -> None:
        """
        Render a player information card.
        
        Args:
            player_data: Dictionary containing player information
            show_photo: Whether to show player photo
            show_stats: Whether to show player statistics
            compact: Use compact layout
        """
        try:
            with st.container():
                if compact:
                    col1, col2 = st.columns([1, 3])
                else:
                    col1, col2, col3 = st.columns([1, 2, 2])
                
                with col1:
                    if show_photo and 'photo_url' in player_data:
                        st.image(player_data['photo_url'], width=80)
                    else:
                        st.markdown("👤")
                
                with col2:
                    st.markdown(f"**{player_data.get('name', 'Unknown Player')}**")
                    st.caption(f"#{player_data.get('number', 'N/A')} • {player_data.get('position', 'N/A')}")
                    st.caption(f"{player_data.get('team', 'Free Agent')}")
                    
                    if 'age' in player_data:
                        st.caption(f"Age: {player_data['age']}")
                
                if not compact:
                    with col3:
                        if show_stats and 'stats' in player_data:
                            stats = player_data['stats']
                            for stat_name, stat_value in stats.items():
                                st.metric(
                                    label=stat_name.replace('_', ' ').title(),
                                    value=f"{stat_value:.1f}" if isinstance(stat_value, float) else str(stat_value)
                                )
                
                st.markdown("---")
                
        except Exception as e:
            logger.error(f"Error rendering player card: {e}")
            st.error("Failed to render player card")

class TeamCard(BaseComponent):
    """Team information and stats card."""
    
    def render(self, 
               team_data: Dict[str, Any],
               show_logo: bool = True,
               show_record: bool = True,
               show_stats: bool = True) -> None:
        """
        Render a team information card.
        
        Args:
            team_data: Dictionary containing team information
            show_logo: Whether to show team logo
            show_record: Whether to show win/loss record
            show_stats: Whether to show team statistics
        """
        try:
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 2])
                
                with col1:
                    if show_logo and 'logo_url' in team_data:
                        st.image(team_data['logo_url'], width=80)
                    else:
                        st.markdown("TEAM")
                
                with col2:
                    st.markdown(f"**{team_data.get('name', 'Unknown Team')}**")
                    st.caption(f"{team_data.get('city', '')} {team_data.get('mascot', '')}")
                    st.caption(f"Conference: {team_data.get('conference', 'N/A')}")
                    
                    if show_record and 'wins' in team_data and 'losses' in team_data:
                        wins = team_data['wins']
                        losses = team_data['losses']
                        win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0
                        st.metric("Record", f"{wins}-{losses}", f"{win_pct:.3f}")
                
                with col3:
                    if show_stats and 'stats' in team_data:
                        stats = team_data['stats']
                        for stat_name, stat_value in list(stats.items())[:3]:  # Show top 3 stats
                            st.metric(
                                label=stat_name.replace('_', ' ').title(),
                                value=f"{stat_value:.1f}" if isinstance(stat_value, float) else str(stat_value)
                            )
                
                st.markdown("---")
                
        except Exception as e:
            logger.error(f"Error rendering team card: {e}")
            st.error("Failed to render team card")

class GameCard(BaseComponent):
    """Game information and score card."""
    
    def render(self, 
               game_data: Dict[str, Any],
               show_details: bool = True,
               is_live: bool = False) -> None:
        """
        Render a game information card.
        
        Args:
            game_data: Dictionary containing game information
            show_details: Whether to show game details
            is_live: Whether the game is currently live
        """
        try:
            with st.container():
                # Live indicator
                if is_live:
                    st.markdown("LIVE")
                
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"**{game_data.get('away_team', 'TBD')}**")
                    if 'away_logo' in game_data:
                        st.image(game_data['away_logo'], width=40)
                
                with col2:
                    if 'away_score' in game_data and 'home_score' in game_data:
                        st.markdown(f"### {game_data['away_score']}-{game_data['home_score']}")
                    else:
                        st.markdown("### VS")
                    
                    if is_live and 'time_remaining' in game_data:
                        st.caption(f"{game_data.get('quarter', 'Q1')} - {game_data['time_remaining']}")
                    elif 'game_time' in game_data:
                        st.caption(game_data['game_time'])
                
                with col3:
                    st.markdown(f"**{game_data.get('home_team', 'TBD')}**")
                    if 'home_logo' in game_data:
                        st.image(game_data['home_logo'], width=40)
                
                if show_details:
                    st.caption(f"Venue: {game_data.get('venue', 'TBD')}")
                    if 'date' in game_data:
                        st.caption(f"Date: {game_data['date']}")
                
                st.markdown("---")
                
        except Exception as e:
            logger.error(f"Error rendering game card: {e}")
            st.error("Failed to render game card")

class StatisticsTable(BaseComponent):
    """Advanced statistics table with sorting and filtering."""
    
    def render(self, 
               data: pd.DataFrame,
               title: str = "Statistics",
               searchable: bool = True,
               sortable: bool = True,
               paginated: bool = True,
               page_size: int = 10) -> None:
        """
        Render an interactive statistics table.
        
        Args:
            data: DataFrame containing the statistics
            title: Table title
            searchable: Enable search functionality
            sortable: Enable column sorting
            paginated: Enable pagination
            page_size: Number of rows per page
        """
        try:
            st.subheader(title)
            
            if data.empty:
                st.info("No data available")
                return
            
            # Search functionality
            if searchable:
                search_term = st.text_input("Search", placeholder="Search in table...")
                if search_term:
                    # Search across all string columns
                    string_columns = data.select_dtypes(include=['object']).columns
                    mask = data[string_columns].astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    data = data[mask]
            
            # Sorting functionality
            if sortable and not data.empty:
                sort_column = st.selectbox(
                    "Sort by",
                    options=data.columns.tolist(),
                    index=0
                )
                sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
                
                ascending = sort_order == "Ascending"
                data = data.sort_values(by=sort_column, ascending=ascending)
            
            # Pagination
            if paginated and len(data) > page_size:
                total_pages = len(data) // page_size + (1 if len(data) % page_size > 0 else 0)
                page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1))
                
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                data = data.iloc[start_idx:end_idx]
            
            # Display table with custom styling
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True
            )
            
            # Table summary
            st.caption(f"Showing {len(data)} rows")
            
        except Exception as e:
            logger.error(f"Error rendering statistics table: {e}")
            st.error("Failed to render statistics table")

class TrendIndicator(BaseComponent):
    """Trend indicator with arrow and percentage change."""
    
    def render(self, 
               current_value: float,
               previous_value: float,
               label: str = "Trend",
               show_percentage: bool = True,
               show_arrow: bool = True) -> None:
        """
        Render a trend indicator showing change between two values.
        
        Args:
            current_value: Current period value
            previous_value: Previous period value
            label: Indicator label
            show_percentage: Show percentage change
            show_arrow: Show directional arrow
        """
        try:
            if previous_value == 0:
                change_pct = 0
            else:
                change_pct = ((current_value - previous_value) / previous_value) * 100
            
            change_abs = current_value - previous_value
            
            # Determine trend direction and color
            if change_abs > 0:
                trend_color = "green"
                arrow = "UP" if show_arrow else ""
                delta_color = "normal"
            elif change_abs < 0:
                trend_color = "red"
                arrow = "DOWN" if show_arrow else ""
                delta_color = "inverse"
            else:
                trend_color = "gray"
                arrow = "FLAT" if show_arrow else ""
                delta_color = "off"
            
            # Format display text
            if show_percentage:
                trend_text = f"{arrow} {abs(change_pct):.1f}%"
            else:
                trend_text = f"{arrow} {abs(change_abs):.1f}"
            
            # Display using Streamlit metric
            st.metric(
                label=label,
                value=f"{current_value:.1f}",
                delta=trend_text,
                delta_color=delta_color
            )
            
        except Exception as e:
            logger.error(f"Error rendering trend indicator: {e}")
            st.error("Failed to render trend indicator")

class ProgressBar(BaseComponent):
    """Custom progress bar with labels and styling."""
    
    def render(self, 
               value: float,
               max_value: float = 100,
               label: str = "Progress",
               show_percentage: bool = True,
               color: str = None) -> None:
        """
        Render a custom progress bar.
        
        Args:
            value: Current value
            max_value: Maximum value
            label: Progress bar label
            show_percentage: Show percentage text
            color: Custom color for the bar
        """
        try:
            percentage = (value / max_value) * 100 if max_value > 0 else 0
            percentage = min(100, max(0, percentage))  # Clamp between 0-100
            
            color = color or self.style.primary_color
            
            # Custom HTML for styled progress bar
            progress_html = f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: {self.style.text_color}; font-weight: bold;">{label}</span>
                    {f'<span style="color: {self.style.text_color};">{percentage:.1f}%</span>' if show_percentage else ''}
                </div>
                <div style="background-color: rgba(255,255,255,0.2); border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, {color}, {self.style.secondary_color}); 
                                height: 100%; width: {percentage}%; border-radius: 10px; 
                                transition: width 0.3s ease;"></div>
                </div>
            </div>
            """
            
            st.markdown(progress_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering progress bar: {e}")
            st.error("Failed to render progress bar")

class AlertBanner(BaseComponent):
    """Alert banner for important notifications."""
    
    def render(self, 
               message: str,
               alert_type: str = "info",
               dismissible: bool = False,
               icon: Optional[str] = None) -> None:
        """
        Render an alert banner.
        
        Args:
            message: Alert message
            alert_type: Type of alert (info, success, warning, error)
            dismissible: Whether the alert can be dismissed
            icon: Optional icon for the alert
        """
        try:
            # Define alert colors and icons
            alert_config = {
                "info": {"color": "#17a2b8", "icon": "INFO"},
                "success": {"color": "#28a745", "icon": "SUCCESS"},
                "warning": {"color": "#ffc107", "icon": "WARNING"},
                "error": {"color": "#dc3545", "icon": "ERROR"}
            }
            
            config = alert_config.get(alert_type, alert_config["info"])
            display_icon = icon or config["icon"]
            
            # Use Streamlit's built-in alert functions
            if alert_type == "success":
                st.success(f"{display_icon} {message}")
            elif alert_type == "warning":
                st.warning(f"{display_icon} {message}")
            elif alert_type == "error":
                st.error(f"{display_icon} {message}")
            else:
                st.info(f"{display_icon} {message}")
                
        except Exception as e:
            logger.error(f"Error rendering alert banner: {e}")
            st.error("Failed to render alert banner") 