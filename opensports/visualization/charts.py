"""
Advanced Chart Components for Sports Analytics

Professional-grade interactive charts using Plotly for sports data visualization.
Includes performance trends, comparisons, heatmaps, and network analysis.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BaseChart:
    """Base class for all chart components."""
    
    def __init__(self):
        self.default_layout = {
            'template': 'plotly_dark',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'
        }
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def apply_layout(self, fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
        """Apply consistent layout styling to figure."""
        layout_updates = {**self.default_layout, **kwargs}
        if title:
            layout_updates['title'] = {
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'white'}
            }
        
        fig.update_layout(**layout_updates)
        return fig

class PerformanceChart(BaseChart):
    """Performance analysis charts."""
    
    def create_trend_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create performance trend chart over time."""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Performance Score', 'Efficiency & Impact'),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
            )
            
            # Performance score trend
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['performance_score'],
                    mode='lines+markers',
                    name='Performance Score',
                    line=dict(color=self.color_palette[0], width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>%{y:.1f}</b><br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Efficiency trend
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['efficiency'],
                    mode='lines+markers',
                    name='Efficiency',
                    line=dict(color=self.color_palette[1], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Impact trend (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['impact'],
                    mode='lines+markers',
                    name='Impact',
                    line=dict(color=self.color_palette[2], width=2),
                    marker=dict(size=4),
                    yaxis='y4',
                    hovertemplate='<b>%{y:.1f}</b><br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Performance Score", row=1, col=1)
            fig.update_yaxes(title_text="Efficiency", row=2, col=1)
            fig.update_yaxes(title_text="Impact", secondary_y=True, row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            return self.apply_layout(fig, "Performance Trends Over Time")
            
        except Exception as e:
            logger.error(f"Error creating trend chart: {e}")
            return go.Figure()
    
    def create_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create performance distribution chart."""
        try:
            fig = go.Figure()
            
            # Performance score distribution
            fig.add_trace(go.Histogram(
                x=data['performance_score'],
                nbinsx=20,
                name='Performance Distribution',
                marker_color=self.color_palette[0],
                opacity=0.7,
                hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_score = data['performance_score'].mean()
            fig.add_vline(
                x=mean_score,
                line_dash="dash",
                line_color="white",
                annotation_text=f"Mean: {mean_score:.1f}"
            )
            
            fig.update_xaxes(title_text="Performance Score")
            fig.update_yaxes(title_text="Frequency")
            
            return self.apply_layout(fig, "Performance Score Distribution")
            
        except Exception as e:
            logger.error(f"Error creating distribution chart: {e}")
            return go.Figure()
    
    def create_comparison_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create player comparison radar chart."""
        try:
            # Pivot data for radar chart
            pivot_data = data.pivot(index='player', columns='metric', values='value')
            
            fig = go.Figure()
            
            for i, player in enumerate(pivot_data.index):
                fig.add_trace(go.Scatterpolar(
                    r=pivot_data.loc[player].values,
                    theta=pivot_data.columns,
                    fill='toself',
                    name=player,
                    line_color=self.color_palette[i % len(self.color_palette)],
                    hovertemplate='<b>%{theta}</b><br>%{r:.1f}<extra></extra>'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(pivot_data.max())]
                    )
                )
            )
            
            return self.apply_layout(fig, "Player Performance Comparison")
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return go.Figure()

class TeamComparisonChart(BaseChart):
    """Team comparison and analysis charts."""
    
    def create_comparison_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create team comparison bar chart."""
        try:
            fig = go.Figure()
            
            metrics = ['wins', 'losses', 'points_avg', 'defense_rating']
            
            for i, metric in enumerate(metrics):
                if metric in data.columns:
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=data['team_name'],
                        y=data[metric],
                        marker_color=self.color_palette[i],
                        hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
                    ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title="Teams",
                yaxis_title="Value"
            )
            
            return self.apply_layout(fig, "Team Performance Comparison")
            
        except Exception as e:
            logger.error(f"Error creating team comparison chart: {e}")
            return go.Figure()
    
    def create_standings_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create team standings visualization."""
        try:
            # Sort by wins descending
            data_sorted = data.sort_values('wins', ascending=False)
            
            fig = go.Figure()
            
            # Wins bars
            fig.add_trace(go.Bar(
                name='Wins',
                x=data_sorted['team_name'],
                y=data_sorted['wins'],
                marker_color=self.color_palette[0],
                hovertemplate='<b>%{x}</b><br>Wins: %{y}<extra></extra>'
            ))
            
            # Losses bars (negative for visual effect)
            fig.add_trace(go.Bar(
                name='Losses',
                x=data_sorted['team_name'],
                y=-data_sorted['losses'],
                marker_color=self.color_palette[1],
                hovertemplate='<b>%{x}</b><br>Losses: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                barmode='relative',
                xaxis_title="Teams",
                yaxis_title="Games",
                yaxis=dict(tickformat='.0f')
            )
            
            return self.apply_layout(fig, "Team Standings")
            
        except Exception as e:
            logger.error(f"Error creating standings chart: {e}")
            return go.Figure()

class PlayerTrajectoryChart(BaseChart):
    """Player trajectory and forecast charts."""
    
    def create_trajectory_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create player performance trajectory."""
        try:
            fig = go.Figure()
            
            # Historical performance
            historical_data = data[data['type'] == 'historical']
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['performance'],
                mode='lines+markers',
                name='Historical Performance',
                line=dict(color=self.color_palette[0], width=3),
                marker=dict(size=6),
                hovertemplate='<b>%{y:.1f}</b><br>%{x}<extra></extra>'
            ))
            
            # Trend line
            if len(historical_data) > 1:
                z = np.polyfit(range(len(historical_data)), historical_data['performance'], 1)
                trend_line = np.poly1d(z)(range(len(historical_data)))
                
                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='white', width=2, dash='dash'),
                    hovertemplate='Trend: %{y:.1f}<extra></extra>'
                ))
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Performance Score")
            
            return self.apply_layout(fig, "Player Performance Trajectory")
            
        except Exception as e:
            logger.error(f"Error creating trajectory chart: {e}")
            return go.Figure()
    
    def create_forecast_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create player performance forecast."""
        try:
            fig = go.Figure()
            
            # Historical data
            historical = data[data['type'] == 'historical']
            forecast = data[data['type'] == 'forecast']
            
            # Historical line
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['performance'],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.color_palette[0], width=3),
                marker=dict(size=6)
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['performance'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_palette[1], width=3, dash='dot'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            # Confidence interval
            if 'confidence_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                    y=forecast['confidence_upper'].tolist() + forecast['confidence_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(68, 205, 196, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Performance Score")
            
            return self.apply_layout(fig, "Performance Forecast")
            
        except Exception as e:
            logger.error(f"Error creating forecast chart: {e}")
            return go.Figure()

class GameFlowChart(BaseChart):
    """Game flow and momentum charts."""
    
    def create_flow_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create game flow momentum chart."""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Score Progression', 'Momentum'),
                vertical_spacing=0.1
            )
            
            # Score progression
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['home_score'],
                    mode='lines',
                    name='Home Team',
                    line=dict(color=self.color_palette[0], width=3),
                    hovertemplate='<b>Home: %{y}</b><br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['away_score'],
                    mode='lines',
                    name='Away Team',
                    line=dict(color=self.color_palette[1], width=3),
                    hovertemplate='<b>Away: %{y}</b><br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Momentum (score differential)
            score_diff = data['home_score'] - data['away_score']
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=score_diff,
                    mode='lines',
                    name='Momentum',
                    line=dict(color=self.color_palette[2], width=2),
                    fill='tonexty' if score_diff.iloc[0] >= 0 else 'tozeroy',
                    hovertemplate='<b>Differential: %{y}</b><br>Time: %{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line for momentum
            fig.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=1)
            
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Score Differential", row=2, col=1)
            fig.update_xaxes(title_text="Game Time", row=2, col=1)
            
            return self.apply_layout(fig, "Game Flow Analysis")
            
        except Exception as e:
            logger.error(f"Error creating flow chart: {e}")
            return go.Figure()
    
    def create_quarter_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create quarter-by-quarter analysis."""
        try:
            fig = go.Figure()
            
            quarters = data['quarter'].unique()
            
            for i, quarter in enumerate(quarters):
                quarter_data = data[data['quarter'] == quarter]
                
                fig.add_trace(go.Bar(
                    name=f'Q{quarter}',
                    x=['Home', 'Away'],
                    y=[quarter_data['home_points'].sum(), quarter_data['away_points'].sum()],
                    marker_color=self.color_palette[i],
                    hovertemplate='<b>%{x}</b><br>Points: %{y}<extra></extra>'
                ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title="Team",
                yaxis_title="Points"
            )
            
            return self.apply_layout(fig, "Quarter Analysis")
            
        except Exception as e:
            logger.error(f"Error creating quarter analysis: {e}")
            return go.Figure()

class HeatmapChart(BaseChart):
    """Heatmap visualizations for performance analysis."""
    
    def create_performance_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create performance heatmap."""
        try:
            # Pivot data for heatmap
            if 'player' in data.columns and 'metric' in data.columns and 'value' in data.columns:
                heatmap_data = data.pivot(index='player', columns='metric', values='value')
            else:
                # Generate sample heatmap data
                players = [f'Player {i}' for i in range(1, 11)]
                metrics = ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks']
                heatmap_data = pd.DataFrame(
                    np.random.rand(len(players), len(metrics)) * 100,
                    index=players,
                    columns=metrics
                )
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>'
            ))
            
            fig.update_xaxes(title_text="Metrics")
            fig.update_yaxes(title_text="Players")
            
            return self.apply_layout(fig, "Performance Heatmap")
            
        except Exception as e:
            logger.error(f"Error creating performance heatmap: {e}")
            return go.Figure()
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        try:
            # Calculate correlation matrix
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_columns].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False,
                hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_xaxes(title_text="Metrics")
            fig.update_yaxes(title_text="Metrics")
            
            return self.apply_layout(fig, "Correlation Analysis")
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()

class NetworkChart(BaseChart):
    """Network analysis charts for team interactions."""
    
    def create_team_network(self, data: Dict) -> go.Figure:
        """Create team interaction network."""
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (teams)
            for team in data.get('teams', []):
                G.add_node(team['name'], **team)
            
            # Add edges (interactions)
            for interaction in data.get('interactions', []):
                G.add_edge(
                    interaction['team1'],
                    interaction['team2'],
                    weight=interaction.get('strength', 1)
                )
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Extract node and edge information
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(255,255,255,0.5)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=30,
                    color=self.color_palette[:len(node_text)],
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=False
            ))
            
            fig.update_layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Team Interaction Network",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='white', size=14)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return self.apply_layout(fig, "Team Network Analysis")
            
        except Exception as e:
            logger.error(f"Error creating network chart: {e}")
            return go.Figure()
    
    def create_player_network(self, data: Dict) -> go.Figure:
        """Create player interaction network."""
        try:
            # Similar to team network but for players
            G = nx.Graph()
            
            # Add nodes (players)
            for player in data.get('players', []):
                G.add_node(player['name'], **player)
            
            # Add edges (interactions/connections)
            for connection in data.get('connections', []):
                G.add_edge(
                    connection['player1'],
                    connection['player2'],
                    weight=connection.get('strength', 1)
                )
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1.5, iterations=50)
            
            # Create visualization similar to team network
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=20,
                    color=self.color_palette[:len(node_text)],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=False
            ))
            
            fig.update_layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return self.apply_layout(fig, "Player Network Analysis")
            
        except Exception as e:
            logger.error(f"Error creating player network: {e}")
            return go.Figure() 