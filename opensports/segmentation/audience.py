"""
Audience Segmentation for Sports Analytics

Advanced fan and audience segmentation using machine learning clustering techniques.
Includes behavioral analysis, engagement patterns, and predictive segmentation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from opensports.core.config import settings
from opensports.core.logging import get_logger
from opensports.core.database import get_database
from opensports.core.cache import cache_async_result

logger = get_logger(__name__)


class AudienceSegmenter:
    """
    Advanced audience segmentation system for sports analytics.
    
    Features:
    - Multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)
    - UMAP dimensionality reduction for visualization
    - Behavioral pattern analysis
    - Engagement scoring and prediction
    - Dynamic segment updates
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.segment_profiles = {}
        self.db = get_database()
        
    async def prepare_features(self, audience_data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for audience segmentation.
        
        Args:
            audience_data: Raw audience/fan data DataFrame
            
        Returns:
            Engineered features DataFrame
        """
        logger.info(f"Preparing features for {len(audience_data)} audience records")
        
        features = audience_data.copy()
        
        # Engagement features
        await self._add_engagement_features(features)
        
        # Behavioral features
        await self._add_behavioral_features(features)
        
        # Temporal features
        await self._add_temporal_features(features)
        
        # Preference features
        await self._add_preference_features(features)
        
        # Social features
        await self._add_social_features(features)
        
        # Economic features
        await self._add_economic_features(features)
        
        logger.info(f"Feature engineering complete. Shape: {features.shape}")
        return features
    
    async def _add_engagement_features(self, features: pd.DataFrame) -> None:
        """Add engagement-related features."""
        # Content consumption
        features['total_content_views'] = features.get('content_views', 0)
        features['avg_session_duration'] = features.get('session_duration', 0)
        features['content_completion_rate'] = features.get('completion_rate', 0.5)
        
        # Interaction patterns
        features['social_shares'] = features.get('shares', 0)
        features['comments_posted'] = features.get('comments', 0)
        features['likes_given'] = features.get('likes', 0)
        
        # Frequency metrics
        features['days_active_last_30'] = features.get('active_days', 0)
        features['avg_daily_sessions'] = features.get('daily_sessions', 0)
        
        # Engagement score (composite)
        features['engagement_score'] = (
            features['total_content_views'] * 0.3 +
            features['avg_session_duration'] * 0.2 +
            features['social_shares'] * 0.2 +
            features['comments_posted'] * 0.15 +
            features['days_active_last_30'] * 0.15
        )
    
    async def _add_behavioral_features(self, features: pd.DataFrame) -> None:
        """Add behavioral pattern features."""
        # Content preferences
        features['prefers_live_content'] = features.get('live_content_pct', 0.5)
        features['prefers_highlights'] = features.get('highlights_pct', 0.3)
        features['prefers_analysis'] = features.get('analysis_pct', 0.2)
        
        # Device usage patterns
        features['mobile_usage_pct'] = features.get('mobile_pct', 0.6)
        features['desktop_usage_pct'] = features.get('desktop_pct', 0.3)
        features['tv_usage_pct'] = features.get('tv_pct', 0.1)
        
        # Time-based behavior
        features['peak_hour'] = features.get('peak_hour', 20)  # 8 PM default
        features['weekend_activity_pct'] = features.get('weekend_pct', 0.4)
        
        # Loyalty indicators
        features['account_age_days'] = features.get('account_age', 365)
        features['subscription_renewals'] = features.get('renewals', 0)
        features['referrals_made'] = features.get('referrals', 0)
    
    async def _add_temporal_features(self, features: pd.DataFrame) -> None:
        """Add temporal behavior features."""
        # Seasonality patterns
        features['activity_variance'] = features.get('activity_variance', 0.2)
        features['peak_season_multiplier'] = features.get('peak_multiplier', 1.5)
        
        # Recency metrics
        features['days_since_last_visit'] = features.get('days_since_visit', 7)
        features['days_since_last_purchase'] = features.get('days_since_purchase', 30)
        
        # Frequency patterns
        features['visit_frequency_score'] = features.get('frequency_score', 0.5)
        features['content_binge_tendency'] = features.get('binge_tendency', 0.3)
    
    async def _add_preference_features(self, features: pd.DataFrame) -> None:
        """Add sports and content preference features."""
        # Sport preferences
        features['favorite_sport'] = features.get('favorite_sport', 'nba')
        features['num_sports_followed'] = features.get('sports_count', 1)
        features['sport_diversity_score'] = features.get('diversity_score', 0.3)
        
        # Team preferences
        features['has_favorite_team'] = features.get('has_favorite', 1)
        features['team_loyalty_score'] = features.get('loyalty_score', 0.7)
        features['follows_multiple_teams'] = features.get('multiple_teams', 0)
        
        # Content type preferences
        features['news_consumption'] = features.get('news_pct', 0.4)
        features['stats_consumption'] = features.get('stats_pct', 0.3)
        features['video_consumption'] = features.get('video_pct', 0.6)
    
    async def _add_social_features(self, features: pd.DataFrame) -> None:
        """Add social behavior features."""
        # Social network size
        features['followers_count'] = features.get('followers', 0)
        features['following_count'] = features.get('following', 0)
        features['social_influence_score'] = features.get('influence_score', 0.1)
        
        # Community participation
        features['forum_posts'] = features.get('forum_posts', 0)
        features['community_reputation'] = features.get('reputation', 0)
        features['group_memberships'] = features.get('groups', 0)
        
        # Viral behavior
        features['content_shared_ratio'] = features.get('share_ratio', 0.1)
        features['viral_content_created'] = features.get('viral_content', 0)
    
    async def _add_economic_features(self, features: pd.DataFrame) -> None:
        """Add economic and monetization features."""
        # Spending behavior
        features['total_spent'] = features.get('total_spent', 0)
        features['avg_transaction_value'] = features.get('avg_transaction', 0)
        features['purchase_frequency'] = features.get('purchase_freq', 0)
        
        # Subscription behavior
        features['is_subscriber'] = features.get('is_subscriber', 0)
        features['subscription_tier'] = features.get('sub_tier', 0)
        features['subscription_length'] = features.get('sub_length', 0)
        
        # Price sensitivity
        features['responds_to_discounts'] = features.get('discount_response', 0.5)
        features['premium_content_usage'] = features.get('premium_usage', 0.2)
    
    async def segment_audience(
        self,
        audience_data: pd.DataFrame,
        n_clusters: Optional[int] = None,
        algorithm: str = "kmeans",
        use_umap: bool = True
    ) -> Dict[str, Any]:
        """
        Perform audience segmentation using specified clustering algorithm.
        
        Args:
            audience_data: Audience data DataFrame
            n_clusters: Number of clusters (auto-determined if None)
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            use_umap: Whether to use UMAP for dimensionality reduction
            
        Returns:
            Segmentation results with cluster assignments and profiles
        """
        logger.info(f"Starting audience segmentation with {algorithm}")
        
        # Prepare features
        features = await self.prepare_features(audience_data)
        
        # Select feature columns
        exclude_cols = ['user_id', 'created_at', 'updated_at']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        X = features[self.feature_columns]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale features
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X_scaled = self.scalers['scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['scaler'].transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Apply UMAP for dimensionality reduction if requested
        if use_umap:
            umap_reducer = umap.UMAP(
                n_components=min(10, len(self.feature_columns)),
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
            X_reduced = umap_reducer.fit_transform(X_scaled)
            X_for_clustering = X_reduced
        else:
            X_for_clustering = X_scaled
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None and algorithm in ['kmeans', 'hierarchical']:
            n_clusters = await self._find_optimal_clusters(X_for_clustering, algorithm)
        
        # Perform clustering
        if algorithm == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X_for_clustering)
        elif algorithm == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(X_for_clustering)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        elif algorithm == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X_for_clustering)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # Calculate clustering metrics
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_for_clustering, cluster_labels)
        else:
            silhouette_avg = 0
            calinski_harabasz = 0
        
        # Create segment profiles
        features['cluster'] = cluster_labels
        segment_profiles = await self._create_segment_profiles(features)
        
        # Store results
        self.models[algorithm] = {
            'clusterer': clusterer,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'use_umap': use_umap
        }
        
        if use_umap:
            self.models[algorithm]['umap_reducer'] = umap_reducer
        
        self.segment_profiles = segment_profiles
        
        results = {
            'algorithm': algorithm,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'segment_profiles': segment_profiles,
            'feature_importance': await self._calculate_feature_importance(features),
        }
        
        logger.info(f"Segmentation complete. Found {n_clusters} segments with silhouette score: {silhouette_avg:.3f}")
        return results
    
    async def _find_optimal_clusters(self, X: np.ndarray, algorithm: str) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        max_clusters = min(10, len(X) // 10)
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_clusters + 1):
            if algorithm == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif algorithm == "hierarchical":
                clusterer = AgglomerativeClustering(n_clusters=k)
            
            cluster_labels = clusterer.fit_predict(X)
            
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            if algorithm == "kmeans":
                inertias.append(clusterer.inertia_)
        
        # Find optimal k using silhouette score
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    async def _create_segment_profiles(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Create detailed profiles for each segment."""
        profiles = {}
        
        for cluster_id in features['cluster'].unique():
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = features[features['cluster'] == cluster_id]
            
            # Calculate segment characteristics
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features) * 100,
                'characteristics': {},
                'top_features': {},
                'behavioral_patterns': {}
            }
            
            # Numerical feature statistics
            numerical_cols = features.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'cluster']
            
            for col in numerical_cols:
                profile['characteristics'][col] = {
                    'mean': float(cluster_data[col].mean()),
                    'median': float(cluster_data[col].median()),
                    'std': float(cluster_data[col].std())
                }
            
            # Identify top distinguishing features
            overall_means = features[numerical_cols].mean()
            cluster_means = cluster_data[numerical_cols].mean()
            
            # Calculate relative differences
            relative_diffs = ((cluster_means - overall_means) / overall_means).abs()
            top_features = relative_diffs.nlargest(5)
            
            profile['top_features'] = {
                feature: {
                    'cluster_value': float(cluster_means[feature]),
                    'overall_value': float(overall_means[feature]),
                    'relative_difference': float(relative_diffs[feature])
                }
                for feature in top_features.index
            }
            
            # Behavioral pattern analysis
            profile['behavioral_patterns'] = {
                'engagement_level': self._categorize_engagement(cluster_data),
                'loyalty_level': self._categorize_loyalty(cluster_data),
                'spending_level': self._categorize_spending(cluster_data),
                'activity_pattern': self._categorize_activity(cluster_data)
            }
            
            profiles[f'segment_{cluster_id}'] = profile
        
        return profiles
    
    def _categorize_engagement(self, data: pd.DataFrame) -> str:
        """Categorize engagement level for a segment."""
        avg_engagement = data.get('engagement_score', pd.Series([0])).mean()
        
        if avg_engagement > 0.7:
            return "High"
        elif avg_engagement > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _categorize_loyalty(self, data: pd.DataFrame) -> str:
        """Categorize loyalty level for a segment."""
        avg_loyalty = data.get('team_loyalty_score', pd.Series([0.5])).mean()
        avg_account_age = data.get('account_age_days', pd.Series([365])).mean()
        
        if avg_loyalty > 0.8 and avg_account_age > 730:
            return "Very High"
        elif avg_loyalty > 0.6 and avg_account_age > 365:
            return "High"
        elif avg_loyalty > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _categorize_spending(self, data: pd.DataFrame) -> str:
        """Categorize spending level for a segment."""
        avg_spending = data.get('total_spent', pd.Series([0])).mean()
        
        if avg_spending > 500:
            return "High"
        elif avg_spending > 100:
            return "Medium"
        elif avg_spending > 0:
            return "Low"
        else:
            return "None"
    
    def _categorize_activity(self, data: pd.DataFrame) -> str:
        """Categorize activity pattern for a segment."""
        avg_frequency = data.get('visit_frequency_score', pd.Series([0.5])).mean()
        avg_session_duration = data.get('avg_session_duration', pd.Series([0])).mean()
        
        if avg_frequency > 0.8 and avg_session_duration > 30:
            return "Heavy User"
        elif avg_frequency > 0.5 and avg_session_duration > 15:
            return "Regular User"
        elif avg_frequency > 0.2:
            return "Casual User"
        else:
            return "Inactive"
    
    async def _calculate_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for segmentation."""
        # Use variance-based importance for now
        # This could be enhanced with more sophisticated methods
        
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'cluster']
        
        feature_importance = {}
        
        for col in numerical_cols:
            # Calculate variance across clusters
            cluster_means = features.groupby('cluster')[col].mean()
            overall_mean = features[col].mean()
            
            # Calculate between-cluster variance
            between_var = ((cluster_means - overall_mean) ** 2).mean()
            total_var = features[col].var()
            
            # Importance as ratio of between-cluster to total variance
            importance = between_var / total_var if total_var > 0 else 0
            feature_importance[col] = float(importance)
        
        # Normalize to sum to 1
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance 
                for k, v in feature_importance.items()
            }
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    async def predict_segment(self, user_data: Dict[str, Any], algorithm: str = "kmeans") -> Dict[str, Any]:
        """
        Predict segment for a new user.
        
        Args:
            user_data: User data dictionary
            algorithm: Clustering algorithm to use for prediction
            
        Returns:
            Predicted segment and confidence
        """
        if algorithm not in self.models:
            raise ValueError(f"No trained model found for algorithm: {algorithm}")
        
        # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Prepare features
        features = await self.prepare_features(user_df)
        X = features[self.feature_columns]
        
        # Handle categorical encoding
        for col in X.columns:
            if col in self.encoders:
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    X[col] = 0
        
        # Scale features
        if 'scaler' in self.scalers:
            X_scaled = self.scalers['scaler'].transform(X)
        else:
            X_scaled = X.values
        
        # Apply UMAP if used during training
        model_info = self.models[algorithm]
        if model_info.get('use_umap', False):
            X_for_prediction = model_info['umap_reducer'].transform(X_scaled)
        else:
            X_for_prediction = X_scaled
        
        # Predict cluster
        clusterer = model_info['clusterer']
        
        if hasattr(clusterer, 'predict'):
            predicted_cluster = clusterer.predict(X_for_prediction)[0]
        else:
            # For algorithms without predict method, find nearest cluster center
            cluster_centers = []
            for cluster_id in range(model_info['n_clusters']):
                cluster_mask = model_info['cluster_labels'] == cluster_id
                if np.any(cluster_mask):
                    # This is a simplified approach - in practice, you'd store cluster centers
                    cluster_centers.append(cluster_id)
            
            # Use the first cluster as default (this should be improved)
            predicted_cluster = 0 if cluster_centers else -1
        
        # Get segment profile
        segment_key = f'segment_{predicted_cluster}'
        segment_profile = self.segment_profiles.get(segment_key, {})
        
        return {
            'predicted_segment': int(predicted_cluster),
            'segment_profile': segment_profile,
            'algorithm_used': algorithm,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    async def create_visualization(self, algorithm: str = "kmeans") -> Dict[str, Any]:
        """
        Create visualization of the segmentation results.
        
        Args:
            algorithm: Algorithm to visualize
            
        Returns:
            Plotly figure data for visualization
        """
        if algorithm not in self.models:
            raise ValueError(f"No trained model found for algorithm: {algorithm}")
        
        model_info = self.models[algorithm]
        
        # For visualization, we'll use PCA to reduce to 2D
        # In practice, you'd store the reduced data from training
        
        # Create a simple scatter plot representation
        visualization_data = {
            'type': 'scatter',
            'data': {
                'x': list(range(len(model_info['cluster_labels']))),
                'y': model_info['cluster_labels'],
                'mode': 'markers',
                'marker': {
                    'color': model_info['cluster_labels'],
                    'colorscale': 'Viridis',
                    'size': 8
                }
            },
            'layout': {
                'title': f'Audience Segmentation - {algorithm.title()}',
                'xaxis': {'title': 'Sample Index'},
                'yaxis': {'title': 'Cluster'},
                'showlegend': False
            }
        }
        
        return visualization_data
    
    @cache_async_result(ttl=3600)
    async def get_segmentation_summary(self) -> Dict[str, Any]:
        """Get summary of all segmentation results."""
        summary = {
            'available_algorithms': list(self.models.keys()),
            'total_segments': {},
            'segment_profiles': self.segment_profiles,
            'last_updated': datetime.now().isoformat()
        }
        
        for algorithm, model_info in self.models.items():
            summary['total_segments'][algorithm] = model_info['n_clusters']
        
        return summary 