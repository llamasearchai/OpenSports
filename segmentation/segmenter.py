from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class AudienceSegmenter:
    """
    Handles audience segmentation tasks.
    """

    def __init__(self):
        """
        Initializes the AudienceSegmenter.
        """
        logger.info("AudienceSegmenter initialized.")

    def perform_kmeans_segmentation(
        self, 
        feature_vectors: List[List[float]], 
        n_clusters: int,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Performs k-means clustering on the provided feature vectors.

        Args:
            feature_vectors: A list of lists, where each inner list is a feature vector.
            n_clusters: The number of clusters to form.
            random_state: Determines random number generation for centroid initialization.

        Returns:
            A dictionary containing:
                - 'labels': A list of cluster labels for each input vector.
                - 'cluster_centers': A list of coordinates for cluster centers.
                - 'inertia': Sum of squared distances of samples to their closest cluster center.
        """
        if not feature_vectors:
            logger.warn("perform_kmeans_segmentation called with empty feature_vectors.")
            return {"labels": [], "cluster_centers": [], "inertia": 0.0, "error": "Input feature_vectors is empty."}
        
        if n_clusters <= 0:
            logger.warn("perform_kmeans_segmentation called with invalid n_clusters.", n_clusters=n_clusters)
            return {"labels": [], "cluster_centers": [], "inertia": 0.0, "error": "n_clusters must be greater than 0."}

        try:
            X = np.array(feature_vectors)
            if X.ndim != 2:
                logger.warn("perform_kmeans_segmentation: feature_vectors must be a 2D array-like structure.", shape=X.shape)
                return {"labels": [], "cluster_centers": [], "inertia": 0.0, "error": "feature_vectors must be 2D."}

            if X.shape[0] < n_clusters:
                 logger.warn("perform_kmeans_segmentation: n_samples < n_clusters.", n_samples=X.shape[0], n_clusters=n_clusters)
                 #  KMeans would raise ValueError: n_samples=X should be >= n_clusters=Y.
                 #  We can either return an error or adjust n_clusters. For now, returning error.
                 return {"labels": [], "cluster_centers": [], "inertia": 0.0, "error": f"Number of samples ({X.shape[0]}) must be greater than or equal to n_clusters ({n_clusters})."}


            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
            kmeans.fit(X)
            
            labels = kmeans.labels_.tolist()
            cluster_centers = kmeans.cluster_centers_.tolist()
            inertia = kmeans.inertia_

            logger.info("K-means segmentation successful.", n_clusters=n_clusters, num_samples=len(feature_vectors))
            return {
                "labels": labels,
                "cluster_centers": cluster_centers,
                "inertia": inertia
            }
        except Exception as e:
            logger.error("Error during k-means segmentation", exc_info=True)
            return {"labels": [], "cluster_centers": [], "inertia": 0.0, "error": str(e)}

if __name__ == '__main__':
    # Example Usage
    segmenter = AudienceSegmenter()
    
    # Sample data
    sample_features = [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [5.0, 6.0, 7.0],
        [5.1, 6.1, 7.1],
        [10.0, 11.0, 12.0],
        [10.1, 11.1, 12.1]
    ]
    num_clusters = 3
    
    results = segmenter.perform_kmeans_segmentation(sample_features, num_clusters)
    
    if "error" in results:
        print(f"Segmentation failed: {results['error']}")
    else:
        print(f"Segmentation successful!")
        print(f"Labels: {results['labels']}")
        print(f"Cluster Centers: {results['cluster_centers']}")
        print(f"Inertia: {results['inertia']}")

    # Example with too few samples for n_clusters
    sample_features_few = [
        [1.0, 2.0]
    ]
    results_few = segmenter.perform_kmeans_segmentation(sample_features_few, num_clusters)
    if "error" in results_few:
        print(f"Segmentation (few samples) failed: {results_few['error']}")

    # Example with empty data
    results_empty = segmenter.perform_kmeans_segmentation([], num_clusters)
    if "error" in results_empty:
         print(f"Segmentation (empty data) failed: {results_empty['error']}") 