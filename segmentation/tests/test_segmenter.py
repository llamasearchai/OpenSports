import unittest
import numpy as np
from OpenInsight.segmentation.segmenter import AudienceSegmenter

class TestAudienceSegmenter(unittest.TestCase):

    def setUp(self):
        self.segmenter = AudienceSegmenter()
        self.sample_data = np.array([
            [1, 2, 3],
            [1, 2, 4],
            [5, 6, 7],
            [5, 6, 8],
            [9, 0, 1]
        ])

    def test_perform_kmeans_segmentation_successful(self):
        n_clusters = 3
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation(self.sample_data.tolist(), n_clusters)
        self.assertIsNone(error)
        self.assertIsNotNone(labels)
        self.assertIsNotNone(centroids)
        self.assertEqual(len(labels), len(self.sample_data))
        # Kmeans might sometimes produce fewer clusters if data points are identical or k is too high
        # For this specific test data and n_clusters=3, it should produce 3.
        self.assertEqual(len(np.unique(labels)), n_clusters)
        self.assertEqual(centroids.shape, (n_clusters, self.sample_data.shape[1]))

    def test_perform_kmeans_segmentation_n_clusters_too_high(self):
        # n_clusters > n_samples
        n_clusters = len(self.sample_data) + 1
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation(self.sample_data.tolist(), n_clusters)
        self.assertIsNotNone(error)        
        self.assertIn(f"Number of clusters {n_clusters} exceeds number of samples {len(self.sample_data)}", error)
        self.assertIsNone(labels)
        self.assertIsNone(centroids)

    def test_perform_kmeans_segmentation_empty_data(self):
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation([], 3)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Input feature_vectors cannot be empty.")
        self.assertIsNone(labels)
        self.assertIsNone(centroids)

    def test_perform_kmeans_segmentation_n_clusters_zero(self):
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation(self.sample_data.tolist(), 0)
        self.assertIsNotNone(error)
        # This specific error message comes from scikit-learn's KMeans
        self.assertIn("Number of clusters should be an integer greater than 0. Got 0 instead.", error)
        self.assertIsNone(labels)
        self.assertIsNone(centroids)
        
    def test_perform_kmeans_segmentation_n_clusters_one(self):
        n_clusters = 1
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation(self.sample_data.tolist(), n_clusters)
        self.assertIsNone(error)
        self.assertIsNotNone(labels)
        self.assertIsNotNone(centroids)
        self.assertEqual(len(labels), len(self.sample_data))
        self.assertEqual(len(np.unique(labels)), n_clusters)
        self.assertEqual(centroids.shape, (n_clusters, self.sample_data.shape[1]))

    def test_input_data_not_list_of_lists_of_numbers(self):
        # Test with a list of numbers instead of list of lists of numbers
        invalid_data = [1, 2, 3, 4, 5]
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation(invalid_data, 2) # type: ignore
        self.assertIsNotNone(error)
        # This error comes from _validate_data in scikit-learn
        self.assertIn("Expected 2D array, got 1D array instead", error)
        self.assertIsNone(labels)
        self.assertIsNone(centroids)

    def test_input_data_inconsistent_vector_lengths(self):
        invalid_data = [[1, 2], [3, 4, 5]] # Ragged list
        labels, centroids, error = self.segmenter.perform_kmeans_segmentation(invalid_data, 2)
        self.assertIsNotNone(error)
        # This error is from numpy trying to create an array from a ragged list, then caught by _validate_data in sklearn
        self.assertIn("setting an array element with a sequence.", error)
        self.assertIsNone(labels)
        self.assertIsNone(centroids)

if __name__ == '__main__':
    unittest.main() 