"""Clustering service - clusters conditions using K-Means."""
from typing import List
import numpy as np
from sklearn.cluster import KMeans


class ClusteringService:
    """Service for clustering embeddings using K-Means."""

    def cluster_conditions(
        self,
        embeddings: List[List[float]],
        max_clusters: int = 20,
        max_cluster_size: int = 30,
    ) -> List[int]:
        """
        Cluster embeddings using K-Means with elbow method.
        
        Args:
            embeddings: List of embedding vectors
            max_clusters: Maximum number of clusters to consider
            max_cluster_size: Maximum size for a single cluster
            
        Returns:
            List of cluster assignments (one per embedding)
        """
        if not embeddings or len(embeddings) < 2:
            return list(range(len(embeddings)))
        
        # Convert to numpy array
        data = np.array(embeddings)
        
        # Handle edge case where we have fewer samples than potential clusters
        n_samples = len(embeddings)
        min_k = max(2, n_samples // 10)
        max_k = min(max_clusters, n_samples // 5, n_samples - 1)
        
        if max_k <= min_k:
            # Not enough data for meaningful clustering
            return list(range(n_samples))
        
        # Calculate WCSS for different k values
        wcss_list = []
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss = kmeans.inertia_
            wcss_list.append(wcss)
        
        # Find optimal k using elbow method
        optimal_k_index = self._find_elbow_point(wcss_list)
        optimal_k = min_k + optimal_k_index
        
        # Run final clustering with optimal k
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_assignments = final_kmeans.fit_predict(data)
        
        return cluster_assignments.tolist()

    def _find_elbow_point(self, wcss_list: List[float]) -> int:
        """
        Find the elbow point in WCSS curve.
        
        Args:
            wcss_list: List of WCSS values for different k
            
        Returns:
            Index of the optimal k
        """
        if len(wcss_list) <= 2:
            return 0
        
        max_diff = float("-inf")
        optimal_k_index = 0
        
        for i in range(1, len(wcss_list) - 1):
            diff = wcss_list[i - 1] - wcss_list[i]
            if diff > max_diff:
                max_diff = diff
                optimal_k_index = i
        
        return optimal_k_index

    @staticmethod
    def map_to_clusters(
        items: List,
        cluster_assignments: List[int]
    ) -> dict:
        """
        Map items to their cluster assignments.
        
        Args:
            items: List of items (e.g., requirements)
            cluster_assignments: Cluster assignment for each item
            
        Returns:
            Dictionary mapping cluster ID to list of items
        """
        cluster_map = {}
        for item, cluster_id in zip(items, cluster_assignments):
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = []
            cluster_map[cluster_id].append(item)
        return cluster_map
