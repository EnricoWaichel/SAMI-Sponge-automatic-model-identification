"""
Content-Based Image Retrieval (CBIR) utilities for SAMI
Find similar sponge specimens based on visual features
"""

import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
import torch


def build_index(embeddings: np.ndarray, metric: str = 'cosine', n_neighbors: int = 10):
    """
    Build a nearest neighbor index for CBIR
    
    Args:
        embeddings: Feature embeddings (N x D)
        metric: Distance metric ('cosine', 'euclidean', etc.)
        n_neighbors: Number of neighbors to find
    
    Returns:
        Fitted NearestNeighbors model
    """
    if metric == 'cosine':
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metric = 'euclidean'  # Normalized euclidean = cosine
    
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='auto')
    nn_model.fit(embeddings)
    
    return nn_model


def retrieve_similar(query_embedding: np.ndarray,
                    nn_model: NearestNeighbors,
                    k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve k most similar images to a query
    
    Args:
        query_embedding: Query feature embedding (1 x D or D)
        nn_model: Fitted NearestNeighbors model
        k: Number of results to return
    
    Returns:
        Tuple of (distances, indices)
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    distances, indices = nn_model.kneighbors(query_embedding, n_neighbors=k)
    
    return distances[0], indices[0]


def batch_retrieve(query_embeddings: np.ndarray,
                  nn_model: NearestNeighbors,
                  k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve similar images for multiple queries
    
    Args:
        query_embeddings: Query embeddings (N x D)
        nn_model: Fitted NearestNeighbors model
        k: Number of results per query
    
    Returns:
        Tuple of (distances, indices) arrays
    """
    distances, indices = nn_model.kneighbors(query_embeddings, n_neighbors=k)
    return distances, indices


def compute_recall_at_k(query_labels: np.ndarray,
                       retrieved_indices: np.ndarray,
                       database_labels: np.ndarray,
                       k_values: List[int] = [1, 5, 10]) -> dict:
    """
    Compute Recall@k for retrieval results
    
    Args:
        query_labels: Ground truth labels for queries
        retrieved_indices: Retrieved indices for each query (N x k)
        database_labels: Labels for all database images
        k_values: k values to compute recall for
    
    Returns:
        Dictionary of Recall@k values
    """
    results = {}
    n_queries = len(query_labels)
    
    for k in k_values:
        correct = 0
        for i in range(n_queries):
            query_label = query_labels[i]
            # Get top k retrieved labels
            retrieved_labels = database_labels[retrieved_indices[i, :k]]
            # Check if any match the query label
            if query_label in retrieved_labels:
                correct += 1
        
        recall = correct / n_queries
        results[f'Recall@{k}'] = recall
    
    return results


def compute_precision_at_k(query_labels: np.ndarray,
                           retrieved_indices: np.ndarray,
                           database_labels: np.ndarray,
                           k_values: List[int] = [1, 5, 10]) -> dict:
    """
    Compute Precision@k for retrieval results
    
    Args:
        query_labels: Ground truth labels for queries
        retrieved_indices: Retrieved indices for each query (N x k)
        database_labels: Labels for all database images
        k_values: k values to compute precision for
    
    Returns:
        Dictionary of Precision@k values
    """
    results = {}
    n_queries = len(query_labels)
    
    for k in k_values:
        total_precision = 0.0
        for i in range(n_queries):
            query_label = query_labels[i]
            # Get top k retrieved labels
            retrieved_labels = database_labels[retrieved_indices[i, :k]]
            # Count how many match
            n_relevant = np.sum(retrieved_labels == query_label)
            precision = n_relevant / k
            total_precision += precision
        
        avg_precision = total_precision / n_queries
        results[f'Precision@{k}'] = avg_precision
    
    return results


def compute_map_at_k(query_labels: np.ndarray,
                    retrieved_indices: np.ndarray,
                    database_labels: np.ndarray,
                    k: int = 10) -> float:
    """
    Compute Mean Average Precision at k (MAP@k)
    
    Args:
        query_labels: Ground truth labels for queries
        retrieved_indices: Retrieved indices for each query (N x k)
        database_labels: Labels for all database images
        k: Maximum k value
    
    Returns:
        MAP@k score
    """
    n_queries = len(query_labels)
    average_precisions = []
    
    for i in range(n_queries):
        query_label = query_labels[i]
        retrieved_labels = database_labels[retrieved_indices[i, :k]]
        
        # Compute average precision for this query
        precisions = []
        n_relevant = 0
        
        for j in range(k):
            if retrieved_labels[j] == query_label:
                n_relevant += 1
                precision_at_j = n_relevant / (j + 1)
                precisions.append(precision_at_j)
        
        if len(precisions) > 0:
            avg_precision = np.mean(precisions)
        else:
            avg_precision = 0.0
        
        average_precisions.append(avg_precision)
    
    return np.mean(average_precisions)


def evaluate_cbir(embeddings: np.ndarray,
                 labels: np.ndarray,
                 k_values: List[int] = [1, 5, 10],
                 metric: str = 'cosine') -> dict:
    """
    Evaluate CBIR performance on a dataset
    Uses leave-one-out: each image is a query, rest are database
    
    Args:
        embeddings: Feature embeddings
        labels: Class labels
        k_values: k values for metrics
        metric: Distance metric
    
    Returns:
        Dictionary of metrics
    """
    n_samples = len(embeddings)
    max_k = max(k_values) + 1  # +1 because query itself will be top result
    
    # Normalize if using cosine
    if metric == 'cosine':
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metric = 'euclidean'
    
    # Build index with all samples
    nn_model = NearestNeighbors(n_neighbors=max_k, metric=metric)
    nn_model.fit(embeddings)
    
    # Retrieve for all queries
    _, indices = nn_model.kneighbors(embeddings)
    
    # Remove first column (query itself)
    indices = indices[:, 1:]
    
    # Compute metrics
    results = {}
    results.update(compute_recall_at_k(labels, indices, labels, k_values))
    results.update(compute_precision_at_k(labels, indices, labels, k_values))
    results['MAP@10'] = compute_map_at_k(labels, indices, labels, k=10)
    
    return results


def find_hard_negatives(embeddings: np.ndarray,
                       labels: np.ndarray,
                       k: int = 10) -> List[Tuple[int, int, float]]:
    """
    Find hard negative pairs (similar images from different classes)
    
    Args:
        embeddings: Feature embeddings
        labels: Class labels
        k: Number of neighbors to check
    
    Returns:
        List of (query_idx, negative_idx, distance) tuples
    """
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn_model.fit(embeddings)
    
    distances, indices = nn_model.kneighbors(embeddings)
    
    hard_negatives = []
    
    for i in range(len(embeddings)):
        query_label = labels[i]
        for j, idx in enumerate(indices[i, 1:]):  # Skip first (itself)
            if labels[idx] != query_label:
                hard_negatives.append((i, idx, distances[i, j+1]))
                break  # Only take the closest hard negative
    
    return hard_negatives
