"""
Evaluation utilities for SAMI
Classification metrics, visualization, and reporting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from typing import List, Dict
import pandas as pd


def train_knn_classifier(train_embeddings: np.ndarray,
                        train_labels: np.ndarray,
                        n_neighbors: int = 5,
                        metric: str = 'cosine') -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier
    
    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels
        n_neighbors: Number of neighbors
        metric: Distance metric
    
    Returns:
        Trained KNN classifier
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(train_embeddings, train_labels)
    return knn


def evaluate_knn(knn: KNeighborsClassifier,
                test_embeddings: np.ndarray,
                test_labels: np.ndarray,
                class_names: List[str] = None) -> Dict:
    """
    Evaluate KNN classifier and return metrics
    
    Args:
        knn: Trained KNN classifier
        test_embeddings: Test embeddings
        test_labels: Test labels
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    predictions = knn.predict(test_embeddings)
    
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'macro_f1': f1_score(test_labels, predictions, average='macro'),
        'weighted_f1': f1_score(test_labels, predictions, average='weighted'),
        'macro_precision': precision_score(test_labels, predictions, average='macro', zero_division=0),
        'macro_recall': recall_score(test_labels, predictions, average='macro', zero_division=0),
    }
    
    # Classification report
    if class_names is not None:
        report = classification_report(test_labels, predictions,
                                      target_names=class_names,
                                      zero_division=0,
                                      output_dict=True)
        metrics['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    metrics['confusion_matrix'] = cm
    metrics['predictions'] = predictions
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         title: str = 'Confusion Matrix',
                         save_path: str = None,
                         figsize: tuple = (12, 10)):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_tsne(embeddings: np.ndarray,
             labels: np.ndarray,
             class_names: List[str],
             title: str = 't-SNE Visualization',
             save_path: str = None,
             figsize: tuple = (12, 10),
             perplexity: int = 30):
    """
    Create t-SNE visualization of embeddings
    
    Args:
        embeddings: Feature embeddings
        labels: Class labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
    """
    print("Computing t-SNE... (this may take a while)")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    
    # Plot each class
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   label=class_name, alpha=0.6, s=50)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    plt.close()


def compute_class_wise_metrics(true_labels: np.ndarray,
                               predictions: np.ndarray,
                               class_names: List[str]) -> pd.DataFrame:
    """
    Compute per-class metrics
    
    Args:
        true_labels: Ground truth labels
        predictions: Predicted labels
        class_names: List of class names
    
    Returns:
        DataFrame with per-class metrics
    """
    n_classes = len(class_names)
    metrics = []
    
    for class_idx in range(n_classes):
        # Binary classification for this class
        true_binary = (true_labels == class_idx).astype(int)
        pred_binary = (predictions == class_idx).astype(int)
        
        # Compute metrics
        tp = np.sum((true_binary == 1) & (pred_binary == 1))
        fp = np.sum((true_binary == 0) & (pred_binary == 1))
        fn = np.sum((true_binary == 1) & (pred_binary == 0))
        tn = np.sum((true_binary == 0) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        support = np.sum(true_labels == class_idx)
        
        metrics.append({
            'Class': class_names[class_idx],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
    
    df = pd.DataFrame(metrics)
    return df


def save_evaluation_report(metrics: Dict,
                          class_names: List[str],
                          save_path: str):
    """
    Save comprehensive evaluation report
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SAMI - Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n\n")
        
        if 'classification_report' in metrics:
            f.write("Per-Class Metrics:\n")
            f.write("-" * 80 + "\n")
            report = metrics['classification_report']
            
            for class_name in class_names:
                if class_name in report:
                    class_metrics = report[class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {class_metrics['support']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Evaluation report saved to {save_path}")


def compare_k_values(embeddings: np.ndarray,
                    labels: np.ndarray,
                    k_values: List[int] = [1, 3, 5, 7, 9, 11],
                    metric: str = 'cosine') -> pd.DataFrame:
    """
    Compare KNN performance for different k values
    
    Args:
        embeddings: Feature embeddings
        labels: Class labels
        k_values: List of k values to test
        metric: Distance metric
    
    Returns:
        DataFrame with results for each k
    """
    results = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(embeddings, labels)
        predictions = knn.predict(embeddings)
        
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        
        results.append({
            'k': k,
            'accuracy': acc,
            'macro_f1': f1
        })
    
    return pd.DataFrame(results)
