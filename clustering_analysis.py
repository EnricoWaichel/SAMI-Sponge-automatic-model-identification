"""
SAMI - Clustering Analysis
Automatic discovery of sponge groups without prior species labels
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from PIL import Image
import shutil

from vision_transformer import vit_small, vit_base
from utils import load_dataset_images, extract_features, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description='SAMI Clustering Analysis')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to unlabeled images directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model weights')
    parser.add_argument('--model_arch', type=str, default='vit_small',
                       choices=['vit_small', 'vit_base'],
                       help='Model architecture')
    parser.add_argument('--output_dir', type=str, default='./clustering_results',
                       help='Output directory for results')
    parser.add_argument('--n_clusters_range', type=int, nargs='+', 
                       default=[3, 5, 7, 10],
                       help='Range of cluster numbers to test')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical', 'all'],
                       help='Clustering method')
    parser.add_argument('--save_cluster_images', action='store_true',
                       help='Save images organized by cluster')
    parser.add_argument('--max_images_per_cluster', type=int, default=20,
                       help='Max example images to save per cluster')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    return parser.parse_args()


def load_model(args):
    """Load Vision Transformer model"""
    print(f"Loading {args.model_arch} model...")
    
    if args.model_arch == 'vit_small':
        model = vit_small(patch_size=16)
    elif args.model_arch == 'vit_base':
        model = vit_base(patch_size=16)
    
    if args.model_path:
        print(f"Loading weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("Warning: Using random initialization (no pretrained weights)")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return model, device


def compute_tsne(embeddings, perplexity=30, random_state=42):
    """Compute t-SNE for visualization"""
    print(f"Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    coords_2d = tsne.fit_transform(embeddings)
    return coords_2d


def plot_tsne_clusters(coords_2d, labels, title, save_path, n_clusters=None):
    """Plot t-SNE visualization with cluster colors"""
    plt.figure(figsize=(12, 10))
    
    if n_clusters:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = labels == i
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i}', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # For DBSCAN with noise points
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, color in zip(unique_labels, colors):
            if i == -1:
                # Noise points in black
                color = 'black'
                label = 'Noise'
            else:
                label = f'Cluster {i}'
            
            mask = labels == i
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[color], label=label, alpha=0.6, s=50,
                       edgecolors='black', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


def kmeans_clustering(embeddings, n_clusters_range, coords_2d, output_dir):
    """Perform K-Means clustering with multiple k values"""
    print("\n" + "="*80)
    print("K-MEANS CLUSTERING")
    print("="*80)
    
    results = []
    
    for n_clusters in n_clusters_range:
        print(f"\nTesting k={n_clusters}...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute metrics
        silhouette = silhouette_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)
        inertia = kmeans.inertia_
        
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, >0.5 is good)")
        print(f"  Calinski-Harabasz: {calinski:.2f} (higher is better)")
        print(f"  Inertia: {inertia:.2f} (lower is better)")
        
        # Count samples per cluster
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Samples per cluster: {dict(zip(unique, counts))}")
        
        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'inertia': inertia,
            'labels': labels
        })
        
        # Plot
        plot_tsne_clusters(
            coords_2d, labels,
            f'K-Means Clustering (k={n_clusters})\nSilhouette: {silhouette:.3f}',
            output_dir / f'kmeans_k{n_clusters}.png',
            n_clusters=n_clusters
        )
    
    # Save comparison
    df = pd.DataFrame([{
        'n_clusters': r['n_clusters'],
        'silhouette': r['silhouette'],
        'calinski_harabasz': r['calinski_harabasz'],
        'inertia': r['inertia']
    } for r in results])
    
    df.to_csv(output_dir / 'kmeans_comparison.csv', index=False)
    print(f"\nComparison saved to {output_dir / 'kmeans_comparison.csv'}")
    
    # Find best k by silhouette score
    best_idx = df['silhouette'].idxmax()
    best_k = df.loc[best_idx, 'n_clusters']
    print(f"\nBest k by Silhouette Score: {int(best_k)}")
    
    return results


def dbscan_clustering(embeddings, coords_2d, output_dir):
    """Perform DBSCAN clustering (density-based)"""
    print("\n" + "="*80)
    print("DBSCAN CLUSTERING (Density-Based)")
    print("="*80)
    
    # Try different eps values
    eps_values = [0.3, 0.5, 0.7, 1.0]
    
    for eps in eps_values:
        print(f"\nTesting eps={eps}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
        labels = dbscan.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        if n_clusters > 1:
            # Only compute silhouette if we have at least 2 clusters
            valid_mask = labels != -1
            if valid_mask.sum() > 1:
                silhouette = silhouette_score(embeddings[valid_mask], labels[valid_mask])
                print(f"  Silhouette Score: {silhouette:.4f}")
        
        # Count samples per cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        print(f"  Samples per cluster: {cluster_sizes}")
        
        # Plot
        plot_tsne_clusters(
            coords_2d, labels,
            f'DBSCAN Clustering (eps={eps})\nClusters: {n_clusters}, Noise: {n_noise}',
            output_dir / f'dbscan_eps{eps}.png'
        )


def hierarchical_clustering(embeddings, coords_2d, output_dir, n_clusters=5):
    """Perform Hierarchical clustering with dendrogram"""
    print("\n" + "="*80)
    print("HIERARCHICAL CLUSTERING")
    print("="*80)
    
    # Compute linkage matrix
    print("Computing linkage matrix...")
    linkage_matrix = linkage(embeddings, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, no_labels=True)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_dir / 'hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dendrogram saved to {output_dir / 'hierarchical_dendrogram.png'}")
    
    # Cut tree at different heights
    for n in [3, 5, 7, 10]:
        clustering = AgglomerativeClustering(n_clusters=n, linkage='ward')
        labels = clustering.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, labels)
        print(f"\nn_clusters={n}: Silhouette={silhouette:.4f}")
        
        # Count samples
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Samples per cluster: {dict(zip(unique, counts))}")
        
        # Plot
        plot_tsne_clusters(
            coords_2d, labels,
            f'Hierarchical Clustering (n={n})\nSilhouette: {silhouette:.3f}',
            output_dir / f'hierarchical_n{n}.png',
            n_clusters=n
        )


def save_cluster_examples(image_paths, labels, output_dir, max_per_cluster=20):
    """Save example images for each cluster"""
    print("\n" + "="*80)
    print("SAVING CLUSTER EXAMPLES")
    print("="*80)
    
    clusters_dir = output_dir / 'cluster_images'
    clusters_dir.mkdir(exist_ok=True)
    
    unique_labels = sorted(set(labels))
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            cluster_name = 'noise'
        else:
            cluster_name = f'cluster_{cluster_id}'
        
        cluster_dir = clusters_dir / cluster_name
        cluster_dir.mkdir(exist_ok=True)
        
        # Get images in this cluster
        cluster_mask = labels == cluster_id
        cluster_paths = [image_paths[i] for i in range(len(labels)) if cluster_mask[i]]
        
        # Save up to max_per_cluster examples
        for i, img_path in enumerate(cluster_paths[:max_per_cluster]):
            src = Path(img_path)
            dst = cluster_dir / f'{i:03d}_{src.name}'
            shutil.copy(src, dst)
        
        print(f"  {cluster_name}: {len(cluster_paths)} images ({min(len(cluster_paths), max_per_cluster)} saved)")
    
    print(f"\nCluster examples saved to {clusters_dir}")


def generate_report(output_dir):
    """Generate summary report"""
    report_path = output_dir / 'clustering_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAMI - Clustering Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("This report contains automatic clustering results for unlabeled sponge images.\n\n")
        
        f.write("Files Generated:\n")
        f.write("-"*80 + "\n")
        f.write("- kmeans_k*.png: K-Means clustering visualizations\n")
        f.write("- kmeans_comparison.csv: Comparison of different k values\n")
        f.write("- dbscan_eps*.png: DBSCAN clustering with different epsilon values\n")
        f.write("- hierarchical_*.png: Hierarchical clustering results\n")
        f.write("- hierarchical_dendrogram.png: Dendrogram showing similarity tree\n")
        f.write("- cluster_images/: Example images from each cluster\n\n")
        
        f.write("How to Interpret Results:\n")
        f.write("-"*80 + "\n")
        f.write("1. Silhouette Score (0 to 1):\n")
        f.write("   - >0.7: Strong, well-separated clusters\n")
        f.write("   - 0.5-0.7: Good clustering\n")
        f.write("   - 0.25-0.5: Weak clustering, overlapping groups\n")
        f.write("   - <0.25: Poor clustering\n\n")
        
        f.write("2. t-SNE Plots:\n")
        f.write("   - Points close together = similar images\n")
        f.write("   - Well-separated colored groups = distinct morphologies\n")
        f.write("   - Mixed colors = overlapping characteristics\n\n")
        
        f.write("3. Dendrogram:\n")
        f.write("   - Shows hierarchical relationships\n")
        f.write("   - Longer vertical lines = more distinct groups\n")
        f.write("   - Can help decide optimal number of clusters\n\n")
        
        f.write("Next Steps:\n")
        f.write("-"*80 + "\n")
        f.write("1. Review cluster_images/ folder\n")
        f.write("2. Visually inspect if clusters make sense morphologically\n")
        f.write("3. Choose best clustering method and number of clusters\n")
        f.write("4. Rename cluster folders to species/morphotype names\n")
        f.write("5. Use renamed structure for supervised training\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nReport saved to {report_path}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SAMI - Clustering Analysis")
    print("Automatic Discovery of Sponge Groups")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading images...")
    images_tensor, image_paths, _ = load_dataset_images(args.data_path, img_size=224)
    
    # Load model and extract features
    model, device = load_model(args)
    
    print("\nExtracting features...")
    embeddings = extract_features(model, images_tensor, batch_size=32, device=device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Normalize embeddings for better clustering
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute t-SNE
    coords_2d = compute_tsne(embeddings, perplexity=args.perplexity)
    
    # Perform clustering
    if args.method == 'kmeans' or args.method == 'all':
        results = kmeans_clustering(embeddings, args.n_clusters_range, coords_2d, output_dir)
        
        # Save images for best k
        if args.save_cluster_images and results:
            best_idx = np.argmax([r['silhouette'] for r in results])
            best_labels = results[best_idx]['labels']
            save_cluster_examples(image_paths, best_labels, output_dir, args.max_images_per_cluster)
    
    if args.method == 'dbscan' or args.method == 'all':
        dbscan_clustering(embeddings, coords_2d, output_dir)
    
    if args.method == 'hierarchical' or args.method == 'all':
        hierarchical_clustering(embeddings, coords_2d, output_dir)
    
    # Generate report
    generate_report(output_dir)
    
    print("\n" + "="*80)
    print("Clustering Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the clustering visualizations")
    print("2. Inspect cluster_images/ folder")
    print("3. Choose the clustering that best represents your data")
    print("4. Organize your images based on the best clustering")
    print("="*80)


if __name__ == "__main__":
    main()
