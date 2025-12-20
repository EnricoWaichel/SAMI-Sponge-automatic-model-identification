"""
SAMI - Multi-Scale Patch Clustering (Memory Optimized)
Extract convolutional windows at different scales, cluster them, and visualize
OPTIMIZED: Processes images one-by-one to avoid memory errors
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm
import pandas as pd
import gc

from vision_transformer import vit_small
from utils import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Scale Patch Clustering (Optimized)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory (containing leptomitid/ and choialike/)')
    parser.add_argument('--output_dir', type=str, default='./patch_clustering_results',
                       help='Output directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model weights')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[64, 128, 256],
                       help='Window sizes for patch extraction')
    parser.add_argument('--stride', type=int, default=32,
                       help='Stride for sliding window')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters')
    parser.add_argument('--viz_method', type=str, default='tsne',
                       choices=['tsne', 'pca', 'both'],
                       help='Visualization method')
    parser.add_argument('--max_patches_per_image', type=int, default=50,
                       help='Maximum patches to extract per image (to limit memory)')
    parser.add_argument('--feature_batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    return parser.parse_args()


def extract_patches_from_image(image_path, window_sizes, stride, max_patches=50):
    """
    Extract patches from a SINGLE image (memory efficient)
    
    Args:
        image_path: Path to image
        window_sizes: List of window sizes
        stride: Stride for sliding window
        max_patches: Maximum patches to extract (to limit memory)
    
    Returns:
        List of (patch_array, metadata) tuples
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    patches = []
    
    for window_size in window_sizes:
        # Calculate how many patches we'll get
        n_patches_h = (h - window_size) // stride + 1
        n_patches_w = (w - window_size) // stride + 1
        total_patches_this_size = n_patches_h * n_patches_w
        
        # Sample patches uniformly if too many
        if total_patches_this_size > max_patches // len(window_sizes):
            # Sample uniformly
            step_h = max(1, n_patches_h // int(np.sqrt(max_patches // len(window_sizes))))
            step_w = max(1, n_patches_w // int(np.sqrt(max_patches // len(window_sizes))))
        else:
            step_h = 1
            step_w = 1
        
        count = 0
        for i, y in enumerate(range(0, h - window_size + 1, stride)):
            if i % step_h != 0:
                continue
            for j, x in enumerate(range(0, w - window_size + 1, stride)):
                if j % step_w != 0:
                    continue
                
                patch = img_rgb[y:y+window_size, x:x+window_size].copy()
                
                metadata = {
                    'image_path': str(image_path),
                    'image_name': image_path.stem,
                    'window_size': window_size,
                    'x': x,
                    'y': y,
                    'coords': f'{x}_{y}'
                }
                
                patches.append((patch, metadata))
                count += 1
                
                if count >= max_patches // len(window_sizes):
                    break
            if count >= max_patches // len(window_sizes):
                break
    
    # Clean up
    del img, img_rgb
    gc.collect()
    
    return patches


def process_dataset_iteratively(data_path, window_sizes, stride, max_patches_per_image):
    """
    Process dataset ONE IMAGE AT A TIME (memory efficient)
    """
    data_path = Path(data_path)
    
    # Get all class folders
    class_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
    
    # Collect all image paths first
    all_image_paths = []
    class_names = []
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        image_files = list(class_folder.glob('*.jpg')) + \
                     list(class_folder.glob('*.jpeg')) + \
                     list(class_folder.glob('*.png'))
        
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            all_image_paths.append(img_file)
            class_names.append(class_name)
    
    return all_image_paths, class_names


def extract_all_patches(image_paths, class_names, window_sizes, stride, max_patches_per_image):
    """
    Extract patches from all images, processing one at a time
    """
    all_patches = []
    
    print(f"\nExtracting patches from {len(image_paths)} images...")
    
    for img_path, class_name in tqdm(zip(image_paths, class_names), total=len(image_paths)):
        patches = extract_patches_from_image(img_path, window_sizes, stride, max_patches_per_image)
        
        # Add class to metadata
        for patch, metadata in patches:
            metadata['class'] = class_name
            metadata['prefix'] = f"{class_name}/{metadata['image_name']}/window_{metadata['window_size']}"
            all_patches.append((patch, metadata))
        
        # Force garbage collection every 10 images
        if len(all_patches) % 100 == 0:
            gc.collect()
    
    print(f"Total patches extracted: {len(all_patches)}")
    
    # Print statistics
    for class_name in set(class_names):
        n_patches = sum(1 for _, m in all_patches if m['class'] == class_name)
        print(f"  {class_name}: {n_patches} patches")
    
    return all_patches


def extract_features_from_patches(patches, model, device, batch_size=32):
    """
    Extract features from patches using ViT model (batch processing)
    """
    print("\nExtracting features from patches...")
    
    model.eval()
    all_features = []
    
    # Process in batches
    for i in tqdm(range(0, len(patches), batch_size)):
        batch_patches = patches[i:i+batch_size]
        
        # Prepare batch
        batch_tensors = []
        for patch_img, _ in batch_patches:
            # Resize to 224x224 for ViT
            patch_pil = Image.fromarray(patch_img)
            patch_tensor = preprocess_image(patch_pil, img_size=224)
            batch_tensors.append(patch_tensor)
        
        batch = torch.stack(batch_tensors).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(batch)
        
        all_features.append(features.cpu().numpy())
        
        # Clean up
        del batch, features
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Garbage collection every few batches
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    features_array = np.vstack(all_features)
    print(f"Features shape: {features_array.shape}")
    
    return features_array


def cluster_patches(features, n_clusters, metadata_list):
    """
    Cluster patches based on features
    """
    print(f"\nClustering patches into {n_clusters} groups...")
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features_norm)
    
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(features_norm, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_patches = mask.sum()
        
        # Count by class
        classes = [metadata_list[i]['class'] for i in range(len(labels)) if mask[i]]
        class_counts = {}
        for c in classes:
            class_counts[c] = class_counts.get(c, 0) + 1
        
        print(f"  Cluster {cluster_id}: {n_patches} patches - {class_counts}")
    
    return labels, kmeans, silhouette


def visualize_tsne(features, labels, metadata_list, output_path, silhouette_score):
    """Create t-SNE visualization"""
    print("\nComputing t-SNE (this may take a few minutes)...")
    
    # Use fewer samples if too many
    if len(features) > 5000:
        print(f"  Sampling 5000 patches from {len(features)} for faster t-SNE...")
        indices = np.random.choice(len(features), 5000, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
        metadata_sample = [metadata_list[i] for i in indices]
    else:
        features_sample = features
        labels_sample = labels
        metadata_sample = metadata_list
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample) // 4))
    coords_2d = tsne.fit_transform(features_sample)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Colored by cluster
    ax1 = axes[0]
    n_clusters = len(set(labels_sample))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels_sample == cluster_id
        if mask.sum() > 0:
            ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    
    ax1.set_title(f't-SNE: Colored by Cluster\nSilhouette Score: {silhouette_score:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Plot 2: Colored by class
    ax2 = axes[1]
    classes = list(set([m['class'] for m in metadata_sample]))
    class_colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, class_name in enumerate(classes):
        mask = np.array([m['class'] == class_name for m in metadata_sample])
        if mask.sum() > 0:
            ax2.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[class_colors[i]], label=class_name,
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    
    ax2.set_title('t-SNE: Colored by Original Class', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to {output_path}")


def visualize_pca(features, labels, metadata_list, output_path):
    """Create PCA visualization"""
    print("\nComputing PCA...")
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(features)
    
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Colored by cluster
    ax1 = axes[0]
    n_clusters = len(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    
    ax1.set_title('PCA: Colored by Cluster', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Plot 2: Colored by class
    ax2 = axes[1]
    classes = list(set([m['class'] for m in metadata_list]))
    class_colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, class_name in enumerate(classes):
        mask = np.array([m['class'] == class_name for m in metadata_list])
        ax2.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[class_colors[i]], label=class_name,
                   alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    
    ax2.set_title('PCA: Colored by Original Class', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA plot saved to {output_path}")


def save_cluster_summary(labels, metadata_list, output_path):
    """Save detailed cluster summary"""
    print("\nSaving cluster summary...")
    
    data = []
    for i, (label, metadata) in enumerate(zip(labels, metadata_list)):
        data.append({
            'patch_id': i,
            'cluster': label,
            'class': metadata['class'],
            'image_name': metadata['image_name'],
            'window_size': metadata['window_size'],
            'x': metadata['x'],
            'y': metadata['y'],
            'prefix': metadata['prefix']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Cluster summary saved to {output_path}")
    
    # Print cross-tabulation
    print("\nCross-tabulation: Class vs Cluster")
    ct = pd.crosstab(df['class'], df['cluster'], margins=True)
    print(ct)
    
    print("\nCross-tabulation: Window Size vs Cluster")
    ct2 = pd.crosstab(df['window_size'], df['cluster'], margins=True)
    print(ct2)


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SAMI - Multi-Scale Patch Clustering (Memory Optimized)")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Window sizes: {args.window_sizes}")
    print(f"Stride: {args.stride}")
    print(f"Max patches per image: {args.max_patches_per_image}")
    print(f"Number of clusters: {args.n_clusters}")
    print("="*80 + "\n")
    
    # Step 1: Get all image paths
    image_paths, class_names = process_dataset_iteratively(
        Path(args.data_path), 
        args.window_sizes, 
        args.stride,
        args.max_patches_per_image
    )
    
    # Step 2: Extract patches (one image at a time)
    all_patches = extract_all_patches(
        image_paths, 
        class_names, 
        args.window_sizes, 
        args.stride,
        args.max_patches_per_image
    )
    
    if len(all_patches) == 0:
        print("ERROR: No patches extracted! Check your images.")
        return
    
    # Step 3: Load model
    print("\nLoading Vision Transformer model...")
    model = vit_small(patch_size=16)
    
    if args.model_path:
        print(f"Loading weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("Warning: Using random initialization (results will be poor)")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Step 4: Extract features
    features = extract_features_from_patches(all_patches, model, device, args.feature_batch_size)
    
    # Extract metadata
    metadata_list = [metadata for _, metadata in all_patches]
    
    # Clean up patches to free memory
    del all_patches
    gc.collect()
    
    # Step 5: Cluster
    labels, kmeans, silhouette = cluster_patches(features, args.n_clusters, metadata_list)
    
    # Step 6: Visualize
    if args.viz_method in ['tsne', 'both']:
        visualize_tsne(features, labels, metadata_list, 
                      output_dir / 'tsne_visualization.png', silhouette)
    
    if args.viz_method in ['pca', 'both']:
        visualize_pca(features, labels, metadata_list,
                     output_dir / 'pca_visualization.png')
    
    # Step 7: Save results
    save_cluster_summary(labels, metadata_list, output_dir / 'cluster_summary.csv')
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print("\nGenerated files:")
    if args.viz_method in ['tsne', 'both']:
        print("  - tsne_visualization.png")
    if args.viz_method in ['pca', 'both']:
        print("  - pca_visualization.png")
    print("  - cluster_summary.csv")
    print("="*80)


if __name__ == "__main__":
    main()
