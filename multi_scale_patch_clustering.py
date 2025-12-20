"""
SAMI - Multi-Scale Patch Clustering (Enhanced Version)
Improvements based on feedback:
- UMAP instead of t-SNE
- Percentile normalization (5%-95%)
- Border/background removal
- RGB pattern preservation
- Patch visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import cv2
from tqdm import tqdm
import pandas as pd
import gc

from vision_transformer import vit_small
from utils import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Scale Patch Clustering (Enhanced)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory')
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
    parser.add_argument('--viz_method', type=str, default='umap',
                       choices=['umap', 'pca', 'both'],
                       help='Visualization method')
    parser.add_argument('--max_patches_per_image', type=int, default=50,
                       help='Maximum patches per image')
    parser.add_argument('--feature_batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--min_content_ratio', type=float, default=0.7,
                       help='Minimum content ratio to accept patch (0.7 = 70%% non-background)')
    parser.add_argument('--visualize_patches', action='store_true',
                       help='Save visualization of extracted patches')
    return parser.parse_args()


def percentile_normalize_image(img, lower_percentile=5, upper_percentile=95):
    """
    Normalize image using percentile clipping for better contrast
    Apply per-channel normalization to preserve RGB patterns
    
    Args:
        img: RGB image array (H, W, 3)
        lower_percentile: Lower percentile to clip (default 5%%)
        upper_percentile: Upper percentile to clip (default 95%%)
    
    Returns:
        Normalized image in range [0, 255]
    """
    img_normalized = np.zeros_like(img, dtype=np.float32)
    
    # Normalize each channel independently
    for c in range(3):  # R, G, B
        channel = img[:, :, c].astype(np.float32)
        
        # Calculate percentiles
        p_low = np.percentile(channel, lower_percentile)
        p_high = np.percentile(channel, upper_percentile)
        
        # Clip and normalize to 0-255
        channel_clipped = np.clip(channel, p_low, p_high)
        
        if p_high > p_low:
            channel_normalized = (channel_clipped - p_low) / (p_high - p_low) * 255.0
        else:
            channel_normalized = channel_clipped
        
        img_normalized[:, :, c] = channel_normalized
    
    return img_normalized.astype(np.uint8)


def is_valid_patch(patch, min_content_ratio=0.7):
    """
    Check if patch contains enough content (not mostly background/border)
    
    Args:
        patch: RGB patch array
        min_content_ratio: Minimum ratio of non-background pixels
    
    Returns:
        True if patch is valid, False if mostly background
    """
    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    
    # Simple threshold: background is usually white (>240) or very dark (<15)
    background_mask = (gray > 240) | (gray < 15)
    
    # Calculate content ratio
    content_ratio = 1.0 - (np.sum(background_mask) / background_mask.size)
    
    return content_ratio >= min_content_ratio


def extract_patches_from_image(image_path, window_sizes, stride, max_patches=50, min_content_ratio=0.7):
    """
    Extract patches from image with:
    - Percentile normalization (5%%-95%%)
    - Background removal
    - RGB preservation
    """
    # Read image in RGB (not BGR!)
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply percentile normalization to each channel
    img_normalized = percentile_normalize_image(img_rgb, lower_percentile=5, upper_percentile=95)
    
    h, w = img_normalized.shape[:2]
    
    patches = []
    
    for window_size in window_sizes:
        valid_patches_this_size = 0
        max_per_size = max_patches // len(window_sizes)
        
        # Sample positions
        n_patches_h = (h - window_size) // stride + 1
        n_patches_w = (w - window_size) // stride + 1
        
        # Create grid of positions
        positions_y = list(range(0, h - window_size + 1, stride))
        positions_x = list(range(0, w - window_size + 1, stride))
        
        # Shuffle to get random sampling
        np.random.seed(42)
        positions = [(y, x) for y in positions_y for x in positions_x]
        np.random.shuffle(positions)
        
        for y, x in positions:
            if valid_patches_this_size >= max_per_size:
                break
            
            patch = img_normalized[y:y+window_size, x:x+window_size].copy()
            
            # Check if patch has enough content (not mostly background)
            if not is_valid_patch(patch, min_content_ratio):
                continue
            
            metadata = {
                'image_path': str(image_path),
                'image_name': image_path.stem,
                'window_size': window_size,
                'x': x,
                'y': y,
                'coords': f'{x}_{y}'
            }
            
            patches.append((patch, metadata))
            valid_patches_this_size += 1
    
    # Clean up
    del img, img_rgb, img_normalized
    gc.collect()
    
    return patches


def process_dataset_iteratively(data_path, window_sizes, stride, max_patches_per_image, min_content_ratio):
    """Process dataset and collect image paths"""
    data_path = Path(data_path)
    
    class_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_folders)} classes: {[d.name for d in class_folders]}")
    
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


def extract_all_patches(image_paths, class_names, window_sizes, stride, max_patches_per_image, min_content_ratio):
    """Extract patches from all images with enhanced preprocessing"""
    all_patches = []
    
    print(f"\nExtracting patches with:")
    print(f"  - Percentile normalization (5%%-95%%)")
    print(f"  - Background removal (min content: {min_content_ratio*100}%%)")
    print(f"  - RGB pattern preservation")
    print()
    
    for img_path, class_name in tqdm(zip(image_paths, class_names), total=len(image_paths)):
        patches = extract_patches_from_image(
            img_path, window_sizes, stride, max_patches_per_image, min_content_ratio
        )
        
        for patch, metadata in patches:
            metadata['class'] = class_name
            metadata['prefix'] = f"{class_name}/{metadata['image_name']}/window_{metadata['window_size']}"
            all_patches.append((patch, metadata))
        
        if len(all_patches) % 100 == 0:
            gc.collect()
    
    print(f"Total patches extracted: {len(all_patches)}")
    
    for class_name in set(class_names):
        n_patches = sum(1 for _, m in all_patches if m['class'] == class_name)
        print(f"  {class_name}: {n_patches} patches")
    
    return all_patches


def extract_features_from_patches(patches, model, device, batch_size=32):
    """Extract features preserving RGB information"""
    print("\nExtracting features from patches...")
    
    model.eval()
    all_features = []
    
    for i in tqdm(range(0, len(patches), batch_size)):
        batch_patches = patches[i:i+batch_size]
        
        batch_tensors = []
        for patch_img, _ in batch_patches:
            # Convert to PIL maintaining RGB
            patch_pil = Image.fromarray(patch_img)  # Already in RGB!
            patch_tensor = preprocess_image(patch_pil, img_size=224)
            batch_tensors.append(patch_tensor)
        
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            features = model(batch)
        
        all_features.append(features.cpu().numpy())
        
        del batch, features
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    features_array = np.vstack(all_features)
    print(f"Features shape: {features_array.shape}")
    
    return features_array


def cluster_patches(features, n_clusters, metadata_list):
    """Cluster patches"""
    print(f"\nClustering patches into {n_clusters} groups...")
    
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features_norm)
    
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(features_norm, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    print("\nCluster Statistics:")
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_patches = mask.sum()
        
        classes = [metadata_list[i]['class'] for i in range(len(labels)) if mask[i]]
        class_counts = {}
        for c in classes:
            class_counts[c] = class_counts.get(c, 0) + 1
        
        print(f"  Cluster {cluster_id}: {n_patches} patches - {class_counts}")
    
    return labels, kmeans, silhouette


def visualize_umap(features, labels, metadata_list, output_path, silhouette_score):
    """Create UMAP visualization (better than t-SNE!)"""
    print("\nComputing UMAP projection...")
    
    # Sample if too many points
    if len(features) > 5000:
        print(f"  Sampling 5000 patches from {len(features)} for visualization...")
        indices = np.random.choice(len(features), 5000, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
        metadata_sample = [metadata_list[i] for i in indices]
    else:
        features_sample = features
        labels_sample = labels
        metadata_sample = metadata_list
    
    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    coords_2d = reducer.fit_transform(features_sample)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: By cluster
    ax1 = axes[0]
    n_clusters = len(set(labels_sample))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels_sample == cluster_id
        if mask.sum() > 0:
            ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    
    ax1.set_title(f'UMAP: Colored by Cluster\nSilhouette Score: {silhouette_score:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Plot 2: By class
    ax2 = axes[1]
    classes = list(set([m['class'] for m in metadata_sample]))
    class_colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, class_name in enumerate(classes):
        mask = np.array([m['class'] == class_name for m in metadata_sample])
        if mask.sum() > 0:
            ax2.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[class_colors[i]], label=class_name,
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    
    ax2.set_title('UMAP: Colored by Original Class', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP plot saved to {output_path}")


def visualize_pca(features, labels, metadata_list, output_path):
    """Create PCA visualization"""
    print("\nComputing PCA...")
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(features)
    
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # By cluster
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
    
    # By class
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


def visualize_patch_examples(patches, labels, metadata_list, output_dir, n_examples=5):
    """Visualize example patches from each cluster"""
    print("\nCreating patch visualization...")
    
    n_clusters = len(set(labels))
    
    fig, axes = plt.subplots(n_clusters, n_examples, figsize=(n_examples*2, n_clusters*2))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for cluster_id in range(n_clusters):
        # Get patches from this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        
        # Sample random examples
        if len(cluster_indices) > n_examples:
            sampled_indices = np.random.choice(cluster_indices, n_examples, replace=False)
        else:
            sampled_indices = cluster_indices[:n_examples]
        
        for col_idx, patch_idx in enumerate(sampled_indices):
            patch_img, metadata = patches[patch_idx]
            
            axes[cluster_id, col_idx].imshow(patch_img)
            axes[cluster_id, col_idx].axis('off')
            
            if col_idx == 0:
                axes[cluster_id, col_idx].set_ylabel(f'Cluster {cluster_id}', 
                                                     fontsize=10, fontweight='bold')
            
            # Add class label
            class_name = metadata['class']
            axes[cluster_id, col_idx].set_title(class_name, fontsize=8)
    
    plt.suptitle('Example Patches per Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'patch_examples.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Patch examples saved to {output_dir / 'patch_examples.png'}")


def save_cluster_summary(labels, metadata_list, output_path):
    """Save cluster summary"""
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
    
    print("\nCross-tabulation: Class vs Cluster")
    ct = pd.crosstab(df['class'], df['cluster'], margins=True)
    print(ct)
    
    print("\nCross-tabulation: Window Size vs Cluster")
    ct2 = pd.crosstab(df['window_size'], df['cluster'], margins=True)
    print(ct2)


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SAMI - Multi-Scale Patch Clustering (Enhanced with UMAP)")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Window sizes: {args.window_sizes}")
    print(f"Stride: {args.stride}")
    print(f"Max patches per image: {args.max_patches_per_image}")
    print(f"Min content ratio: {args.min_content_ratio}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Visualization: {args.viz_method}")
    print("="*80 + "\n")
    
    # Step 1: Get image paths
    image_paths, class_names = process_dataset_iteratively(
        Path(args.data_path), args.window_sizes, args.stride, 
        args.max_patches_per_image, args.min_content_ratio
    )
    
    # Step 2: Extract patches with enhanced preprocessing
    all_patches = extract_all_patches(
        image_paths, class_names, args.window_sizes, 
        args.stride, args.max_patches_per_image, args.min_content_ratio
    )
    
    if len(all_patches) == 0:
        print("ERROR: No patches extracted!")
        return
    
    # Step 3: Load model
    print("\nLoading Vision Transformer...")
    model = vit_small(patch_size=16)
    
    if args.model_path:
        print(f"Loading weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("Warning: Using random initialization")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model on {device}")
    
    # Step 4: Extract features
    features = extract_features_from_patches(all_patches, model, device, args.feature_batch_size)
    
    metadata_list = [metadata for _, metadata in all_patches]
    
    # Step 5: Cluster
    labels, kmeans, silhouette = cluster_patches(features, args.n_clusters, metadata_list)
    
    # Step 6: Visualize
    if args.viz_method in ['umap', 'both']:
        visualize_umap(features, labels, metadata_list, 
                      output_dir / 'umap_visualization.png', silhouette)
    
    if args.viz_method in ['pca', 'both']:
        visualize_pca(features, labels, metadata_list,
                     output_dir / 'pca_visualization.png')
    
    # Step 7: Visualize patches
    if args.visualize_patches:
        visualize_patch_examples(all_patches, labels, metadata_list, output_dir)
    
    # Step 8: Save results
    save_cluster_summary(labels, metadata_list, output_dir / 'cluster_summary.csv')
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Results: {output_dir}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print("\nGenerated files:")
    if args.viz_method in ['umap', 'both']:
        print("  - umap_visualization.png")
    if args.viz_method in ['pca', 'both']:
        print("  - pca_visualization.png")
    if args.visualize_patches:
        print("  - patch_examples.png")
    print("  - cluster_summary.csv")
    print("="*80)


if __name__ == "__main__":
    main()
