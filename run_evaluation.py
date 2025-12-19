"""
SAMI - Main Evaluation Script
Comprehensive evaluation of SAMI models on sponge dataset
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from vision_transformer import vit_small, vit_base
from utils import (
    load_dataset_images, 
    get_class_names, 
    extract_features,
    save_embeddings,
    load_embeddings
)
from utils_cbir import evaluate_cbir
from utils_eval import (
    train_knn_classifier,
    evaluate_knn,
    plot_confusion_matrix,
    plot_tsne,
    save_evaluation_report,
    compute_class_wise_metrics,
    compare_k_values
)


def parse_args():
    parser = argparse.ArgumentParser(description='SAMI Evaluation')
    parser.add_argument('--data_path', type=str, default='./imagefolder_cambrian_sponges',
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--model_arch', type=str, default='vit_small',
                       choices=['vit_small', 'vit_base'],
                       help='Model architecture')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for ViT')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--k_neighbors', type=int, default=7,
                       help='Number of neighbors for KNN')
    parser.add_argument('--save_embeddings_path', type=str, default=None,
                       help='Path to save extracted embeddings')
    parser.add_argument('--load_embeddings_path', type=str, default=None,
                       help='Path to load pre-computed embeddings')
    return parser.parse_args()


def load_model(args):
    """Load Vision Transformer model"""
    print(f"Loading {args.model_arch} model...")
    
    if args.model_arch == 'vit_small':
        model = vit_small(patch_size=args.patch_size)
    elif args.model_arch == 'vit_base':
        model = vit_base(patch_size=args.patch_size)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_arch}")
    
    if args.model_path:
        print(f"Loading weights from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("Warning: No model weights provided. Using random initialization.")
        print("To use pretrained weights, provide --model_path")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    
    return model, device


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SAMI - Sponge Automatic Model Identification")
    print("Evaluation Pipeline")
    print("="*80)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Get class names
    class_names = get_class_names(args.data_path)
    print(f"Found {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print()
    
    # Load or compute embeddings
    if args.load_embeddings_path:
        print(f"Loading pre-computed embeddings from {args.load_embeddings_path}")
        data = load_embeddings(args.load_embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        image_paths = data['image_paths']
    else:
        # Load dataset
        print("Loading dataset images...")
        images_tensor, image_paths, labels = load_dataset_images(
            args.data_path, 
            img_size=args.img_size
        )
        
        # Load model
        model, device = load_model(args)
        
        # Extract features
        print("Extracting features...")
        embeddings = extract_features(
            model, 
            images_tensor, 
            batch_size=args.batch_size,
            device=device
        )
        
        # Save embeddings if requested
        if args.save_embeddings_path:
            save_embeddings(
                embeddings, 
                labels, 
                image_paths,
                args.save_embeddings_path
            )
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of samples: {len(labels)}\n")
    
    # Evaluation 1: CBIR (Content-Based Image Retrieval)
    print("="*80)
    print("Evaluation 1: Content-Based Image Retrieval (CBIR)")
    print("="*80)
    cbir_results = evaluate_cbir(
        embeddings, 
        labels, 
        k_values=[1, 5, 10, 20],
        metric='cosine'
    )
    
    print("CBIR Results:")
    for metric, value in cbir_results.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # Evaluation 2: K-NN Classification
    print("="*80)
    print("Evaluation 2: K-Nearest Neighbors Classification")
    print("="*80)
    
    # Train KNN
    print(f"Training KNN classifier (k={args.k_neighbors})...")
    knn = train_knn_classifier(
        embeddings, 
        labels, 
        n_neighbors=args.k_neighbors,
        metric='cosine'
    )
    
    # Evaluate (using leave-one-out implicitly through KNN)
    print("Evaluating KNN...")
    knn_metrics = evaluate_knn(
        knn, 
        embeddings, 
        labels, 
        class_names=class_names
    )
    
    print("\nKNN Classification Results:")
    print(f"  Accuracy: {knn_metrics['accuracy']:.4f}")
    print(f"  Macro F1-Score: {knn_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score: {knn_metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision: {knn_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {knn_metrics['macro_recall']:.4f}")
    print()
    
    # Evaluation 3: Per-class metrics
    print("="*80)
    print("Evaluation 3: Per-Class Metrics")
    print("="*80)
    
    class_metrics = compute_class_wise_metrics(
        labels,
        knn_metrics['predictions'],
        class_names
    )
    print(class_metrics.to_string(index=False))
    print()
    
    # Save metrics to CSV
    class_metrics.to_csv(output_dir / 'class_metrics.csv', index=False)
    print(f"Per-class metrics saved to {output_dir / 'class_metrics.csv'}\n")
    
    # Evaluation 4: Compare different k values
    print("="*80)
    print("Evaluation 4: K-value Comparison")
    print("="*80)
    
    k_comparison = compare_k_values(
        embeddings, 
        labels, 
        k_values=[1, 3, 5, 7, 9, 11, 15],
        metric='cosine'
    )
    print(k_comparison.to_string(index=False))
    k_comparison.to_csv(output_dir / 'k_comparison.csv', index=False)
    print(f"\nK-value comparison saved to {output_dir / 'k_comparison.csv'}\n")
    
    # Visualization 1: Confusion Matrix
    print("="*80)
    print("Generating Visualizations")
    print("="*80)
    
    print("Creating confusion matrix...")
    plot_confusion_matrix(
        knn_metrics['confusion_matrix'],
        class_names,
        title=f'SAMI Confusion Matrix (k={args.k_neighbors})',
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    
    # Visualization 2: t-SNE
    print("Creating t-SNE visualization...")
    plot_tsne(
        embeddings,
        labels,
        class_names,
        title='SAMI t-SNE Embedding Visualization',
        save_path=str(output_dir / 't-sne_visualization.png'),
        perplexity=min(30, len(embeddings) // 5)  # Adjust perplexity based on dataset size
    )
    
    # Save comprehensive report
    print("\nSaving evaluation report...")
    all_metrics = {
        **knn_metrics,
        'cbir': cbir_results
    }
    save_evaluation_report(
        all_metrics,
        class_names,
        save_path=str(output_dir / 'evaluation_report.txt')
    )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - class_metrics.csv")
    print(f"  - k_comparison.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - t-sne_visualization.png")
    print(f"  - evaluation_report.txt")
    if args.save_embeddings_path:
        print(f"  - {args.save_embeddings_path}")
    print("="*80)


if __name__ == "__main__":
    main()
