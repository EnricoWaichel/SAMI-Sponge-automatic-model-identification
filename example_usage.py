"""
SAMI - Example Usage
Basic example of using SAMI for sponge fossil identification
"""

import torch
import numpy as np
from vision_transformer import vit_small
from utils import load_image, preprocess_image, get_class_names
from utils_cbir import build_index, retrieve_similar
import matplotlib.pyplot as plt
from PIL import Image


def main():
    """Basic example of SAMI usage"""
    
    # 1. Load pretrained model
    print("Loading model...")
    model = vit_small(patch_size=16)
    
    # Option A: Load SCAMPI pretrained weights as starting point
    # model.load_state_dict(torch.load('path/to/scampi_weights.pth'))
    
    # Option B: Load your own trained weights
    # model.load_state_dict(torch.load('path/to/sami_weights.pth'))
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # 2. Load and preprocess a single image
    print("\nProcessing query image...")
    query_image_path = "path/to/your/sponge_image.jpg"
    
    # Load image
    img = load_image(query_image_path)
    if img is None:
        print(f"Error: Could not load image from {query_image_path}")
        print("Please update the path to a valid image file.")
        return
    
    # Preprocess
    img_tensor = preprocess_image(img, img_size=224)
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # 3. Extract features
    print("Extracting features...")
    with torch.no_grad():
        features = model(img_batch)
    
    features_np = features.cpu().numpy()
    print(f"Feature vector shape: {features_np.shape}")
    print(f"Feature vector (first 10 values): {features_np[0, :10]}")
    
    # 4. Example: Find similar images (if you have a database)
    print("\n" + "="*80)
    print("To find similar images, you need:")
    print("1. A database of labeled sponge images")
    print("2. Pre-computed embeddings for all database images")
    print("="*80)
    
    # Example code (uncomment when you have data):
    """
    # Load database embeddings
    from utils import load_embeddings
    db_data = load_embeddings('path/to/database_embeddings.npz')
    db_embeddings = db_data['embeddings']
    db_labels = db_data['labels']
    db_paths = db_data['image_paths']
    
    # Build search index
    nn_model = build_index(db_embeddings, metric='cosine', n_neighbors=10)
    
    # Find similar images
    distances, indices = retrieve_similar(features_np, nn_model, k=5)
    
    print(f"\nTop 5 most similar specimens:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        print(f"{i+1}. {db_paths[idx]} (distance: {dist:.4f}, class: {db_labels[idx]})")
    """
    
    # 5. Information about next steps
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Organize your sponge images into folders by species")
    print("2. Use utils.load_dataset_images() to load all images")
    print("3. Extract features for all images")
    print("4. Run evaluation with run_evaluation.py")
    print("\nExample directory structure:")
    print("  imagefolder_cambrian_sponges/")
    print("    ├── Archaeocyatha_sp1/")
    print("    │   ├── specimen_001.jpg")
    print("    │   ├── specimen_002.jpg")
    print("    │   └── ...")
    print("    ├── Porifera_sp2/")
    print("    │   └── ...")
    print("    └── ...")
    print("="*80)


def extract_database_features_example():
    """
    Example of how to extract features for your entire database
    Run this once to create embeddings file
    """
    from utils import load_dataset_images, save_embeddings, extract_features
    
    print("Loading dataset...")
    data_path = "./imagefolder_cambrian_sponges"
    
    # Load all images
    images_tensor, image_paths, labels = load_dataset_images(data_path, img_size=224)
    
    # Load model
    print("Loading model...")
    model = vit_small(patch_size=16)
    # model.load_state_dict(torch.load('weights.pth'))
    model.eval()
    
    # Extract features
    print("Extracting features...")
    embeddings = extract_features(
        model, 
        images_tensor, 
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(
        embeddings, 
        labels, 
        image_paths,
        output_path="sponge_embeddings.npz"
    )
    
    print("Done! Embeddings saved to sponge_embeddings.npz")


if __name__ == "__main__":
    print("="*80)
    print("SAMI - Sponge Automatic Model Identification")
    print("Example Usage")
    print("="*80)
    
    main()
    
    print("\n\nTo extract features for your entire database, run:")
    print("  python example_usage.py --extract-database")
