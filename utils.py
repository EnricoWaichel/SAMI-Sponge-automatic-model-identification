"""
General utility functions for SAMI
Image loading, preprocessing, and data handling
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import torchvision.transforms as transforms


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_transform(img_size: int = 224, is_training: bool = False):
    """
    Get image transformation pipeline
    
    Args:
        img_size: Target image size
        is_training: Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # Slightly larger
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def preprocess_image(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single image for model input
    
    Args:
        img: PIL Image
        img_size: Target size
    
    Returns:
        Preprocessed tensor
    """
    transform = get_transform(img_size, is_training=False)
    return transform(img)


def load_dataset_images(data_path: str, 
                       img_size: int = 224) -> Tuple[torch.Tensor, List[str], List[int]]:
    """
    Load all images from a dataset directory organized by class folders
    
    Args:
        data_path: Path to dataset root directory
        img_size: Target image size
    
    Returns:
        Tuple of (images_tensor, image_paths, labels)
    """
    data_path = Path(data_path)
    
    # Get class folders
    class_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {cls_folder.name: idx for idx, cls_folder in enumerate(class_folders)}
    
    images = []
    labels = []
    image_paths = []
    
    transform = get_transform(img_size, is_training=False)
    
    print(f"Loading images from {len(class_folders)} classes...")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]
        
        # Get all image files
        image_files = list(class_folder.glob('*.jpg')) + \
                     list(class_folder.glob('*.jpeg')) + \
                     list(class_folder.glob('*.png'))
        
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img = load_image(str(img_file))
            if img is not None:
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(class_idx)
                image_paths.append(str(img_file))
    
    images_tensor = torch.stack(images)
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Tensor shape: {images_tensor.shape}")
    
    return images_tensor, image_paths, labels


def get_class_names(data_path: str) -> List[str]:
    """Get list of class names from dataset directory"""
    data_path = Path(data_path)
    class_folders = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    return class_folders


def save_embeddings(embeddings: np.ndarray, 
                   labels: List[int],
                   image_paths: List[str],
                   output_path: str):
    """
    Save embeddings and metadata to file
    
    Args:
        embeddings: Numpy array of embeddings
        labels: List of class labels
        image_paths: List of image file paths
        output_path: Where to save
    """
    data = {
        'embeddings': embeddings,
        'labels': np.array(labels),
        'image_paths': image_paths
    }
    
    np.savez(output_path, **data)
    print(f"Embeddings saved to {output_path}")


def load_embeddings(embeddings_path: str) -> Dict:
    """Load saved embeddings"""
    data = np.load(embeddings_path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'image_paths': data['image_paths']
    }


def create_splits(n_samples: int, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits
    
    Args:
        n_samples: Total number of samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def extract_features(model: torch.nn.Module,
                    images: torch.Tensor,
                    batch_size: int = 32,
                    device: str = 'cuda') -> np.ndarray:
    """
    Extract features from images using a model
    
    Args:
        model: Feature extractor model
        images: Batch of images
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        Numpy array of features
    """
    model.eval()
    model = model.to(device)
    
    all_features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        features = model(batch)
        all_features.append(features.cpu().numpy())
    
    return np.vstack(all_features)
