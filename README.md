# SAMI - Sponge Automatic Model Identification

## About

**SAMI (Sponge Automatic Model Identification)** is an adaptation of the [SCAMPI benchmark](https://github.com/equinor/scampi-benchmark) for automatic identification and classification of **Cambrian sponge macrofossils**.

While SCAMPI focuses on microfossils (small organic remains visible under microscope), SAMI extends this methodology to larger macrofossil specimens of sponges from the Cambrian period, enabling rapid species classification and separation in seconds.

## Project Structure

```
sami-project/
├── imagefolder_cambrian_sponges/  # Your sponge species images
│   ├── species_1/
│   ├── species_2/
│   └── ...
├── example_usage.py               # Basic usage example
├── run_evaluation.py              # Evaluation pipeline
├── utils.py                       # General utilities
├── utils_cbir.py                  # Content-Based Image Retrieval utilities
├── utils_eval.py                  # Evaluation utilities
├── vision_transformer.py          # Vision Transformer model
├── vision_transformer_mae.py      # Masked Autoencoder ViT
└── requirements.txt               # Dependencies
```

## Key Differences from SCAMPI

1. **Target**: Cambrian sponge macrofossils (larger specimens) vs. microfossils
2. **Scale**: Macrofossil images are typically larger and may require different preprocessing
3. **Species**: Custom sponge species dataset vs. dinoflagellate cysts and palynomorphs
4. **Application**: Rapid species identification for paleontological research

## Features

- Vision Transformer (ViT) based feature extraction
- Self-supervised learning with DINO
- Content-Based Image Retrieval (CBIR)
- K-Nearest Neighbors classification
- Precision, Recall, and F1-score metrics
- t-SNE visualization of embeddings

## Installation

```bash
# Create conda environment
conda create -n sami python=3.11
conda activate sami

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Organize your sponge images in the following structure:

```
imagefolder_cambrian_sponges/
├── Archaeocyatha_species1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Porifera_species2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 2. Basic Usage

```python
import torch
from vision_transformer import vit_small
from utils import load_image, preprocess_image

# Load pretrained model (you can start with SCAMPI weights or train your own)
model = vit_small(patch_size=16)
model.load_state_dict(torch.load('path/to/weights.pth'))
model.eval()

# Process an image
image = load_image('path/to/sponge_image.jpg')
image_tensor = preprocess_image(image)

# Extract features
with torch.no_grad():
    features = model(image_tensor.unsqueeze(0))

print(f"Feature vector shape: {features.shape}")
```

### 3. Run Evaluation

```bash
python run_evaluation.py --data_path ./imagefolder_cambrian_sponges --model_path path/to/weights.pth
```

## Training Your Own Model

To train a model on your sponge dataset:

1. Prepare a large dataset of unlabeled sponge images
2. Use self-supervised learning (DINO) to pretrain
3. Fine-tune on your labeled dataset
4. Evaluate on test set

See `training/` directory for training scripts (coming soon).

## Model Performance

Performance metrics will be added as you develop the model:

| Model | Accuracy | F1-Score | P@5 |
|-------|----------|----------|-----|
| SAMI ViT-S/16 | TBD | TBD | TBD |
| SAMI ViT-B/16 | TBD | TBD | TBD |

## Benchmarking

The evaluation pipeline includes:

- **CBIR (Content-Based Image Retrieval)**: Find similar sponge specimens
- **K-NN Classification**: Classify specimens based on nearest neighbors
- **Precision@k**: Measure retrieval accuracy
- **Class-wise metrics**: Per-species performance analysis

## Roadmap

- [ ] Initial data collection and organization
- [ ] Adapt preprocessing for macrofossil images
- [ ] Train initial ViT-S model
- [ ] Benchmark against SCAMPI pretrained weights
- [ ] Develop species-specific augmentations
- [ ] Create web interface for identification
- [ ] Publish results and trained models

## Contributing

This is an early-stage research project. Contributions, suggestions, and collaborations are welcome!

## Citation

If you use SAMI in your research, please cite both this work and the original SCAMPI paper:

```bibtex
@article{martinsen2024fossil,
  title={The 3-billion fossil question: How to automate classification of microfossils},
  author={Martinsen, Iver and Wade, David and Ricaud, Benjamin and Godtliebsen, Fred},
  journal={Artificial Intelligence in Geosciences},
  year={2024},
  doi={10.1016/j.aiig.2024.100080}
}
```

## License

Apache 2.0 License - Same as SCAMPI

## Acknowledgements

- Based on [SCAMPI benchmark](https://github.com/equinor/scampi-benchmark) by UiT and Equinor
- DINO self-supervised learning from Meta AI Research
- Vision Transformer architecture from Google Research

## Contact

For questions and collaborations, please open an issue on GitHub.
