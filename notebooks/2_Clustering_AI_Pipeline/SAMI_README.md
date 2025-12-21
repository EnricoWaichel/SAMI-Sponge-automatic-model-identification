# SAMI - Sponge Automatic Model Identification

Pipeline para identificação automática de espécimes de esponjas fósseis usando Vision Transformers.

---

## Requisitos

- Anaconda ou Miniconda instalado
- Python 3.10+
- GPU NVIDIA com CUDA 12.1 (opcional, mas recomendado)

---

## Instalação

### 1. Criar ambiente Conda

Abra o Anaconda Prompt e execute:

```bash
conda create -n sami python=3.10 -y
```

### 2. Ativar ambiente

```bash
conda activate sami
```

### 3. Instalar PyTorch

**Com GPU NVIDIA (CUDA 12.1):**

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Sem GPU (CPU only):**

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### 4. Instalar dependências científicas

```bash
conda install numpy pandas matplotlib seaborn scikit-learn tqdm pillow -c conda-forge -y
```

### 5. Instalar dependências adicionais via pip

```bash
pip install umap-learn opencv-python
```

### 6. Instalar Jupyter

```bash
pip install jupyter notebook
```

---

## Estrutura de Diretórios

O dataset deve seguir o formato ImageFolder:

```
imagefolder_cambrian_sponges/
├── Classe_A/
│   ├── imagem_001.jpg
│   ├── imagem_002.jpg
│   └── ...
├── Classe_B/
│   ├── imagem_001.jpg
│   └── ...
└── Classe_C/
    └── ...
```

Cada subpasta representa uma classe/espécie.

---

## Uso

### Iniciar Jupyter Notebook

```bash
conda activate sami
jupyter notebook
```

Abra o arquivo `SAMI_Complete.ipynb` e execute as células.

### Pipeline de Avaliação Completa

```python
results = run_full_evaluation(
    data_path='./imagefolder_cambrian_sponges',
    model_path=None,              # Caminho para pesos pré-treinados (opcional)
    model_arch='vit_small',       # 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'
    output_dir='./results',
    k_neighbors=7,
    batch_size=32
)
```

### Análise de Clustering

```python
# K-Means, DBSCAN, Hierárquico
results, df = kmeans_clustering(embeddings, [3, 5, 7, 10], coords_2d, output_dir)
dbscan_clustering(embeddings, coords_2d, output_dir)
hierarchical_clustering(embeddings, coords_2d, output_dir)
```

### Multi-Scale Patch Clustering

```python
all_patches = extract_all_patches(
    image_paths, class_names,
    window_sizes=[64, 128, 256],
    stride=32,
    max_patches_per_image=50,
    min_content_ratio=0.7
)
```

---

## Arquiteturas Disponíveis

| Modelo | Embed Dim | Depth | Heads | Parâmetros |
|--------|-----------|-------|-------|------------|
| vit_tiny | 192 | 12 | 3 | ~5.7M |
| vit_small | 384 | 12 | 6 | ~22M |
| vit_base | 768 | 12 | 12 | ~86M |
| vit_large | 1024 | 24 | 16 | ~307M |

---

## Saídas Geradas

Após execução do pipeline:

| Arquivo | Descrição |
|---------|-----------|
| `class_metrics.csv` | Precision, Recall, F1 por classe |
| `k_comparison.csv` | Comparação de diferentes valores de k |
| `confusion_matrix.png` | Matriz de confusão normalizada |
| `t-sne_visualization.png` | Projeção t-SNE dos embeddings |
| `evaluation_report.txt` | Relatório completo de métricas |
| `embeddings.npz` | Embeddings salvos para reutilização |

---

## Métricas de Avaliação

### CBIR (Content-Based Image Retrieval)
- **Recall@k**: Proporção de queries com pelo menos um resultado correto nos top-k
- **Precision@k**: Proporção média de resultados corretos nos top-k
- **MAP@k**: Mean Average Precision

### Classificação KNN
- **Accuracy**: Acurácia global
- **Macro F1**: F1-Score médio entre classes (não ponderado)
- **Weighted F1**: F1-Score ponderado pelo suporte de cada classe

### Clustering
- **Silhouette Score**: [-1, 1] — valores > 0.5 indicam boa separação
- **Calinski-Harabasz**: Quanto maior, melhor a definição dos clusters

---

## Troubleshooting

### CUDA não disponível

Verifique a instalação:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Memória insuficiente

Reduza o `batch_size`:

```python
embeddings = extract_features(model, images, batch_size=16, device=DEVICE)
```

### UMAP não encontrado

```bash
pip install umap-learn
```

### OpenCV erro de importação

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

---

## Referências

- Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- DINO: [Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- UMAP: [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426)

---

## Licença

MIT License
