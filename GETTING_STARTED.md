# Getting Started with SAMI

Este guia vai te ajudar a começar a usar o SAMI para identificação automática de esponjas do Cambriano.

## 1. Instalação

### Pré-requisitos
- Python 3.8 ou superior
- CUDA (opcional, mas recomendado para GPU)

### Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## 2. Preparar Seus Dados

### Estrutura de Diretórios

Organize suas imagens de esponjas da seguinte forma:

```
imagefolder_cambrian_sponges/
├── Archaeocyatha_species1/
│   ├── specimen_001.jpg
│   ├── specimen_002.jpg
│   ├── specimen_003.jpg
│   └── ...
├── Porifera_species2/
│   ├── specimen_001.jpg
│   ├── specimen_002.jpg
│   └── ...
├── Hexactinellida_species3/
│   └── ...
└── ...
```

**Importante:**
- Cada pasta representa uma espécie diferente
- O nome da pasta será usado como o nome da classe
- Formatos suportados: `.jpg`, `.jpeg`, `.png`
- Recomendado: pelo menos 20-30 imagens por espécie para resultados confiáveis

## 3. Primeiros Passos

### Opção A: Começar do Zero (Sem Modelo Pré-treinado)

Se você ainda não tem um modelo treinado:

```bash
# 1. Testar o exemplo básico
python example_usage.py

# 2. Extrair features da sua base de dados
# (Edite example_usage.py e descomente extract_database_features_example)
```

**Nota:** Sem um modelo pré-treinado, os resultados serão aleatórios. Você precisará:
1. Treinar um modelo do zero, OU
2. Fazer fine-tuning de um modelo SCAMPI pré-treinado

### Opção B: Usar Modelo SCAMPI Como Ponto de Partida

Baixe os pesos do SCAMPI e use como base:

```bash
# Baixar pesos do SCAMPI ViT-S/16
wget https://huggingface.co/IverMartinsen/scampi-dino-vits16/resolve/main/vit_small_backbone.pth

# Rodar avaliação
python run_evaluation.py \
    --data_path ./imagefolder_cambrian_sponges \
    --model_path ./vit_small_backbone.pth \
    --model_arch vit_small \
    --output_dir ./results
```

## 4. Entendendo os Resultados

Após rodar `run_evaluation.py`, você encontrará em `./results/`:

### `evaluation_report.txt`
Resumo geral com métricas principais:
- **Accuracy**: Acurácia geral
- **F1-Score**: Média harmônica de precisão e recall
- **Precision/Recall**: Por espécie

### `confusion_matrix.png`
Matriz de confusão mostrando:
- Diagonal: Classificações corretas
- Fora da diagonal: Confusões entre espécies

### `t-sne_visualization.png`
Visualização 2D dos embeddings:
- Pontos próximos = espécimes visualmente similares
- Clusters bem separados = espécies bem distinguíveis

### `class_metrics.csv`
Métricas detalhadas por espécie

### `k_comparison.csv`
Performance com diferentes valores de K para KNN

## 5. Interpretando os Resultados

### Bons Resultados
- **Accuracy > 0.80**: Modelo está funcionando bem
- **F1-Score > 0.75**: Boa capacidade de classificação
- **Clusters separados no t-SNE**: Espécies são distinguíveis

### Resultados Ruins
- **Accuracy < 0.60**: Modelo precisa de mais dados ou treinamento
- **Confusão entre espécies similares**: Normal, pode melhorar com mais dados
- **Clusters sobrepostos no t-SNE**: Espécies são muito similares visualmente

## 6. Próximos Passos

### Se os resultados estão bons:
1. Extraia features para toda sua coleção
2. Use para busca por similaridade (CBIR)
3. Documente seu pipeline

### Se os resultados estão ruins:
1. **Coletar mais dados**: Pelo menos 50+ imagens por espécie
2. **Data Augmentation**: Adicionar rotações, flips, zoom
3. **Fine-tuning**: Treinar o modelo especificamente para suas esponjas
4. **Revisão de labels**: Verificar se as classificações estão corretas

## 7. Troubleshooting

### Erro: "CUDA out of memory"
```bash
# Reduzir batch size
python run_evaluation.py --batch_size 16
```

### Erro: "No images found"
- Verifique a estrutura de pastas
- Confirme que as imagens têm extensões corretas (.jpg, .jpeg, .png)

### Resultados aleatórios
- Você está usando modelo sem pesos pré-treinados
- Baixe pesos do SCAMPI ou treine seu próprio modelo

## 8. Exemplo Completo

```bash
# 1. Criar estrutura de dados
mkdir -p imagefolder_cambrian_sponges/Archaeocyatha_sp1
mkdir -p imagefolder_cambrian_sponges/Porifera_sp2

# 2. Copiar suas imagens para as pastas apropriadas
# (faça isso manualmente ou com script)

# 3. Baixar modelo base
wget https://huggingface.co/IverMartinsen/scampi-dino-vits16/resolve/main/vit_small_backbone.pth

# 4. Rodar avaliação
python run_evaluation.py \
    --data_path ./imagefolder_cambrian_sponges \
    --model_path ./vit_small_backbone.pth \
    --model_arch vit_small \
    --img_size 224 \
    --batch_size 32 \
    --k_neighbors 7 \
    --output_dir ./results \
    --save_embeddings_path ./sponge_embeddings.npz

# 5. Ver resultados
cat results/evaluation_report.txt
```

## 9. Ajuda e Suporte

- **Issues**: Abra uma issue no GitHub
- **Dúvidas**: Consulte o README.md principal
- **Paper SCAMPI**: https://doi.org/10.1016/j.aiig.2024.100080

## 10. Checklist Inicial

- [ ] Python e dependências instaladas
- [ ] Imagens organizadas em pastas por espécie
- [ ] Pelo menos 20 imagens por espécie
- [ ] Modelo pré-treinado baixado (opcional)
- [ ] `example_usage.py` executado com sucesso
- [ ] `run_evaluation.py` executado com sucesso
- [ ] Resultados revisados e entendidos

Pronto! Você está preparado para usar o SAMI! 🧽🔬
