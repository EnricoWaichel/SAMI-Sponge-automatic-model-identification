# 🔬 Guia: Análise Multi-Escala de Patches (Janelas de Convolução)

## 📋 O que Este Script Faz

1. **Extrai janelas** de múltiplos tamanhos de cada imagem
2. **Nomeia** cada janela com prefixo: `classe/nome_imagem/window_tamanho`
3. **Extrai features** de cada janela usando Vision Transformer
4. **Clusteriza** janelas similares
5. **Visualiza** resultados com t-SNE ou PCA

---

## 🎯 Caso de Uso do Seu Chefe

### Estrutura de Entrada:

```
data/
├── leptomitid/
│   ├── fossil_001.jpg
│   ├── fossil_002.jpg
│   └── ...
└── choialike/
    ├── specimen_A.jpg
    ├── specimen_B.jpg
    └── ...
```

### O que Acontece:

**1. Extração de Janelas:**
```
fossil_001.jpg (1024x768)
  ↓
Janelas de 64x64 → 12 patches extraídos
Janelas de 128x128 → 6 patches extraídos  
Janelas de 256x256 → 2 patches extraídos
  ↓
Total: 20 patches desta imagem
```

**2. Nomenclatura:**
```
leptomitid/fossil_001/window_64/0_0_000001.jpg
leptomitid/fossil_001/window_64/32_0_000002.jpg
leptomitid/fossil_001/window_128/0_0_000003.jpg
...
```

**3. Clusterização:**
```
Patch 1 (textura porosa) → Cluster 0
Patch 2 (textura porosa) → Cluster 0
Patch 3 (borda lisa) → Cluster 1
Patch 4 (estrutura ramificada) → Cluster 2
...
```

**4. Visualização:**
- **t-SNE**: Reduz features para 2D
- **PCA**: Principais componentes

---

## 🚀 Como Usar

### Comando Básico

```bash
cd C:\Users\enrico\Documents\projeto\SAMI

conda activate sami

python multi_scale_patch_clustering.py \
    --data_path ./sponge_data \
    --window_sizes 64 128 256 \
    --stride 32 \
    --n_clusters 10 \
    --viz_method both \
    --save_patches \
    --output_dir ./patch_results
```

### Parâmetros Explicados:

| Parâmetro | O que faz | Exemplo |
|-----------|-----------|---------|
| `--data_path` | Pasta com leptomitid/ e choialike/ | `./sponge_data` |
| `--window_sizes` | Tamanhos das janelas | `64 128 256` |
| `--stride` | Passo da janela deslizante | `32` (50% overlap) |
| `--n_clusters` | Quantos grupos criar | `10` |
| `--viz_method` | tsne, pca ou both | `both` |
| `--save_patches` | Salvar janelas extraídas | (flag) |
| `--output_dir` | Onde salvar resultados | `./patch_results` |

---

## 🔍 Entendendo Janelas de Convolução

### Tamanho da Janela (Window Size)

**Janelas pequenas (64x64):**
- ✅ Detectam detalhes finos (poros individuais, espículas)
- ✅ Mais janelas por imagem
- ❌ Perdem contexto global

**Janelas médias (128x128):**
- ✅ Balanceiam detalhe e contexto
- ✅ Capturam texturas e padrões médios

**Janelas grandes (256x256):**
- ✅ Capturam estrutura geral
- ✅ Mostram morfologia ampla
- ❌ Menos janelas por imagem

### Stride (Passo)

```
Stride = 32 (50% overlap):
┌──────┬──────┬──────┐
│  W1  │  W2  │  W3  │
│      ├──────┼──────┤
│      │  W4  │  W5  │
└──────┴──────┴──────┘
Muitas janelas, muita informação

Stride = 64 (sem overlap):
┌──────┬──────┬──────┐
│  W1  │  W2  │  W3  │
├──────┼──────┼──────┤
│  W4  │  W5  │  W6  │
└──────┴──────┴──────┘
Menos janelas, mais rápido
```

**Recomendação**: Stride = metade do window_size

---

## 📊 Resultados Gerados

### 1. Visualização t-SNE

**Arquivo**: `tsne_visualization.png`

```
┌─────────────────────┬─────────────────────┐
│  Por Cluster        │  Por Classe         │
│                     │                     │
│  🔴🔴🔴             │  ⬛⬛⬛ leptomitid   │
│   🔴🔴               │   ⬛⬛⬛            │
│                     │                     │
│      🔵🔵🔵         │      ⬜⬜⬜ choialike│
│     🔵🔵🔵🔵        │     ⬜⬜⬜⬜         │
│                     │                     │
│  🟢🟢               │                     │
│ 🟢🟢🟢              │                     │
└─────────────────────┴─────────────────────┘
```

**Como interpretar**:
- **Esquerda**: Cores = clusters automáticos
- **Direita**: Cores = classes originais (leptomitid vs choialike)

**Insights**:
- Se clusters separam classes → Modelo distingue bem
- Se clusters misturam classes → Classes são similares
- Clusters isolados → Características únicas

### 2. Visualização PCA

**Arquivo**: `pca_visualization.png`

Similar ao t-SNE, mas usando Análise de Componentes Principais:
- **PC1**: Primeira componente (maior variância)
- **PC2**: Segunda componente

**Vantagem**: Mais rápido que t-SNE, mais interpretável

### 3. Resumo CSV

**Arquivo**: `cluster_summary.csv`

```csv
patch_id,cluster,class,image_name,window_size,x,y,prefix
0,2,leptomitid,fossil_001,64,0,0,leptomitid/fossil_001/window_64
1,2,leptomitid,fossil_001,64,32,0,leptomitid/fossil_001/window_64
2,5,leptomitid,fossil_001,128,0,0,leptomitid/fossil_001/window_128
...
```

**Análises possíveis**:
```python
import pandas as pd

df = pd.read_csv('cluster_summary.csv')

# Quais janelas de cada classe foram para cada cluster?
pd.crosstab(df['class'], df['cluster'])

# Quais tamanhos de janela dominam cada cluster?
pd.crosstab(df['window_size'], df['cluster'])
```

### 4. Patches Salvos (se --save_patches)

**Estrutura**:
```
extracted_patches/
├── leptomitid/
│   ├── fossil_001/
│   │   ├── window_64/
│   │   │   ├── 0_0_000001.jpg
│   │   │   ├── 32_0_000002.jpg
│   │   │   └── ...
│   │   ├── window_128/
│   │   │   └── ...
│   │   └── window_256/
│   │       └── ...
│   └── fossil_002/
│       └── ...
└── choialike/
    └── ...
```

---

## 💡 Exemplos de Uso

### Exemplo 1: Análise Rápida (sem salvar patches)

```bash
python multi_scale_patch_clustering.py \
    --data_path ./sponge_data \
    --window_sizes 128 \
    --stride 64 \
    --n_clusters 5 \
    --viz_method tsne
```

**Resultado**: 
- Rápido (~5 min)
- Só 1 tamanho de janela
- Visualização t-SNE

---

### Exemplo 2: Análise Completa (multi-escala)

```bash
python multi_scale_patch_clustering.py \
    --data_path ./sponge_data \
    --window_sizes 64 128 256 512 \
    --stride 32 \
    --n_clusters 15 \
    --viz_method both \
    --save_patches \
    --model_path ./scampi_weights.pth
```

**Resultado**:
- 4 escalas diferentes
- Patches salvos em disco
- t-SNE + PCA
- Usa modelo pré-treinado

---

### Exemplo 3: Foco em Detalhes Finos

```bash
python multi_scale_patch_clustering.py \
    --data_path ./sponge_data \
    --window_sizes 32 64 \
    --stride 16 \
    --n_clusters 20 \
    --viz_method tsne
```

**Resultado**:
- Janelas pequenas (detalhes microscópicos)
- Muitas janelas por imagem
- Mais clusters para capturar variação

---

## 🔬 Interpretação Científica

### Pergunta: "Por que clusterizar janelas ao invés de imagens inteiras?"

**Resposta**:

**Imagem Inteira**:
```
┌────────────────────┐
│                    │
│  🧽 Esponja        │
│                    │
│  Mista:            │
│  - Topo: rugoso    │
│  - Centro: poroso  │
│  - Base: liso      │
│                    │
└────────────────────┘
Feature = "média geral"
```

**Janelas Separadas**:
```
┌──────┐ ┌──────┐ ┌──────┐
│Janela│ │Janela│ │Janela│
│  1   │ │  2   │ │  3   │
│Rugoso│ │Poroso│ │Liso  │
└──────┘ └──────┘ └──────┘
   ↓        ↓        ↓
Cluster  Cluster  Cluster
   A        B        C
```

**Vantagens**:
1. **Especialização**: Cada janela captura UMA característica
2. **Localização**: Saber ONDE na esponja está cada textura
3. **Múltiplas escalas**: Detalhes finos + estrutura geral
4. **Mais dados**: 1 imagem → 20+ janelas = 20x mais treino

---

## 📈 Análise Pós-Clusterização

### 1. Verificar Distribuição

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cluster_summary.csv')

# Distribuição de clusters por classe
ct = pd.crosstab(df['class'], df['cluster'], normalize='index')
ct.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Distribuição de Clusters por Classe')
plt.ylabel('Proporção')
plt.show()
```

**Interpretação**:
- Barras similares → Classes parecidas
- Barras diferentes → Classes distintas

### 2. Identificar Clusters Diagnósticos

```python
# Quais clusters são exclusivos de uma classe?
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    class_dist = cluster_data['class'].value_counts(normalize=True)
    
    if class_dist.max() > 0.9:  # 90%+ de uma classe
        print(f"Cluster {cluster}: {class_dist.idxmax()} ({class_dist.max():.1%})")
```

**Resultado**:
```
Cluster 3: leptomitid (95%)  ← Exclusivo!
Cluster 7: choialike (92%)   ← Exclusivo!
```

---

## 🎯 Perguntas que Pode Responder

1. **Texturas distintivas?**
   - Clusters exclusivos de uma classe

2. **Variabilidade dentro de uma classe?**
   - Quantos clusters contêm apenas leptomitid?

3. **Sobreposição entre classes?**
   - Clusters mistos (50%/50%)

4. **Escala mais informativa?**
   - Comparar silhouette score por window_size

5. **Regiões diagnósticas?**
   - Mapear clusters de volta para coordenadas (x, y)

---

## ⚠️ Troubleshooting

**Problema**: "Muitas janelas, memória insuficiente"

**Solução**:
```bash
# Aumentar stride (menos overlap)
--stride 128

# Ou usar menos tamanhos
--window_sizes 128 256
```

---

**Problema**: "Todos os patches no mesmo cluster"

**Solução**:
```bash
# Aumentar número de clusters
--n_clusters 20

# Ou usar modelo pré-treinado
--model_path ./scampi_weights.pth
```

---

**Problema**: "t-SNE demora muito"

**Solução**:
```bash
# Usar PCA (mais rápido)
--viz_method pca
```

---

## 📊 Workflow Recomendado para Seu Projeto

```bash
# 1. Teste rápido
python multi_scale_patch_clustering.py \
    --data_path ./sponge_data \
    --window_sizes 128 \
    --n_clusters 5 \
    --viz_method tsne

# 2. Ver resultados iniciais
# - Abrir tsne_visualization.png
# - Ver se clusters fazem sentido

# 3. Análise completa
python multi_scale_patch_clustering.py \
    --data_path ./sponge_data \
    --window_sizes 64 128 256 \
    --stride 32 \
    --n_clusters 10 \
    --viz_method both \
    --save_patches \
    --model_path ./scampi_weights.pth \
    --output_dir ./final_patch_analysis

# 4. Analisar CSV
python
>>> import pandas as pd
>>> df = pd.read_csv('final_patch_analysis/cluster_summary.csv')
>>> pd.crosstab(df['class'], df['cluster'])

# 5. Apresentar resultados para o chefe!
```

---

## 🎓 Para Apresentação

**Slides Sugeridos**:

1. **Motivação**: Por que janelas multi-escala?
2. **Metodologia**: Vision Transformer + K-Means
3. **Resultados**: t-SNE/PCA plots
4. **Insights**: Clusters diagnósticos, texturas únicas
5. **Conclusão**: leptomitid vs choialike são distinguíveis?

**Figuras para Incluir**:
- tsne_visualization.png (ambos painéis)
- pca_visualization.png
- Exemplos de patches de cada cluster

---

**Boa sorte com a análise! 🔬🧽**
