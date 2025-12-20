# 🧬 Guia de Clusterização do SAMI

Este guia explica como usar a análise de clusterização para descobrir grupos naturais nas suas imagens de esponjas **sem precisar saber as espécies previamenente**.

---

## 📋 Pré-requisitos

1. **Imagens organizadas em UMA pasta** (não precisa separar por espécie ainda)
2. **Ambiente conda ativado**: `conda activate sami`
3. **Modelo pré-treinado** (opcional, mas recomendado)

---

## 🚀 Uso Básico

### Passo 1: Organizar Imagens (Estrutura Simples)

Coloque TODAS as suas imagens em uma pasta:

```
imagefolder_unlabeled_sponges/
├── sponge_001.jpg
├── sponge_002.jpg
├── sponge_003.jpg
├── ...
└── sponge_n.jpg
```

**OU** se quiser manter subpastas (serão ignoradas):

```
imagefolder_unlabeled_sponges/
├── batch_1/
│   ├── img_001.jpg
│   └── ...
├── batch_2/
│   └── ...
└── ...
```

### Passo 2: Rodar Análise Básica

```bash
cd C:\Users\enrico\Documents\projeto\SAMI

conda activate sami

python clustering_analysis.py \
    --data_path ./imagefolder_unlabeled_sponges \
    --model_path ./scampi_weights.pth \
    --output_dir ./clustering_results \
    --save_cluster_images
```

---

## 🎛️ Opções Avançadas

### Testar Diferentes Números de Clusters

```bash
python clustering_analysis.py \
    --data_path ./imagefolder_unlabeled_sponges \
    --n_clusters_range 3 5 7 10 15 \
    --output_dir ./clustering_results
```

### Usar Todos os Métodos de Clusterização

```bash
python clustering_analysis.py \
    --data_path ./imagefolder_unlabeled_sponges \
    --method all \
    --output_dir ./clustering_results
```

### Salvar Mais Exemplos por Cluster

```bash
python clustering_analysis.py \
    --data_path ./imagefolder_unlabeled_sponges \
    --save_cluster_images \
    --max_images_per_cluster 50
```

---

## 📊 Entendendo os Resultados

Após rodar, você terá em `clustering_results/`:

### 1. **Visualizações t-SNE**

**Arquivos**: `kmeans_k3.png`, `kmeans_k5.png`, etc.

```
🔴 Cluster 0: 45 imagens
🔵 Cluster 1: 38 imagens
🟢 Cluster 2: 52 imagens
```

**Como interpretar:**
- ✅ **Clusters bem separados** = Grupos naturais distintos
- ⚠️ **Clusters sobrepostos** = Grupos com características similares
- ❌ **Tudo misturado** = Pode precisar de mais dados ou modelo melhor

### 2. **Tabela de Comparação**

**Arquivo**: `kmeans_comparison.csv`

```csv
n_clusters,silhouette,calinski_harabasz,inertia
3,0.652,245.32,1234.5
5,0.721,312.45,987.3  ← MELHOR (silhouette mais alto)
7,0.598,298.12,856.7
10,0.512,267.89,734.2
```

**Regra geral:**
- **Silhouette > 0.7**: Excelente separação
- **Silhouette 0.5-0.7**: Boa separação
- **Silhouette < 0.5**: Separação fraca

### 3. **Dendrograma Hierárquico**

**Arquivo**: `hierarchical_dendrogram.png`

```
         ┌─────┐
    ┌────┤     │
────┤    └─────┘
    │    ┌─────┐
    └────┤     │
         └─────┘
```

**Como interpretar:**
- **Linhas verticais longas** = Grupos muito diferentes
- **Linhas verticais curtas** = Grupos similares
- **Onde cortar** = Define número de clusters

### 4. **Exemplos de Imagens**

**Pasta**: `clustering_results/cluster_images/`

```
cluster_images/
├── cluster_0/        ← Revisar: São parecidas?
│   ├── 001_sponge_042.jpg
│   ├── 002_sponge_089.jpg
│   └── ...
├── cluster_1/
│   └── ...
└── cluster_2/
    └── ...
```

**O que fazer:**
1. Abra cada pasta
2. Veja se as imagens são visualmente similares
3. Se SIM → Esse cluster faz sentido!
4. Se NÃO → Tente outro número de clusters

---

## 🎯 Workflow Recomendado

### Cenário 1: Primeira Vez Usando

```bash
# 1. Análise exploratória
python clustering_analysis.py \
    --data_path ./minhas_imagens \
    --method all \
    --n_clusters_range 3 5 7 10 \
    --save_cluster_images

# 2. Revisar resultados
# - Abrir clustering_results/
# - Ver t-SNE plots
# - Revisar cluster_images/

# 3. Escolher melhor k (ex: k=5)
# 4. Organizar dataset baseado no clustering
```

### Cenário 2: Você Tem Modelo Pré-treinado

```bash
# Usar pesos do SCAMPI como base
python clustering_analysis.py \
    --data_path ./minhas_imagens \
    --model_path ./scampi_weights.pth \
    --method kmeans \
    --n_clusters_range 5 7 10 \
    --save_cluster_images \
    --max_images_per_cluster 30
```

### Cenário 3: Dataset Pequeno (< 100 imagens)

```bash
# Usar menos clusters e DBSCAN
python clustering_analysis.py \
    --data_path ./minhas_imagens \
    --method dbscan \
    --perplexity 10  # Reduzir perplexity para datasets pequenos
```

---

## 💡 Dicas Práticas

### ✅ Boas Práticas

1. **Comece com K-Means**: Mais fácil de interpretar
2. **Use silhouette score**: Guia objetivo para escolher k
3. **Revise visualmente**: Métricas ajudam, mas seus olhos são importantes
4. **Teste ranges**: 3-10 clusters geralmente é suficiente

### ⚠️ Problemas Comuns

**Problema 1: Todos os clusters têm tamanhos muito diferentes**

```
Cluster 0: 250 imagens
Cluster 1: 5 imagens   ← Muito pequeno!
Cluster 2: 12 imagens
```

**Solução**: Aumentar número de clusters OU usar DBSCAN

---

**Problema 2: Silhouette score muito baixo (<0.3)**

**Possíveis causas:**
- Imagens muito similares (sem grupos naturais)
- Modelo não treinado (usando pesos aleatórios)
- Dataset muito pequeno

**Soluções:**
- Baixar pesos SCAMPI pré-treinados
- Coletar mais imagens
- Tentar clustering hierárquico

---

**Problema 3: DBSCAN encontra só noise (cluster -1)**

```
Cluster -1 (noise): 200 imagens
Cluster 0: 3 imagens
```

**Solução**: Ajustar parâmetro `eps` (testar 0.3, 0.5, 0.7, 1.0)

---

## 🔄 Depois da Clusterização

### Organizar Dataset Baseado nos Resultados

Se você decidiu que **k=5 é o melhor**:

```bash
# 1. Copiar estrutura gerada
cp -r clustering_results/cluster_images/* imagefolder_cambrian_sponges/

# 2. Renomear clusters para morfotipos
# cluster_0 → morphotype_cylindrical
# cluster_1 → morphotype_branched
# cluster_2 → morphotype_globular
# ...
```

### Usar para Treinar Modelo Supervisionado

```bash
# Agora você tem labels!
python run_evaluation.py \
    --data_path ./imagefolder_cambrian_sponges \
    --output_dir ./results_supervised
```

---

## 🔬 Exemplo Completo

```bash
# Cenário: 500 imagens de esponjas não rotuladas

# 1. Análise inicial
python clustering_analysis.py \
    --data_path ./unlabeled_sponges \
    --model_path ./scampi_weights.pth \
    --method all \
    --n_clusters_range 3 5 7 10 \
    --save_cluster_images \
    --max_images_per_cluster 30 \
    --output_dir ./clustering_v1

# 2. Resultado: k=7 tem melhor silhouette (0.68)

# 3. Revisar cluster_images/
# - cluster_0: esponjas cilíndricas (78 imagens) ✅
# - cluster_1: esponjas ramificadas (92 imagens) ✅
# - cluster_2: esponjas globulares (65 imagens) ✅
# - cluster_3: misto de formas (45 imagens) ⚠️
# - cluster_4: pequenas irregulares (55 imagens) ✅
# - cluster_5: grandes cônicas (87 imagens) ✅
# - cluster_6: lâminas achatadas (78 imagens) ✅

# 4. Organizar dataset
mkdir -p imagefolder_cambrian_sponges
mv clustering_v1/cluster_images/cluster_0 imagefolder_cambrian_sponges/cylindrical
mv clustering_v1/cluster_images/cluster_1 imagefolder_cambrian_sponges/branched
# ... etc

# 5. Treinar modelo supervisionado
python run_evaluation.py \
    --data_path ./imagefolder_cambrian_sponges \
    --model_path ./scampi_weights.pth \
    --output_dir ./results_final
```

---

## 📚 Recursos Adicionais

### Parâmetros Importantes

```python
--n_clusters_range 3 5 7 10    # Quais k testar
--method kmeans                # kmeans, dbscan, hierarchical, all
--perplexity 30                # t-SNE (10-50, menor para datasets pequenos)
--save_cluster_images          # Salvar exemplos
--max_images_per_cluster 20    # Quantos exemplos salvar
```

### Algoritmos Disponíveis

| Método | Quando Usar | Vantagem | Desvantagem |
|--------|-------------|----------|-------------|
| **K-Means** | Você sabe aproximadamente quantos grupos | Rápido, fácil de interpretar | Precisa definir k |
| **DBSCAN** | Grupos com densidades diferentes | Encontra k automaticamente | Sensível a parâmetros |
| **Hierarchical** | Quer ver relações entre grupos | Dendrogram mostra hierarquia | Mais lento |

---

## ❓ FAQ

**P: Preciso de modelo pré-treinado?**
R: Não é obrigatório, mas MUITO recomendado. Pesos aleatórios dão resultados ruins.

**P: Quantas imagens preciso?**
R: Mínimo 50-100. Ideal 200+. Menos que 30 é muito pouco.

**P: O clustering decidiu as espécies?**
R: NÃO! Clustering agrupa por similaridade visual. Você ainda precisa nomear os grupos (com ajuda de especialista, se possível).

**P: Posso usar clustering E labels manuais juntos?**
R: SIM! Use clustering para ajudar a organizar, depois refine manualmente.

---

## 🎓 Próximos Passos

1. ✅ Rodar clustering nas suas imagens
2. ✅ Revisar resultados e escolher melhor k
3. ✅ Organizar dataset baseado nos clusters
4. ✅ Nomear clusters com morfotipos/espécies
5. ✅ Treinar modelo supervisionado
6. ✅ Publicar/usar para identificação automática

**Boa sorte com sua análise! 🧽🔬**
