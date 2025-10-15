# 🛍️ Previsão de Compras e Análise de Clusters em Dados de E-commerce

**Dataset:** [E-commerce Behavior Data from Multi-Category Store – Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)  
**Ficheiro usado:** `2019-Nov.csv`

---

---

## 🧭 Ordem de Execução

O projeto está organizado em módulos independentes dentro da pasta `src/`, permitindo execução faseada do pipeline.

### 🔹 1. Pré-processamento e Geração de Parquets
> **Script:** `src/full_pipeline.py`

- Lê o ficheiro original `raw/2019-Nov.csv` em chunks (50.000 linhas).  
- Limpa e transforma os dados.  
- Cria features comportamentais e salva os resultados em `data/clean/part_*.parquet`.

```bash
python src/full_pipeline.py



## 🎯 Objetivo

Prever a probabilidade de compra de utilizadores com base no seu comportamento de navegação e segmentá-los em **clusters** para análise de padrões e apoio à tomada de decisão estratégica.

---

## ⚙️ Pipeline do Projeto

| Componente | Função |
|-------------|--------|
| **SGDClassifier (incremental)** | Previsão da variável `is_purchase` com treino em mini-batches (streaming) |
| **MiniBatchKMeans (k=5)** | Segmentação de utilizadores em 5 clusters comportamentais |
| **IncrementalPCA (3 componentes)** | Redução de dimensionalidade para visualização 3D dos clusters |
| **SMOTE** | Oversampling da classe minoritária para mitigar desbalanceamento e melhorar o F1-score |

---

## 📊 Resultados Principais

| Métrica | Valor |
|---------|-------|
| ✅ **Accuracy** | 91,1% |
| 🎯 **Precision (classe 1 – compra)** | 12,2% |
| 📈 **Recall (classe 1 – compra)** | 85,7% |
| ⚖️ **F1-score (classe 1)** | 21,3% |
| 🧮 **ROC-AUC** | 0.90 |

### 🧠 Interpretação
- O modelo identifica **muito bem quem não compra** (alta precisão na classe 0).  
- Capta a **maioria dos compradores reais** (recall elevado), embora com alguns falsos positivos.  
- O **F1-score** reflete o equilíbrio entre capturar compradores e evitar previsões erradas.  
- O **AUC = 0.90** indica excelente separação entre compradores e não compradores.

---

## 🔑 Features Mais Relevantes

| Feature | Descrição |
|----------|------------|
| **`cluster`** | Representa o perfil comportamental do utilizador — a variável mais influente. |
| **`cart_view_ratio`** | Percentagem de eventos de visualização de carrinho → indica **intenção direta de compra**. |
| **`avg_price_per_user`** | Valor médio dos produtos visualizados → proxy de **poder de compra**. |
| **`events_per_user`** | Mede **nível de engagement** e interação geral. |

💡 **Insight:**  
O **Cluster 3 – “Compradores Fiéis”** destacou-se como o segmento **mais valioso**, com a maior taxa média de conversão.  
Usuários neste grupo exibem **alto engagement, alto `cart_view_ratio`** e **valores médios de produtos superiores**.

---

## 📉 Visualizações Recomendadas

| Gráfico | Objetivo |
|----------|----------|
| 📊 **Importância das Features (barras horizontais)** | Mostrar as variáveis mais influentes no modelo |
| 🧩 **Taxa de Compra por Cluster** | Identificar grupos de maior conversão |
| 🌀 **Projeção PCA 3D** | Visualizar separação entre clusters e padrões de comportamento |
| 🔁 **Curva Precision–Recall e ROC-AUC** | Avaliar equilíbrio e poder discriminativo do modelo |

---

## 🏁 Conclusões

- Os **clusters permitem segmentar utilizadores** com base em comportamento, distinguindo compradores fiéis de exploradores ocasionais.  
- O **SMOTE** melhora significativamente o desempenho para a classe minoritária, mantendo **recall elevado**.  
- As variáveis **`cart_view_ratio`** e **`cluster`** são determinantes para prever a probabilidade de compra.  
- O **pipeline incremental** possibilita o processamento de milhões de registos sem sobrecarga de memória.  
- O sistema é **escalável, interpretável e aplicável** a contextos de **marketing preditivo, recomendação de produtos e retenção de clientes**.

---

## 🚀 Tecnologias e Bibliotecas

- Python 3.10  
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  
- `SMOTE` via `imblearn`  
- Processamento incremental (`partial
