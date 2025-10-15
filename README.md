# 🛍️ Ecommerce Behavior – Previsão de Compras e Análise de Clusters

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

📊 Projeto baseado no dataset **[Ecommerce Behavior Data from Multi Category Store (Kaggle)](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)**,  
usando o ficheiro **`2019-Nov.csv`**.

---

## 🎯 Objetivo

Prever a **probabilidade de compra de utilizadores** e segmentá-los em **clusters comportamentais**, de forma a apoiar decisões de **marketing direcionado** e **recomendação de produtos**.

---

## ⚙️ Pipeline de Processamento

### 🔹 1. Pré-processamento Incremental
- Leitura do ficheiro `2019-Nov.csv` em chunks de 50.000 linhas.  
- Limpeza de dados, conversão de datas e filtragem de eventos de novembro.  
- Criação de features comportamentais e temporais.  
- Salvamento dos dados tratados em formato **Parquet** (`data/clean/part_*.parquet`).

### 🔹 2. Clusterização
- Utilização de **MiniBatchKMeans (k=5)** para criar clusters de utilizadores com base em comportamento.  
- Redução de dimensionalidade via **Incremental PCA (3 componentes)** para visualização.  
- Criação do ficheiro `data/cluster_profile_final.csv` com o perfil médio de cada cluster.

### 🔹 3. Modelação Supervisionada
- **SGDClassifier incremental (regressão logística)** para previsão de `is_purchase`.  
- **StandardScaler incremental** para normalização contínua.  
- **SMOTE** para balanceamento da classe minoritária (eventos de compra).  
- Avaliação em dados não vistos (último chunk).

### 🔹 4. Análise Global
- Avaliação do modelo em todo o dataset (13,5 milhões de eventos).  
- Geração de métricas, gráficos e relatórios.

---

## 🧮 Features Criadas

| Feature | Tipo | Descrição |
|----------|------|-----------|
| `user_event_count` | Contagem | Número total de eventos do utilizador |
| `hour_bucket` | Temporal | Agrupa horas em períodos (0–4, 4–8, etc.) |
| `cart_view_ratio` | Rácio | Percentagem de eventos “cart” (intenção de compra) |
| `hour_span` | Diferencial | Duração da sessão em horas |
| `events_per_hour` | Taxa | Atividade média por hora |
| `avg_price_per_user` | Média | Valor médio dos produtos vistos |
| `unique_brands` / `unique_categories` | Diversidade | Variedade de marcas e categorias vistas |
| `purchase_ratio` | Rácio | Percentagem de eventos de compra |
| `cluster` | Categórica | Cluster comportamental do utilizador |
| `is_purchase` | Target | Variável a prever (0 = não compra, 1 = compra) |

---

## 🧩 Modelos Treinados

### 🔸 **MiniBatchKMeans (k=5)**
Identifica perfis de utilizadores:
| Cluster | Nome | Interpretação |
|----------|------|---------------|
| 0 | Utilizadores Passivos | Pouca atividade, raramente compram |
| 1 | Exploradores Leves | Interagem pouco, conversão baixa |
| 2 | Curiosos Intensivos | Alta interação, baixa conversão |
| 3 | Compradores Fieis | Alta interação e alta taxa de compra |
| 4 | Exploradores Intermédios | Atividade média, compra ocasional |

### 🔸 **SGDClassifier incremental**
- Modelo linear (regressão logística) com treino incremental.  
- Balanceamento de classes via **undersampling** e **SMOTE**.  
- Features principais: `cluster`, `cart_view_ratio`, `avg_price_per_user`, `events_per_user`.

---

## 📊 Resultados do Modelo

### 🔹 Modelo com SMOTE (teste incremental)
| Métrica | Valor |
|----------|-------|
| Accuracy | 0.9106 |
| Precision | 0.1218 |
| Recall | 0.8571 |
| F1-score | 0.2133 |

### 🔹 Avaliação Global (dataset completo – 13,5M eventos)
| Métrica | Valor | Interpretação |
|----------|--------|---------------|
| Accuracy | 0.8940 | 89,4% de previsões corretas |
| Precision (classe 1) | 0.1037 | Previsões de compra certas (~10%) |
| Recall (classe 1) | 0.7806 | 78% dos compradores reais identificados |
| F1-score | 0.1831 | Equilíbrio moderado entre precisão e recall |
| ROC-AUC | 0.9036 | Excelente capacidade discriminativa |

---

## 📈 Visualizações Sugeridas

1. **Distribuição da variável target (`is_purchase`)**  
2. **Matriz de Confusão** – desempenho entre classes  
3. **Curva ROC e Precision–Recall**  
4. **Taxa de compra média por cluster**  
5. **Radar chart** – perfil normalizado de cada cluster  
6. **Importância das features** – gráfico de barras horizontal

---

## 🧠 Insights Principais

- `cluster` e `cart_view_ratio` são as variáveis **mais determinantes**.  
- O **Cluster 3 (“Compradores Fieis”)** tem a maior taxa de conversão.  
- **SMOTE** aumentou o F1-score da classe minoritária sem reduzir recall.  
- O pipeline incremental permite escalar para milhões de registos sem sobrecarga de memória.

---

## 📘 Perguntas Orientadoras

| Pergunta | Resposta |
|-----------|-----------|
| **É possível prever compras em novembro de 2019?** | ✅ Sim — Recall 78% e ROC-AUC 0.90 confirmam boa capacidade preditiva. |
| **Que perfis de utilizadores existem?** | 5 clusters distintos; o Cluster 3 é o mais valioso para campanhas e fidelização. |

---

## 🧭 Ordem de Execução

1️⃣ `src/full_pipeline.py` → Pré-processamento e feature engineering  
2️⃣ `src/update_parquets_with_clusters.py` → Geração e aplicação dos clusters  
3️⃣ `src/analyze_clusters_and_model.py` → Treino e avaliação do modelo supervisionado  
4️⃣ `Relatorio.ipynb` → Análise, visualizações e conclusões finais  


