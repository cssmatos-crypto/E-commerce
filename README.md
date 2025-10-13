https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store


CSV bruto (chunks) 
       │
       ▼
Leitura por chunks → Limpeza & Feature Engineering
       │
       ▼
Guardar Parquet limpo
       │
       ├─> Treino incremental SGDClassifier
       └─> Treino incremental MiniBatchKMeans
       │
       ▼
Checkpoint (modelos + estado)
       │
       ▼
Avaliação final + Atribuição de clusters
       │
       ▼
Resultados & Parquets com clusters

************************************************************************************************

Pipeline de Análise de Compras e Clusters (Novembro 2019)

Objetivo: Prever compras de usuários e segmentá-los em clusters.

Dados: CSV de eventos de novembro 2019, processado em chunks e transformado em Parquets limpos.

Features criadas:

Atividade: user_event_count, events_per_user, events_per_hour

Temporal: hour_bucket, hour_span

Compromisso: cart_view_ratio, purchase_ratio

Diversidade: avg_price_per_user, unique_brands, unique_categories

Cluster do usuário (cluster)

Modelos:

SGDClassifier incremental (com SMOTE para balancear classe de compra)

MiniBatchKMeans (5 clusters)

IncrementalPCA (3 componentes, visualização 3D)

Resultados (último Parquet de teste):

Accuracy: 0.9106

Precision: 0.1218

Recall: 0.8571

F1-score: 0.2133

Clusters principais:

Cluster 0: Usuários Passivos

Cluster 1: Exploradores Leves

Cluster 2: Curiosos Intensivos

Cluster 3: Compradores Fieis

Cluster 4: Exploradores Intermediários

****************** Cluster 3 tem alta taxa de compra; 
****************** Cluster 2 é muito ativo mas pouco converte; 
****************** Clusters 0 e 4 são maioria e compram pouco.