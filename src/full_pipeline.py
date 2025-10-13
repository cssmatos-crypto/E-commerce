import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# -----------------------------
# Caminhos base do projeto
# -----------------------------
base_path = r"C:\Users\claudia.santos-matos\OneDrive\Formacao\DM\TrabalhoFinal-1"
raw_file = os.path.join(base_path, "data", "raw", "2019-Nov.csv")
clean_folder = os.path.join(base_path, "data", "clean")
checkpoint_folder = os.path.join(base_path, "checkpoints")
clusters_folder = os.path.join(base_path, "data", "clusters_features")

os.makedirs(clean_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(clusters_folder, exist_ok=True)

# -----------------------------
# Configurações
# -----------------------------
chunksize = 50_000
kmeans_clusters = 5
modo_teste = False         # True = só processa alguns chunks para teste
num_chunks_teste = 10

# -----------------------------
# Inicializar modelos e scalers
# -----------------------------
sgd_model = SGDClassifier(loss="log_loss", max_iter=5)
kmeans_model = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=5000)
scaler_features = StandardScaler()
scaler_model = StandardScaler()
ipca = IncrementalPCA(n_components=3)

state_file = os.path.join(checkpoint_folder, "state.pkl")
state = {"last_chunk": -1, "total_rows": 0}

if os.path.exists(state_file):
    state = joblib.load(state_file)
    print(f"Retomando do chunk {state['last_chunk']+1}")

# -----------------------------
# Processamento por chunks
# -----------------------------
for i, chunk in enumerate(pd.read_csv(raw_file, chunksize=chunksize)):
    if i <= state["last_chunk"]:
        continue
    if modo_teste and i >= num_chunks_teste:
        break

    print(f"\n🔹 Processando chunk {i}...")

    # Converter datas e filtrar novembro
    chunk['event_time'] = pd.to_datetime(chunk['event_time'], errors='coerce')
    chunk = chunk.dropna(subset=['event_time'])
    chunk = chunk[chunk['event_time'].dt.month == 11]

    # Limpeza básica
    chunk['category_code'] = chunk['category_code'].astype(str).fillna("unknown")
    chunk['brand'] = chunk['brand'].astype(str).fillna("unknown")

    # -----------------------------
    # Feature engineering 
    # -----------------------------
    chunk['user_event_count'] = chunk.groupby('user_id')['event_type'].transform('count')
    chunk['hour_bucket'] = chunk['event_time'].dt.hour // 4
    chunk['is_purchase'] = (chunk['event_type'] == 'purchase').astype(int)
    chunk['events_per_user'] = chunk.groupby('user_id')['event_type'].transform('count')
    chunk['cart_view_ratio'] = chunk.groupby('user_id')['event_type'].transform(lambda x: (x=='cart').mean())
    chunk['hour_span'] = chunk.groupby('user_id')['event_time'].transform(lambda x: (x.max() - x.min()).total_seconds() / 3600)
    chunk['events_per_hour'] = chunk['events_per_user'] / (chunk['hour_span']+1)
    chunk['avg_price_per_user'] = chunk.groupby('user_id')['price'].transform('mean')
    chunk['unique_brands'] = chunk.groupby('user_id')['brand'].transform('nunique')
    chunk['unique_categories'] = chunk.groupby('user_id')['category_code'].transform('nunique')
    chunk['purchase_ratio'] = chunk.groupby('user_id')['event_type'].transform(lambda x: (x=='purchase').mean())

    # -----------------------------
    # Guardar Parquet limpo
    # -----------------------------
    parquet_file = os.path.join(clean_folder, f"part_{i:04d}.parquet")
    chunk.to_parquet(parquet_file, engine="pyarrow")
    print(f"✅ Chunk salvo: {parquet_file}")

    # -----------------------------
    # Preparar features para clustering e modelo
    # -----------------------------
    features_cluster = [
        'user_event_count','events_per_user','cart_view_ratio','hour_span',
        'events_per_hour','avg_price_per_user','unique_brands','unique_categories','purchase_ratio'
    ]
    X_cluster = chunk[features_cluster].replace([np.inf,-np.inf],0).fillna(0).values

    # Normalizar e treinar clustering
    scaler_features.partial_fit(X_cluster)
    X_scaled = scaler_features.transform(X_cluster)
    kmeans_model.partial_fit(X_scaled)
    ipca.partial_fit(X_scaled)

    # Prever clusters
    chunk['cluster'] = kmeans_model.predict(X_scaled)
    X_pca = ipca.transform(X_scaled)
    chunk['pca1'], chunk['pca2'], chunk['pca3'] = X_pca[:,0], X_pca[:,1], X_pca[:,2]

    # -----------------------------
    # Treino incremental do SGDClassifier com balanceamento
    # -----------------------------
    sgd_features = [
        'user_event_count','hour_bucket','events_per_user','cart_view_ratio',
        'hour_span','events_per_hour','avg_price_per_user','unique_brands',
        'unique_categories','cluster'
    ]
    X_sgd = chunk[sgd_features].values
    y_sgd = chunk['is_purchase'].values

    # Balanceamento simples: undersampling da classe majoritária
    mask_0 = y_sgd==0
    mask_1 = y_sgd==1
    if mask_1.sum()>0:
        X_bal = np.vstack([X_sgd[mask_1], X_sgd[mask_0][:mask_1.sum()]])
        y_bal = np.hstack([y_sgd[mask_1], y_sgd[mask_0][:mask_1.sum()]])
    else:
        X_bal = X_sgd
        y_bal = y_sgd

    scaler_model.partial_fit(X_bal)
    X_bal_scaled = scaler_model.transform(X_bal)
    sgd_model.partial_fit(X_bal_scaled, y_bal, classes=[0,1])

    # -----------------------------
    # Guardar checkpoints
    # -----------------------------
    joblib.dump(sgd_model, os.path.join(checkpoint_folder,"sgd_model.pkl"))
    joblib.dump(kmeans_model, os.path.join(checkpoint_folder,"kmeans_model.pkl"))
    joblib.dump(scaler_features, os.path.join(checkpoint_folder,"scaler_features.pkl"))
    joblib.dump(scaler_model, os.path.join(checkpoint_folder,"scaler_model.pkl"))

    # Atualizar estado
    state["last_chunk"] = i
    state["total_rows"] += len(chunk)
    joblib.dump(state, state_file)

print("\n✅ Pipeline incremental completo! Modelos, clusters e Parquets salvos.")
