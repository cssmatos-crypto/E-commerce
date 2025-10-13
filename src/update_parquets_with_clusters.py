import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -----------------------------
# Caminhos
# -----------------------------
base_path = r"C:\Users\claudia.santos-matos\OneDrive\Formacao\DM\TrabalhoFinal-1"
clean_folder = os.path.join(base_path, "data", "clean")
checkpoint_folder = os.path.join(base_path, "checkpoints")
output_csv = os.path.join(base_path, "data", "cluster_profile_final.csv")

# -----------------------------
# Carregar Parquets
# -----------------------------
parquet_files = sorted(glob.glob(os.path.join(clean_folder, "part_*.parquet")))
print(f"🔹 Encontrados {len(parquet_files)} Parquets.")

# -----------------------------
# Carregar modelos
# -----------------------------
sgd_model = joblib.load(os.path.join(checkpoint_folder,"sgd_model.pkl"))
scaler_model = joblib.load(os.path.join(checkpoint_folder,"scaler_model.pkl"))
kmeans_model = joblib.load(os.path.join(checkpoint_folder,"kmeans_model.pkl"))
scaler_features = joblib.load(os.path.join(checkpoint_folder,"scaler_features.pkl"))

ipca_path = os.path.join(checkpoint_folder,"ipca.pkl")
if os.path.exists(ipca_path):
    ipca = joblib.load(ipca_path)
    print("✅ IPCA carregado do checkpoint.")
else:
    ipca = IncrementalPCA(n_components=3)
    print("⚡ IPCA não encontrado, será treinado agora.")

# -----------------------------
# Features para clustering
# -----------------------------
features_cluster = [
    'user_event_count','events_per_user','cart_view_ratio','hour_span',
    'events_per_hour','avg_price_per_user','unique_brands','unique_categories','purchase_ratio'
]

# -----------------------------
# Inicializar agregação incremental para perfil
# -----------------------------
agg_cols = features_cluster + ["is_purchase"]
cluster_totals = {}
cluster_counts = {}

# -----------------------------
# Atualizar Parquets + calcular perfil incremental
# -----------------------------
for idx, f in enumerate(parquet_files):
    df = pd.read_parquet(f)
    df = df.replace([np.inf,-np.inf],0).fillna(0)

    # Corrigir coluna user_session
    if 'user_session' in df.columns:
        df['user_session'] = df['user_session'].fillna('unknown').astype(str)

    # Escalar e aplicar clusters + PCA
    X_cluster = df[features_cluster].values
    X_scaled = scaler_features.transform(X_cluster)

    if not hasattr(ipca,"components_") or ipca.components_.size==0:
        ipca.partial_fit(X_scaled)

    df['cluster'] = kmeans_model.predict(X_scaled)
    X_pca = ipca.transform(X_scaled)
    df['pca1'], df['pca2'], df['pca3'] = X_pca[:,0], X_pca[:,1], X_pca[:,2]

    # Salvar Parquet atualizado
    df.to_parquet(f, index=False)

    # Agregar para perfil
    grouped = df.groupby("cluster")[agg_cols].sum()
    counts = df.groupby("cluster").size()
    for cluster in grouped.index:
        if cluster not in cluster_totals:
            cluster_totals[cluster] = grouped.loc[cluster].copy()
            cluster_counts[cluster] = counts[cluster]
        else:
            cluster_totals[cluster] += grouped.loc[cluster]
            cluster_counts[cluster] += counts[cluster]

    if (idx+1) % 50 == 0:
        print(f"✅ {idx+1}/{len(parquet_files)} Parquets processados")

# -----------------------------
# Calcular perfil final
profile_list = []
for cluster in cluster_totals.keys():
    total = cluster_totals[cluster]
    n = cluster_counts[cluster]
    avg = total / n
    avg["count"] = n
    avg["perc_total"] = (n / sum(cluster_counts.values()) * 100)
    profile_list.append((cluster, avg))

df_profile = pd.DataFrame([x[1] for x in profile_list], index=[x[0] for x in profile_list])
df_profile = df_profile.sort_index()
df_profile.to_csv(output_csv, index=True)
print(f"\n✅ Perfil dos clusters salvo CSV: {output_csv}")
print(df_profile)

# -----------------------------
# Avaliação do modelo usando clusters
# -----------------------------
df_eval = pd.read_parquet(parquet_files[-1])
df_eval = df_eval.replace([np.inf,-np.inf],0).fillna(0)

sgd_features = [
    'user_event_count','hour_bucket','events_per_user','cart_view_ratio',
    'hour_span','events_per_hour','avg_price_per_user','unique_brands',
    'unique_categories','cluster'
]

X_eval = df_eval[sgd_features].values
y_eval = df_eval['is_purchase'].values
X_eval_scaled = scaler_model.transform(X_eval)
y_pred = sgd_model.predict(X_eval_scaled)

print("\n📊 RESULTADOS NO CONJUNTO DE TESTE:")
print(f"Accuracy : {accuracy_score(y_eval,y_pred):.4f}")
print(f"Precision: {precision_score(y_eval,y_pred,zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_eval,y_pred,zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_eval,y_pred,zero_division=0):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_eval,y_pred,digits=4, zero_division=0))

# -----------------------------
# Gráficos automáticos
# -----------------------------
# Distribuição de usuários por cluster
plt.figure(figsize=(8,5))
plt.bar(df_profile.index, df_profile['count'], color='skyblue')
plt.title("Distribuição de usuários por cluster")
plt.xlabel("Cluster")
plt.ylabel("Nº de usuários/linhas")
plt.show()

# Taxa média de compra por cluster
plt.figure(figsize=(8,5))
plt.bar(df_profile.index, df_profile['is_purchase']*100, color='orange')
plt.title("Taxa média de compra por cluster (%)")
plt.xlabel("Cluster")
plt.ylabel("% de compras")
plt.show()

# Radar chart
import numpy as np
cluster_names = {
    0: "Usuários Passivos",
    1: "Exploradores Leves",
    2: "Curiosos Intensivos",
    3: "Compradores Fieis",
    4: "Exploradores Intermediários"
}
categories = ['user_event_count','cart_view_ratio','avg_price_per_user','unique_brands','unique_categories','purchase_ratio','is_purchase']
N = len(categories)
df_radar = df_profile[categories].copy()
df_radar = df_radar / df_radar.max()
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
for cluster_id, row in df_radar.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=cluster_names.get(cluster_id,str(cluster_id)))
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title("Perfil dos Clusters (normalizado)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.show()
