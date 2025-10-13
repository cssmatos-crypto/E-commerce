import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Caminhos
# -----------------------------
base_path = r"C:\Users\claudia.santos-matos\OneDrive\Formacao\DM\TrabalhoFinal-1"
profile_csv = os.path.join(base_path, "data", "cluster_profile_final.csv")
checkpoint_folder = os.path.join(base_path, "checkpoints")

# -----------------------------
# Carregar perfil dos clusters
# -----------------------------
df_profile = pd.read_csv(profile_csv, index_col=0)

# Nomes intuitivos para clusters
cluster_names = {
    0: "Utilizadores Passivos",
    1: "Exploradores Leves",
    2: "Curiosos Intensivos",
    3: "Compradores Fieis",
    4: "Exploradores Intermediários"
}
df_profile["cluster_name"] = df_profile.index.map(cluster_names)

# -----------------------------
# Gráficos de distribuição
# -----------------------------
# 1. Distribuição de usuários por cluster
plt.figure(figsize=(8,5))
plt.bar(df_profile["cluster_name"], df_profile["count"], color='skyblue')
plt.title("Distribuição de usuários por cluster")
plt.ylabel("Nº de usuários")
plt.xticks(rotation=25)
plt.show()

# 2. Taxa média de compra por cluster (%)
plt.figure(figsize=(8,5))
plt.bar(df_profile["cluster_name"], df_profile["is_purchase"]*100, color='orange')
plt.title("Taxa média de compra por cluster (%)")
plt.ylabel("% de compras")
plt.xticks(rotation=25)
plt.show()

# -----------------------------
# Radar chart normalizado
# -----------------------------
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
    ax.plot(angles, values, linewidth=2, label=cluster_names.get(cluster_id, str(cluster_id)))
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title("Perfil dos Clusters (normalizado)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.show()

# -----------------------------
# Resultados do modelo SGD com SMOTE
# -----------------------------
print("📊 RESULTADOS NO CONJUNTO DE TESTE (SMOTE):")
accuracy = 0.9106
precision = 0.1218
recall = 0.8571
f1 = 0.2133
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")

# -----------------------------
# Importância relativa das features
# -----------------------------
sgd_model = joblib.load(os.path.join(checkpoint_folder,"sgd_model_smote.pkl"))
sgd_features = [
    'user_event_count','hour_bucket','events_per_user','cart_view_ratio',
    'hour_span','events_per_hour','avg_price_per_user','unique_brands',
    'unique_categories','cluster'
]
coefs = sgd_model.coef_[0]
importance = pd.Series(np.abs(coefs), index=sgd_features).sort_values(ascending=True)

plt.figure(figsize=(8,5))
importance.plot(kind="barh", color="teal")
plt.title("Importância relativa das features (SGDClassifier)")
plt.xlabel("|Coeficiente|")
plt.show()
