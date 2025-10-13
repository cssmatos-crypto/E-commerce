
import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -----------------------------
# Caminhos
# -----------------------------
base_path = r"C:\Users\claudia.santos-matos\OneDrive\Formacao\DM\TrabalhoFinal-1"
clean_folder = os.path.join(base_path, "data", "clean")
checkpoint_folder = os.path.join(base_path, "checkpoints")
model_file = os.path.join(checkpoint_folder, "sgd_model_smote.pkl")
scaler_file = os.path.join(checkpoint_folder, "scaler_model_smote.pkl")

# -----------------------------
# Inicializar modelos e scalers
# -----------------------------
sgd_model = SGDClassifier(loss="log_loss", max_iter=5, random_state=42)
scaler_model = StandardScaler()

# -----------------------------
# Carregar Parquets
# -----------------------------
parquet_files = sorted(glob.glob(os.path.join(clean_folder,"part_*.parquet")))
print(f"🔹 Encontrados {len(parquet_files)} Parquets.")

# -----------------------------
# Features para treino
# -----------------------------
sgd_features = [
    'user_event_count','hour_bucket','events_per_user','cart_view_ratio',
    'hour_span','events_per_hour','avg_price_per_user','unique_brands',
    'unique_categories','cluster'
]

# -----------------------------
# Treino incremental com SMOTE
# -----------------------------
for idx, f in enumerate(parquet_files):
    df = pd.read_parquet(f)
    df = df.replace([np.inf,-np.inf],0).fillna(0)
    
    X = df[sgd_features].values
    y = df['is_purchase'].values
    
    # Escalar incremental
    scaler_model.partial_fit(X)
    X_scaled = scaler_model.transform(X)
    
    # Aplicar SMOTE apenas se houver ao menos 1 exemplo da classe minoritária
    if y.sum() > 0:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)
    else:
        X_res, y_res = X_scaled, y
    
    # Treino incremental
    sgd_model.partial_fit(X_res, y_res, classes=[0,1])
    
    if (idx+1) % 50 == 0:
        print(f"✅ {idx+1}/{len(parquet_files)} Parquets processados")

# -----------------------------
# Salvar modelo e scaler final
# -----------------------------
joblib.dump(sgd_model, model_file)
joblib.dump(scaler_model, scaler_file)
print(f"\n✅ Modelo SGD com SMOTE salvo em {model_file}")
print(f"✅ Scaler salvo em {scaler_file}")

# -----------------------------
# Avaliar no último Parquet
# -----------------------------
df_eval = pd.read_parquet(parquet_files[-1])
df_eval = df_eval.replace([np.inf,-np.inf],0).fillna(0)
X_eval = df_eval[sgd_features].values
y_eval = df_eval['is_purchase'].values
X_eval_scaled = scaler_model.transform(X_eval)
y_pred = sgd_model.predict(X_eval_scaled)

print("\n📊 RESULTADOS NO CONJUNTO DE TESTE (SMOTE):")
print(f"Accuracy : {accuracy_score(y_eval,y_pred):.4f}")
print(f"Precision: {precision_score(y_eval,y_pred,zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_eval,y_pred,zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_eval,y_pred,zero_division=0):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_eval,y_pred,digits=4, zero_division=0))

