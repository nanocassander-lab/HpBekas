# retrain_model.py
# Jalankan script ini sekali, lalu upload hasil .pkl yang baru ke Streamlit Cloud

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("hasil_clustering_hp.csv")

features = ["ram", "internal_memory", "battery", "rear_camera_mp", "screen_size"]
target = "Cluster"

df_clean = df.dropna(subset=features + [target])

X = df_clean[features]
y = df_clean[target]

# ==========================
# SCALER
# ==========================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# TRAIN RANDOM FOREST
# (pakai parameter terbaik dari Optuna kamu sebelumnya)
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Akurasi model: {score:.4f}")

# ==========================
# SAVE MODEL BARU
# ==========================
with open("rf_optuna.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model dan scaler berhasil disimpan ulang!")
print("Silakan upload rf_optuna.pkl dan scaler.pkl yang baru ke Streamlit Cloud.")
