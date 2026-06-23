import streamlit as st
import pandas as pd
import pickle

# ==========================
# LOAD MODEL & SCALER
# ==========================
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Clustering HP Bekas",
    page_icon="📱",
    layout="centered"
)

# ==========================
# HEADER
# ==========================
st.title("📱 Clustering HP Bekas")

st.write(
    """
    Sistem ini digunakan untuk mengelompokkan HP bekas berdasarkan
    spesifikasi perangkat menggunakan algoritma **K-Means Clustering**.
    """
)

st.divider()

# ==========================
# INPUT USER
# ==========================
st.subheader("Masukkan Spesifikasi HP")

screen_size = st.number_input(
    "Screen Size (inch)",
    min_value=4.0,
    max_value=8.0,
    value=6.5,
    step=0.1
)

rear_camera = st.number_input(
    "Rear Camera (MP)",
    min_value=1.0,
    max_value=200.0,
    value=50.0,
    step=1.0
)

internal_memory = st.number_input(
    "Internal Memory (GB)",
    min_value=16,
    max_value=1024,
    value=128,
    step=16
)

ram = st.number_input(
    "RAM (GB)",
    min_value=1,
    max_value=32,
    value=8,
    step=1
)

battery = st.number_input(
    "Battery (mAh)",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=100
)

# ==========================
# PREDIKSI
# ==========================
if st.button("🔍 Prediksi Cluster"):

    data = pd.DataFrame(
        [[
            ram,
            internal_memory,
            battery,
            rear_camera,
            screen_size
        ]],
        columns=[
            "ram",
            "internal_memory",
            "battery",
            "rear_camera_mp",
            "screen_size"
        ]
    )

    try:

        # Scaling
        data_scaled = scaler.transform(data)

        # Prediksi
        cluster = int(kmeans.predict(data_scaled)[0])

        st.divider()

        # ==========================
        # HASIL
        # ==========================
        cluster_names = {
            0: "Entry Level",
            1: "Mid Range",
            2: "Entry Level Battery Besar"
        }

        st.success(
            f"Hasil Prediksi: {cluster_names.get(cluster, f'Cluster {cluster}')}"
        )

        # ==========================
        # RINGKASAN SPESIFIKASI
        # ==========================
        st.subheader("📋 Ringkasan Spesifikasi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("RAM", f"{ram} GB")
            st.metric("Storage", f"{internal_memory} GB")
            st.metric("Battery", f"{battery} mAh")

        with col2:
            st.metric("Rear Camera", f"{rear_camera} MP")
            st.metric("Screen Size", f"{screen_size} inch")

        st.divider()

        # ==========================
        # INTERPRETASI CLUSTER
        # ==========================
        st.subheader("📊 Interpretasi Cluster")

        if cluster == 0:

            st.info(
                """
### 📱 Cluster 0 — Entry Level

**Karakteristik:**
- Cocok untuk penggunaan ringan
- Harga relatif murah
- RAM dan storage standar
- Cocok untuk chat, browsing, dan media sosial

**Rata-rata Cluster:**
- RAM : 3.55 GB
- Storage : 45.13 GB
- Battery : 1946.14 mAh
- Kamera : 5.69 MP
                """
            )

        elif cluster == 1:

            st.success(
                """
### 🚀 Cluster 1 — Mid Range

**Karakteristik:**
- Cocok untuk multitasking
- Performa cukup baik
- Penyimpanan lebih besar
- Cocok untuk gaming menengah

**Rata-rata Cluster:**
- RAM : 4.23 GB
- Storage : 57.17 GB
- Battery : 3393.63 mAh
- Kamera : 12.14 MP
                """
            )

        elif cluster == 2:

            st.warning(
                """
### 🔋 Cluster 2 — Entry Level Battery Besar

**Karakteristik:**
- Cocok untuk penggunaan ringan
- Daya tahan baterai sangat baik
- Harga relatif murah
- Cocok untuk penggunaan harian jangka panjang

**Rata-rata Cluster:**
- RAM : 3.90 GB
- Storage : 46.11 GB
- Battery : 5791.48 mAh
- Kamera : 6.92 MP
                """
            )

        else:

            st.write(f"Cluster terdeteksi: {cluster}")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
