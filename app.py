import streamlit as st
import pandas as pd
import pickle

# ==========================
# LOAD MODEL & SCALER
# ==========================
with open("rf_optuna.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Klasifikasi Cluster HP Bekas",
    page_icon="📱",
    layout="centered"
)

# ==========================
# HEADER
# ==========================
st.title("📱 Klasifikasi Cluster HP Bekas")

st.markdown("""
Aplikasi ini menggunakan:

- **K-Means Clustering** untuk membentuk cluster HP bekas
- **Random Forest + Optuna** untuk mengklasifikasikan HP baru ke dalam cluster yang telah terbentuk

Masukkan spesifikasi HP untuk mengetahui kategori cluster-nya.
""")

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

    try:

        # URUTAN HARUS SAMA DENGAN SAAT TRAINING
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

        # Scaling
        data_scaled = scaler.transform(data)

        # Prediksi klasifikasi
        cluster = int(model.predict(data_scaled)[0])

        st.divider()

        # ==========================
        # NAMA CLUSTER
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
        st.subheader("📊 Interpretasi Hasil")

        if cluster == 0:

            st.info("""
### 📱 Cluster 0 — Entry Level

**Karakteristik:**
- Cocok untuk penggunaan ringan
- Harga relatif terjangkau
- RAM dan storage standar
- Cocok untuk browsing, chat, dan media sosial

**Profil Cluster:**
- RAM rata-rata: 3.55 GB
- Storage rata-rata: 45.13 GB
- Battery rata-rata: 1946.14 mAh
- Kamera rata-rata: 5.69 MP
""")

        elif cluster == 1:

            st.success("""
### 🚀 Cluster 1 — Mid Range

**Karakteristik:**
- Cocok untuk multitasking
- Performa lebih baik
- Penyimpanan lebih besar
- Cocok untuk gaming menengah

**Profil Cluster:**
- RAM rata-rata: 4.23 GB
- Storage rata-rata: 57.17 GB
- Battery rata-rata: 3393.63 mAh
- Kamera rata-rata: 12.14 MP
""")

        elif cluster == 2:

            st.warning("""
### 🔋 Cluster 2 — Entry Level Battery Besar

**Karakteristik:**
- Cocok untuk penggunaan ringan
- Daya tahan baterai sangat baik
- Harga relatif terjangkau
- Cocok untuk penggunaan harian jangka panjang

**Profil Cluster:**
- RAM rata-rata: 3.90 GB
- Storage rata-rata: 46.11 GB
- Battery rata-rata: 5791.48 mAh
- Kamera rata-rata: 6.92 MP
""")

        else:
            st.write(f"Cluster terdeteksi: {cluster}")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
