import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL
# =========================
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Clustering HP Bekas",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Clustering HP Bekas")
st.write("Masukkan spesifikasi HP untuk mengetahui kategori cluster HP bekas.")

st.divider()

# =========================
# INPUT USER
# =========================
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

# =========================
# PREDIKSI
# =========================
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
        # scaling
        data_scaled = scaler.transform(data)

        # prediksi
        cluster = int(kmeans.predict(data_scaled)[0])

        st.success(f"Hasil Cluster: {cluster}")

        st.subheader("📋 Ringkasan Spesifikasi")

        st.write(f"**RAM:** {ram} GB")
        st.write(f"**Internal Memory:** {internal_memory} GB")
        st.write(f"**Battery:** {battery} mAh")
        st.write(f"**Rear Camera:** {rear_camera} MP")
        st.write(f"**Screen Size:** {screen_size} inch")

        st.divider()

        st.subheader("📊 Interpretasi Cluster")

        if cluster == 0:
            st.info("""
            **Cluster 0**

            Karakteristik:
            - Spesifikasi dasar
            - Cocok untuk penggunaan ringan
            - Harga cenderung lebih terjangkau
            """)

        elif cluster == 1:
            st.info("""
            **Cluster 1**

            Karakteristik:
            - Spesifikasi menengah
            - Cocok untuk penggunaan sehari-hari
            - Performa cukup baik untuk multitasking
            """)

        elif cluster == 2:
            st.info("""
            **Cluster 2**

            Karakteristik:
            - Spesifikasi tinggi
            - RAM dan memori besar
            - Kamera dan baterai unggul
            - Cocok untuk gaming dan produktivitas
            """)

        else:
            st.warning(f"Cluster terdeteksi: {cluster}")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
