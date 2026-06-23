import streamlit as st
import pandas as pd
import pickle

# Load model dan scaler
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Tampilan aplikasi
st.set_page_config(page_title="Clustering HP Bekas", page_icon="📱")

st.title("📱 Clustering HP Bekas")
st.write("Masukkan spesifikasi HP untuk mengetahui cluster-nya.")

# Input fitur
screen_size = st.number_input(
    "Screen Size (inch)",
    min_value=0.0,
    value=6.5,
    step=0.1
)

rear_camera = st.number_input(
    "Rear Camera (MP)",
    min_value=0.0,
    value=50.0,
    step=1.0
)

internal_memory = st.number_input(
    "Internal Memory (GB)",
    min_value=0,
    value=128,
    step=1
)

ram = st.number_input(
    "RAM (GB)",
    min_value=0,
    value=8,
    step=1
)

battery = st.number_input(
    "Battery (mAh)",
    min_value=0,
    value=5000,
    step=100
)

# Tombol prediksi
if st.button("Prediksi Cluster"):

    # Harus sama persis dengan urutan fitur saat training
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

        # Prediksi cluster
        cluster = kmeans.predict(data_scaled)[0]

        st.success(f"Hasil Cluster: {cluster}")

        # Interpretasi cluster (opsional)
        if cluster == 0:
            st.info("Cluster 0")
        elif cluster == 1:
            st.info("Cluster 1")
        elif cluster == 2:
            st.info("Cluster 2")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
