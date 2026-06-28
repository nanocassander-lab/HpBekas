import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Klasifikasi HP Bekas",
    page_icon="📱",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
with open("rf_optuna.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# DAFTAR MERK HP
# ==========================
DAFTAR_MERK = [
    'Acer', 'Alcatel', 'Apple', 'Asus', 'BlackBerry', 'Celkon', 'Coolpad',
    'Gionee', 'Google', 'HTC', 'Honor', 'Huawei', 'Infinix', 'Karbonn',
    'LG', 'Lava', 'Lenovo', 'Meizu', 'Micromax', 'Microsoft', 'Motorola',
    'Nokia', 'OnePlus', 'Oppo', 'Others', 'Panasonic', 'Realme', 'Samsung',
    'Sony', 'Spice', 'Vivo', 'XOLO', 'Xiaomi', 'ZTE'
]

# ==========================
# HEADER
# ==========================
st.title("📱 Sistem Klasifikasi Cluster HP Bekas")

st.markdown("""
Sistem ini menggabungkan:

- **K-Means Clustering** untuk membentuk kelompok HP bekas
- **Random Forest yang dituning menggunakan Optuna** untuk mengklasifikasikan data HP baru ke dalam cluster yang telah terbentuk

Masukkan spesifikasi HP untuk mengetahui kategori cluster-nya.
""")

st.divider()

# ==========================
# INPUT
# ==========================
st.subheader("🔧 Spesifikasi HP")

# Baris 1: Merk HP (full width)
device_brand = st.selectbox(
    "Merk HP",
    options=DAFTAR_MERK,
    index=DAFTAR_MERK.index('Samsung'),
    help="Pilih merk HP bekas yang ingin diklasifikasikan"
)

col1, col2 = st.columns(2)

with col1:

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

with col2:

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

st.divider()

# ==========================
# PREDIKSI
# ==========================
if st.button("🚀 Prediksi Cluster", use_container_width=True):

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

    data_scaled = scaler.transform(data)

    cluster = int(model.predict(data_scaled)[0])

    # Probabilitas
    proba = model.predict_proba(data_scaled)[0]

    st.divider()

    # ==========================
    # HASIL + MERK HP
    # ==========================
    if cluster == 0:

        st.success(f"📱 {device_brand} — Cluster 0: Entry Level")

        kategori = """
        Cocok untuk penggunaan ringan seperti:
        - Chatting
        - Media sosial
        - Browsing
        - Aktivitas harian sederhana
        """

    elif cluster == 1:

        st.success(f"🚀 {device_brand} — Cluster 1: Mid Range")

        kategori = """
        Cocok untuk:
        - Multitasking
        - Gaming menengah
        - Produktivitas harian
        - Penggunaan yang lebih intensif
        """

    else:

        st.success(f"🔋 {device_brand} — Cluster 2: Entry Level Battery Besar")

        kategori = """
        Cocok untuk:
        - Penggunaan ringan
        - Mobilitas tinggi
        - Pengguna yang membutuhkan daya tahan baterai lama
        """

    st.markdown(kategori)

    # ==========================
    # RINGKASAN SPESIFIKASI
    # ==========================
    st.subheader("📋 Ringkasan Spesifikasi")

    # Tampilkan merk di baris tersendiri agar menonjol
    st.markdown(f"**Merk HP:** {device_brand}")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("RAM", f"{ram} GB")
    c2.metric("Storage", f"{internal_memory} GB")
    c3.metric("Battery", f"{battery} mAh")
    c4.metric("Camera", f"{rear_camera} MP")
    c5.metric("Screen", f"{screen_size}\"")

    st.divider()

    # ==========================
    # PROBABILITAS
    # ==========================
    st.subheader("🎯 Tingkat Keyakinan Model")

    st.progress(float(proba[0]))
    st.write(f"Cluster 0 : {proba[0]*100:.2f}%")

    st.progress(float(proba[1]))
    st.write(f"Cluster 1 : {proba[1]*100:.2f}%")

    st.progress(float(proba[2]))
    st.write(f"Cluster 2 : {proba[2]*100:.2f}%")

    st.divider()

    # ==========================
    # INTERPRETASI
    # ==========================
    st.subheader("📊 Profil Cluster")

    if cluster == 0:

        st.info(f"""
**Cluster 0 — Entry Level** *(diprediksi untuk {device_brand})*

Rata-rata:
- RAM : 3.55 GB
- Storage : 45.13 GB
- Battery : 1946.14 mAh
- Kamera : 5.69 MP

Karakteristik:
- Harga relatif murah
- Spesifikasi standar
- Cocok untuk penggunaan ringan
""")

    elif cluster == 1:

        st.info(f"""
**Cluster 1 — Mid Range** *(diprediksi untuk {device_brand})*

Rata-rata:
- RAM : 4.23 GB
- Storage : 57.17 GB
- Battery : 3393.63 mAh
- Kamera : 12.14 MP

Karakteristik:
- Cocok multitasking
- Performa lebih baik
- Cocok untuk gaming menengah
""")

    else:

        st.info(f"""
**Cluster 2 — Entry Level Battery Besar** *(diprediksi untuk {device_brand})*

Rata-rata:
- RAM : 3.90 GB
- Storage : 46.11 GB
- Battery : 5791.48 mAh
- Kamera : 6.92 MP

Karakteristik:
- Daya tahan baterai sangat baik
- Cocok penggunaan harian
- Harga relatif terjangkau
""")

st.divider()

st.caption(
    "Model yang digunakan: Random Forest hasil tuning Optuna dengan target cluster dari K-Means Clustering."
)
