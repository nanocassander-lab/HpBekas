import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Klasifikasi HP Bekas",
    page_icon="📱",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("hasil_clustering_hp.csv")
    return df

@st.cache_resource
def train_model(df):
    features = ["ram", "internal_memory", "battery", "rear_camera_mp", "screen_size"]
    target = "Cluster"

    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

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

    akurasi = model.score(X_test, y_test)
    return model, scaler, akurasi

df_clustered = load_data()
model, scaler, akurasi = train_model(df_clustered)

# ==========================
# MAPPING CONTOH TIPE HP
# ==========================
CONTOH_TIPE_HP = {
    0: {
        "Samsung":  ["Samsung Galaxy A03", "Samsung Galaxy A03s", "Samsung Galaxy M02"],
        "LG":       ["LG K22", "LG K41S", "LG K51S"],
        "Nokia":    ["Nokia C21", "Nokia C31", "Nokia G11"],
        "Alcatel":  ["Alcatel 1S", "Alcatel 3L", "Alcatel 1B"],
        "Huawei":   ["Huawei Y6s", "Huawei Y5 2019", "Huawei Y6 2019"],
        "Micromax": ["Micromax IN 2b", "Micromax IN 2c"],
        "ZTE":      ["ZTE Blade A31", "ZTE Blade A51"],
        "Lenovo":   ["Lenovo K12", "Lenovo K13"],
        "Motorola": ["Motorola Moto E7", "Motorola Moto E7i Power"],
        "Xiaomi":   ["Xiaomi Redmi 9A", "Xiaomi Redmi 9C"],
    },
    1: {
        "Samsung":  ["Samsung Galaxy A25", "Samsung Galaxy A35", "Samsung Galaxy M34"],
        "Huawei":   ["Huawei P30 Lite", "Huawei Nova 5T", "Huawei Y8p"],
        "Honor":    ["Honor X8", "Honor X7", "Honor 90 Lite"],
        "Xiaomi":   ["Xiaomi Redmi Note 11", "Xiaomi Redmi 10", "Xiaomi POCO M4"],
        "Vivo":     ["Vivo Y35", "Vivo Y21", "Vivo Y16"],
        "Oppo":     ["Oppo A57", "Oppo A77", "Oppo Reno 8 Lite"],
        "Lenovo":   ["Lenovo K14 Plus", "Lenovo Tab M10"],
        "LG":       ["LG K92", "LG Velvet", "LG K71"],
        "Realme":   ["Realme C33", "Realme 9i", "Realme Narzo 50"],
        "Motorola": ["Motorola Moto G32", "Motorola Moto G42", "Motorola Moto G52"],
    },
    2: {
        "Samsung":  ["Samsung Galaxy M23", "Samsung Galaxy M33", "Samsung Galaxy F23"],
        "Huawei":   ["Huawei Y9 Prime", "Huawei Y9a", "Huawei Mate 40 Lite"],
        "Lenovo":   ["Lenovo K12 Pro", "Lenovo Tab P11"],
        "Apple":    ["iPhone 11", "iPhone SE 2020", "iPhone XR"],
        "Asus":     ["Asus Zenfone Max Pro M2", "Asus ROG Phone 3 Lite"],
        "LG":       ["LG G8X ThinQ", "LG Wing"],
        "Acer":     ["Acer Liquid Zest Plus", "Acer Iconia Tab"],
        "Realme":   ["Realme C25Y", "Realme C55", "Realme Narzo 50A"],
        "Xiaomi":   ["Xiaomi Redmi 10C", "Xiaomi POCO C55"],
        "Nokia":    ["Nokia G50", "Nokia XR20"],
    }
}

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

st.caption(f"Akurasi model: {akurasi*100:.2f}%")

st.divider()

# ==========================
# INPUT
# ==========================
st.subheader("🔧 Spesifikasi HP")

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

    screen_size_cm = screen_size * 2.54

    data = pd.DataFrame(
        [[ram, internal_memory, battery, rear_camera, screen_size_cm]],
        columns=["ram", "internal_memory", "battery", "rear_camera_mp", "screen_size"]
    )

    data_scaled = scaler.transform(data)
    cluster = int(model.predict(data_scaled)[0])
    proba = model.predict_proba(data_scaled)[0]

    st.divider()

    # ==========================
    # HASIL CLUSTER
    # ==========================
    if cluster == 0:
        st.success("📱 Cluster 0 — Entry Level")
        kategori = """
        Cocok untuk penggunaan ringan seperti:
        - Chatting
        - Media sosial
        - Browsing
        - Aktivitas harian sederhana
        """
    elif cluster == 1:
        st.success("🚀 Cluster 1 — Mid Range")
        kategori = """
        Cocok untuk:
        - Multitasking
        - Gaming menengah
        - Produktivitas harian
        - Penggunaan yang lebih intensif
        """
    else:
        st.success("🔋 Cluster 2 — Entry Level Battery Besar")
        kategori = """
        Cocok untuk:
        - Penggunaan ringan
        - Mobilitas tinggi
        - Pengguna yang membutuhkan daya tahan baterai lama
        """

    st.markdown(kategori)

    # ==========================
    # MERK HP DI CLUSTER INI
    # ==========================
    st.subheader("🏷️ Merk HP dalam Cluster Ini")

    df_cluster = df_clustered[df_clustered["Cluster"] == cluster]

    brand_counts = (
        df_cluster[df_cluster["device_brand"] != "Others"]["device_brand"]
        .value_counts()
        .reset_index()
    )
    brand_counts.columns = ["Merk HP", "Jumlah Data"]
    top_brands = brand_counts.head(10)

    col_brand1, col_brand2 = st.columns([2, 3])

    with col_brand1:
        st.markdown("**Top Merk HP di Cluster Ini:**")
        for _, row in top_brands.iterrows():
            st.markdown(f"- **{row['Merk HP']}** ({row['Jumlah Data']} unit)")

    with col_brand2:
        st.bar_chart(top_brands.set_index("Merk HP")["Jumlah Data"])

    st.divider()

    # ==========================
    # CONTOH TIPE HP PER MERK
    # ==========================
    st.subheader("📱 Contoh Tipe HP dalam Cluster Ini")

    tipe_cluster = CONTOH_TIPE_HP[cluster]

    cols = st.columns(2)
    for i, (merk, tipe_list) in enumerate(tipe_cluster.items()):
        with cols[i % 2]:
            with st.expander(f"**{merk}**"):
                for tipe in tipe_list:
                    st.markdown(f"- {tipe}")

    st.divider()

    # ==========================
    # DETAIL SPESIFIKASI PER MERK
    # ==========================
    st.subheader("🔍 Detail Spesifikasi per Merk")

    merk_list = top_brands["Merk HP"].tolist()
    selected_brand = st.selectbox("Pilih merk untuk lihat rata-rata speknya:", merk_list)

    df_brand = df_cluster[df_cluster["device_brand"] == selected_brand]
    avg_specs = df_brand[["ram", "internal_memory", "battery", "rear_camera_mp", "screen_size"]].mean()
    avg_specs["screen_size"] = avg_specs["screen_size"] / 2.54

    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    col_d1.metric("Rata-rata RAM", f"{avg_specs['ram']:.1f} GB")
    col_d2.metric("Rata-rata Storage", f"{avg_specs['internal_memory']:.0f} GB")
    col_d3.metric("Rata-rata Battery", f"{avg_specs['battery']:.0f} mAh")
    col_d4.metric("Rata-rata Kamera", f"{avg_specs['rear_camera_mp']:.1f} MP")
    col_d5.metric("Rata-rata Layar", f"{avg_specs['screen_size']:.1f} inch")

    st.markdown(f"**Total {selected_brand} di cluster ini: {len(df_brand)} unit**")

    with st.expander(f"📋 Lihat data {selected_brand} di cluster ini"):
        tampil_cols = ["device_brand", "ram", "internal_memory", "battery",
                       "rear_camera_mp", "screen_size", "release_year", "normalized_used_price"]
        df_show = df_brand[tampil_cols].copy()
        df_show["screen_size"] = (df_show["screen_size"] / 2.54).round(1)
        df_show.columns = ["Merk", "RAM (GB)", "Storage (GB)", "Battery (mAh)",
                            "Kamera (MP)", "Layar (inch)", "Tahun Rilis", "Harga Bekas (norm)"]
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True)

    st.divider()

    # ==========================
    # RINGKASAN SPESIFIKASI INPUT
    # ==========================
    st.subheader("📋 Ringkasan Spesifikasi Input")

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
    # PROFIL CLUSTER
    # ==========================
    st.subheader("📊 Profil Cluster")

    if cluster == 0:
        st.info("""
**Cluster 0 — Entry Level**

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
        st.info("""
**Cluster 1 — Mid Range**

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
        st.info("""
**Cluster 2 — Entry Level Battery Besar**

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
