import streamlit as st
import pandas as pd
import pickle

# load model & scaler
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📱 Clustering HP Bekas")

st.write("Masukkan spesifikasi HP:")

screen_size = st.number_input("Screen Size")
rear_camera = st.number_input("Rear Camera (MP)")
front_camera = st.number_input("Front Camera (MP)")
internal_memory = st.number_input("Internal Memory (GB)")
ram = st.number_input("RAM (GB)")
battery = st.number_input("Battery (mAh)")
weight = st.number_input("Weight (gram)")
days_used = st.number_input("Days Used")

if st.button("Prediksi Cluster"):
    
    data = pd.DataFrame([[
        screen_size,
        rear_camera,
        front_camera,
        internal_memory,
        ram,
        battery,
        weight,
        days_used
    ]])

    # scaling
    data_scaled = scaler.transform(data)

    # prediksi
    cluster = kmeans.predict(data_scaled)[0]

    st.success(f"Hasil Cluster: {cluster}")