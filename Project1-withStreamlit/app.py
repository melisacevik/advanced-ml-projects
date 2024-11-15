import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Model ve veri yükleme için global değişkenler
__data_columns = None
__locations = None
__model = None

# Artifacts klasöründen model ve kolon bilgilerini yükleyen fonksiyon
def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    # JSON dosyasını yükle
    try:
        with open("/Users/melisacevik/PycharmProjects/ML-Advanced/Project1/server/artifacts/columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # İlk 3 kolon: sqft, bath, bhk
    except FileNotFoundError:
        st.error("columns.json dosyası bulunamadı! Lütfen doğru dosyayı yüklediğinizden emin olun.")
        st.stop()

    # Modeli yükle
    try:
        with open("/Users/melisacevik/PycharmProjects/ML-Advanced/Project1/model/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model dosyası (banglore_home_prices_model.pickle) bulunamadı! Lütfen doğru dosyayı yüklediğinizden emin olun.")
        st.stop()

# Tahmin fonksiyonu
def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# Artifacts'leri yükle
load_saved_artifacts()

# Streamlit Başlığı
st.title("Bangalore Ev Fiyat Tahmini Uygulaması")
st.write("""
Bu uygulama, Bangalore'daki ev fiyatlarını tahmin etmek için makine öğrenmesi modeli kullanır.
Lütfen aşağıdaki bilgileri doldurun.
""")

# Kullanıcıdan girdiler al
st.header("Girdi Bilgileri")
location = st.selectbox("Konum:", options=__locations)
total_sqft = st.number_input("Toplam Metrekare:", min_value=500, max_value=5000, step=50, value=1000)
bhk = st.selectbox("BHK (Oda Sayısı):", options=[1, 2, 3, 4, 5])
bath = st.selectbox("Banyo Sayısı:", options=[1, 2, 3, 4, 5])

# Tahmin yapma işlemi
if st.button("Fiyatı Tahmin Et"):
    price = get_estimated_price(location, total_sqft, bhk, bath)
    st.success(f"Bu özelliklere sahip bir evin tahmini fiyatı: ₹ {price:,.2f}")

# Ek bilgiler veya analiz
st.sidebar.title("Hakkında")
st.sidebar.info("""
Bu proje, Bangalore'daki ev fiyatlarını tahmin etmek için bir makine öğrenmesi modeli kullanır. 
""")
