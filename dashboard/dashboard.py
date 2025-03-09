import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    file_path = "dashboard/submission_ari.csv"
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df


df = load_data()


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1163/1163624.png", width=100)
st.sidebar.title("🌍 Air Quality Dashboard")
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Data")
station_choice = st.sidebar.selectbox("Pilih Stasiun", df['station'].unique())
df_filtered = df[df['station'] == station_choice]

# 🔹 Header Dashboard
st.title("🌍 Air Quality Dashboard")
st.markdown("### Analisis Kualitas Udara berdasarkan Dataset")
st.markdown("---")

### 📌 **1. Heatmap Tren Polusi (Melihat Polusi Tertinggi)**
st.subheader("📊 Heatmap Tren PM2.5 (2013-2017)")
df_pivot = df_filtered.pivot_table(values="PM2.5", index="year", columns="month", aggfunc="mean")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df_pivot, cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5, ax=ax)
st.pyplot(fig)

### 📌 **2. Tren PM2.5 per Bulan**
st.subheader("📈 Tren PM2.5 per Bulan")
df_monthly = df_filtered.groupby(['year', 'month'])['PM2.5'].mean().reset_index()
df_monthly['time'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))
fig_pm25_trend = px.line(df_monthly, x='time', y='PM2.5', labels={'time': 'Waktu', 'PM2.5': 'PM2.5 Rata-rata'})
st.plotly_chart(fig_pm25_trend, use_container_width=True)

### 📌 **3. Korelasi Polusi dengan Faktor Cuaca**
st.subheader("🔗 Korelasi PM2.5 dengan Faktor Cuaca")
corr_matrix = df_filtered[['PM2.5', 'TEMP', 'PRES', 'DEWP', 'WSPM']].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

### 📌 **4. Regresi Linier PM2.5 vs Faktor Cuaca**
st.subheader("📉 Analisis Regresi PM2.5 dengan Faktor Cuaca")
selected_var = st.selectbox("Pilih Variabel Cuaca:", ['TEMP', 'PRES', 'DEWP', 'WSPM'])
df_filtered = df_filtered.dropna(subset=['PM2.5', selected_var])
X = sm.add_constant(df_filtered[selected_var])  # Menambahkan Intersep
y = df_filtered['PM2.5']
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Plot hasil regresi
fig, ax = plt.subplots()
ax.scatter(df_filtered[selected_var], df_filtered['PM2.5'], alpha=0.3, label="Data Aktual")
ax.plot(df_filtered[selected_var], predictions, color='red', label="Regresi Linier")
ax.set_xlabel(selected_var)
ax.set_ylabel("PM2.5")
ax.legend()
st.pyplot(fig)

# Menampilkan Koefisien Regresi
st.markdown(f"**Koefisien Regresi:** {model.params[1]:.2f} (p-value: {model.pvalues[1]:.5f})")
st.markdown(f"**R-squared:** {model.rsquared:.3f} (Menjelaskan {model.rsquared * 100:.1f}% Variabilitas PM2.5)")


