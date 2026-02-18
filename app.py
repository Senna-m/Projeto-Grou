import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostRegressor

st.set_page_config(page_title="Painel Veículos & Diesel", layout="wide")

# ---------- Carregar modelo ----------
@st.cache_resource
def load_model():
    m = CatBoostRegressor()
    m.load_model("modelo_diesel.cbm")  # precisa estar na mesma pasta do app.py
    return m

model = load_model()

# ---------- Limpeza (igual ao seu notebook) ----------
def limpar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df_model = df[["VEICULO", "KM RODADA", "DIESEL"]].copy()
    df_model = df_model.dropna(how="all")

    df_model["VEICULO"] = df_model["VEICULO"].astype(str).str.strip()
    df_model["KM RODADA"] = pd.to_numeric(df_model["KM RODADA"], errors="coerce")

    s = df_model["DIESEL"].astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    df_model["DIESEL"] = pd.to_numeric(s, errors="coerce")

    df_model = df_model.dropna(subset=["VEICULO", "KM RODADA", "DIESEL"])
    df_model = df_model[(df_model["KM RODADA"] > 0) & (df_model["DIESEL"] > 0)]
    return df_model

def recomendar_veiculos(df_model: pd.DataFrame, km_rota: float, top_n: int = 5) -> pd.DataFrame:
    veiculos = df_model[["VEICULO"]].drop_duplicates().reset_index(drop=True)
    veiculos["KM RODADA"] = km_rota
    veiculos["LITROS_PREVISTOS"] = model.predict(veiculos[["VEICULO", "KM RODADA"]])
    return veiculos.sort_values("LITROS_PREVISTOS").head(top_n)

# ---------- UI ----------
st.title("🚚 Painel — Histórico de Veículos e Consumo (Diesel)")

# Lê CSVs locais da pasta data/
data_dir = Path("data")
csvs = sorted([p for p in data_dir.glob("*.csv")])

if not data_dir.exists():
    st.error('Crie a pasta **data/** e coloque seu CSV lá dentro.')
    st.stop()

if not csvs:
    st.error('Não encontrei CSV em **data/**. Coloque um arquivo como "data/abastecimento.csv".')
    st.stop()

arquivo_escolhido = st.selectbox(
    "Selecione o CSV da pasta data/",
    options=csvs,
    format_func=lambda p: p.name
)

# Se seu CSV usa ; como separador, troque sep=";" abaixo.
df_raw = pd.read_csv(arquivo_escolhido)

st.subheader("Prévia do CSV (bruto)")
st.dataframe(df_raw.head(30), use_container_width=True)

try:
    df_model = limpar_df(df_raw)
except Exception as e:
    st.error(f"Erro ao limpar/selecionar colunas: {e}")
    st.info('Confirme se o CSV tem as colunas: "VEICULO", "KM RODADA", "DIESEL".')
    st.stop()

st.subheader("Dados após limpeza")
st.write(f"Linhas: {df_model.shape[0]} | Colunas: {df_model.shape[1]}")
st.dataframe(df_model.head(30), use_container_width=True)

st.subheader("Filtros")
veiculo_sel = st.selectbox("Veículo", sorted(df_model["VEICULO"].unique().tolist()))
mostrar_n = st.slider("Mostrar últimos N registros", 10, 300, 60)

df_v = df_model[df_model["VEICULO"] == veiculo_sel].tail(mostrar_n)

st.subheader(f"Consumo do veículo: {veiculo_sel}")
fig = plt.figure()
plt.scatter(df_v["KM RODADA"], df_v["DIESEL"])
plt.xlabel("KM RODADA")
plt.ylabel("DIESEL (litros)")
st.pyplot(fig)

st.subheader("Simular previsão para 1 veículo")
km_sim = st.number_input("KM da rota", min_value=1.0, value=100.0, step=1.0)
if st.button("Prever litros para este veículo"):
    X_one = pd.DataFrame({"VEICULO": [veiculo_sel], "KM RODADA": [km_sim]})
    litros = float(model.predict(X_one)[0])
    st.success(f"Estimativa: **{litros:.2f} litros**")

st.subheader("Ranking de melhores veículos para uma rota")
km_rota = st.number_input("KM da rota (ranking)", min_value=1.0, value=120.0, step=1.0, key="km_rota")
top_n = st.number_input("Top N", min_value=1, max_value=50, value=5, step=1)

if st.button("Gerar ranking"):
    ranking = recomendar_veiculos(df_model, km_rota=float(km_rota), top_n=int(top_n))
    st.dataframe(ranking, use_container_width=True)
