import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# 1. Configuração da Página
st.set_page_config(page_title="NutriML - Ponto Clínico", layout="centered", page_icon="🏥")

# Design Profissional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .result-card { background-color: #ffffff; padding: 20px; border-radius: 15px; border-left: 5px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. Carregamento do Modelo Pré-Treinado (Ultra Rápido)
@st.cache_resource
def carregar_modelo():
    modelo = xgb.XGBRegressor()
    # Verifica se o arquivo do Colab foi subido
    if os.path.exists("modelo_nutri.json"):
        modelo.load_model("modelo_nutri.json")
    return modelo

modelo_ia = carregar_modelo()

# --- INTERFACE ---
st.title("🏥 NutriML - Gasto Energético")
st.caption("Desenvolvido por Ponto Clínico")

with st.form("form_nutri"):
    st.info("Insira os dados para predição via Machine Learning.")
    col1, col2 = st.columns(2)
    
    with col1:
        peso = st.number_input("Peso (kg)", value=70, step=1, format="%d")
        altura = st.number_input("Estatura (cm)", value=170, step=1, format="%d")
        idade = st.number_input("Idade", value=50, step=1, format="%d")
    
    with col2:
        sexo = st.selectbox("Sexo Biológico", ["Masculino", "Feminino"])
        fc = st.number_input("FC (bpm)", value=80, step=1, format="%d")
        temp = st.number_input("Temp. Axilar (°C)", value=36.5, step=0.1)

    with st.expander("🩺 Parâmetros Avançados"):
        col3, col4 = st.columns(2)
        with col3:
            pcr = st.number_input("PCR (mg/dL)", value=1.0, step=0.1)
            ureia = st.number_input("Ureia (mg/dL)", value=30, step=1, format="%d")
            # VOLTOU: Creatinina com casas decimais
            creat = st.number_input("Creatinina (mg/dL)", value=1.0, step=0.1)
        with col4:
            ph = st.number_input("pH Arterial", value=7.40, step=0.01)
            vm = st.number_input("Volume Minuto", value=0, step=1, format="%d")
            peep = st.number_input("PEEP", value=0, step=1, format="%d")

    btn = st.form_submit_button("CALCULAR GASTO METABÓLICO")

if btn:
    if modelo_ia:
        s_num = 1 if sexo == "Feminino" else 0
        entrada = pd.DataFrame([[idade, peso, altura, s_num, fc, temp, pcr, ureia, creat, ph, vm, peep]], 
                               columns=['idade', 'peso', 'altura', 'sexo', 'fc', 'temp', 'pcr', 'ureia', 'creatinina', 'ph_arterial', 'volume_minuto', 'peep'])
        
        predicao = modelo_ia.predict(entrada)[0]
        
        st.markdown("---")
        st.markdown(f"""
            <div class="result-card">
                <h3 style='color: #155724; margin-top: 0;'>Gasto Energético Predito</h3>
                <p style='font-size: 32px; font-weight: bold; margin-bottom: 5px;'>{predicao:.0f} Kcal/dia</p>
                <p style='font-size: 14px; color: #666;'>Algoritmo XGBoost | Ponto Clínico</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Erro: Arquivo 'modelo_nutri.json' não encontrado no GitHub.")

st.markdown("---")
st.caption("⚠️ Nota: Suporte à decisão clínica. Baseado no estudo Ponce et al. (2020).")
