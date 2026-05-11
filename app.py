import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="Predicción de gravedad de accidentes",
    page_icon="🚗",
    layout="centered"
)

# ============================================================
# Cargar modelo serializado
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_accidentalidad_envigado.pkl")

modelo_artefacto = cargar_modelo()
modelo = modelo_artefacto["modelo"]

# ============================================================
# Título aplicación
# ============================================================

st.title("🚗 Predicción de gravedad de accidentes de tránsito")

st.write("""
Esta aplicación utiliza un modelo de Random Forest entrenado con datos
de accidentalidad del municipio de Envigado para estimar el nivel de gravedad
de un accidente de tránsito.
""")

# ============================================================
# Variables de entrada
# ============================================================

st.subheader("Ingrese las características del accidente")

mes = st.selectbox(
    "Mes",
    list(range(1, 13))
)

dia_semana = st.selectbox(
    "Día de la semana",
    list(range(1, 8))
)

hora = st.selectbox(
    "Hora del accidente",
    list(range(0, 24))
)

resultado_beodez = st.number_input(
    "Resultado de beodez",
    min_value=0,
    step=1
)

clase_accidente = st.selectbox(
    "Clase de accidente",
    [
        "CHOQUE",
        "ATROPELLO",
        "CAIDA OCUPANTE",
        "VOLCAMIENTO",
        "OTRO"
    ]
)

causa = st.text_input(
    "Causa del accidente"
)

barrio = st.text_input(
    "Barrio"
)

# ============================================================
# Predicción
# ============================================================

if st.button("Predecir gravedad"):

    datos_entrada = pd.DataFrame([{
        "mes": mes,
        "dia_semana": dia_semana,
        "hora": hora,
        "resultado_beodez": resultado_beodez,
        "clase_accidente": clase_accidente,
        "causa": causa,
        "barrio": barrio
    }])

    # Asegurar que las columnas de entrada coincidan con las del entrenamiento
    # Esto es crucial si el modelo fue entrenado con OneHotEncoder o similar
    modelo_columnas = modelo_artefacto["columnas"]
    for col in modelo_columnas:
        if col not in datos_entrada.columns:
            datos_entrada[col] = None # O un valor predeterminado apropiado

    datos_entrada = datos_entrada[modelo_columnas]

    prediccion = modelo.predict(datos_entrada)[0]

    st.success(f"Nivel de gravedad estimado: {prediccion}")
