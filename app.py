import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción de gravedad de accidentes",
    page_icon="🚗",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_accidentalidad_envigado.pkl")

modelo = cargar_modelo()

st.title("🚗 Predicción de gravedad de accidentes de tránsito")

st.markdown("""
Esta aplicación utiliza un modelo Random Forest entrenado con datos de accidentalidad del municipio de Envigado
para estimar el nivel de gravedad de un accidente de tránsito.

**Fuente del dataset:** Datos Abiertos Colombia  
**Dataset:** Accidentalidad Municipio de Envigado  
**Autor / entidad responsable:** Alcaldía de Envigado  
**Enlace:** https://www.datos.gov.co/Transporte/Accidentalidad-Municipio-de-Envigado/t5sw-amxr/about_data
""")

st.divider()

st.subheader("Ingrese las características del accidente")

with st.expander("¿Cómo debe diligenciarse cada campo?"):
    st.markdown("""
    **Mes:** seleccione el mes en el que ocurrió el accidente.

    **Día de la semana:** seleccione el día correspondiente al accidente.

    **Hora del accidente:** indique la hora aproximada en formato de 0 a 23.

    **Resultado de beodez:** registre el valor reportado en la base de datos. Si no se cuenta con información, puede dejarse en 0.

    **Clase de accidente:** seleccione el tipo de evento registrado, por ejemplo choque, atropello, volcamiento o caída de ocupante.

    **Causa del accidente:** escriba la causa principal reportada. Si no se conoce, puede usar “NO REPORTADA”.

    **Barrio:** escriba el barrio donde ocurrió el accidente. Si no se conoce, puede usar “NO REPORTADO”.
    """)

meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

dias = {
    "Lunes": 1,
    "Martes": 2,
    "Miércoles": 3,
    "Jueves": 4,
    "Viernes": 5,
    "Sábado": 6,
    "Domingo": 7
}

with st.form("formulario_prediccion"):

    col1, col2 = st.columns(2)

    with col1:
        mes_nombre = st.selectbox("Mes", list(meses.keys()))
        dia_nombre = st.selectbox("Día de la semana", list(dias.keys()))
        hora = st.selectbox("Hora del accidente", list(range(0, 24)))

    with col2:
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

        barrio = st.text_input("Barrio", value="NO REPORTADO")

    causa = st.text_input("Causa del accidente", value="NO REPORTADA")

    enviar = st.form_submit_button("Predecir gravedad")

if enviar:

    datos_entrada = pd.DataFrame([{
        "mes": meses[mes_nombre],
        "dia_semana": dias[dia_nombre],
        "hora": hora,
        "resultado_beodez": resultado_beodez,
        "clase_accidente": clase_accidente,
        "causa": causa.upper().strip(),
        "barrio": barrio.upper().strip()
    }])

    prediccion = modelo.predict(datos_entrada)[0]

    st.divider()
    st.subheader("Resultado de la predicción")

    if prediccion == "SOLO DAÑOS":
        st.success(f"Nivel de gravedad estimado: {prediccion}")
        st.write("El modelo estima que el accidente tendría una afectación principalmente material.")

    elif prediccion == "HERIDOS":
        st.warning(f"Nivel de gravedad estimado: {prediccion}")
        st.write("El modelo estima que el accidente podría estar asociado con personas lesionadas.")

    elif prediccion == "MUERTOS":
        st.error(f"Nivel de gravedad estimado: {prediccion}")
        st.write("El modelo estima un evento de alta gravedad. Esta categoría debe interpretarse con cuidado por el desbalance del conjunto de datos.")

    else:
        st.info(f"Nivel de gravedad estimado: {prediccion}")

    with st.expander("Ver datos ingresados"):
        st.dataframe(datos_entrada, use_container_width=True)

st.divider()

st.subheader("Desempeño del modelo")

st.markdown("""
El modelo final corresponde a un Random Forest ajustado y evaluado sobre el conjunto de prueba. 
Las métricas obtenidas fueron:

- **Accuracy:** 0.7742  
- **Precision:** 0.5108  
- **Recall:** 0.4959  
- **F1 Macro:** 0.4759  

Estos resultados muestran un desempeño general aceptable, aunque el F1 Macro evidencia que todavía existen retos en la clasificación de las categorías menos frecuentes.
""")
