import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Alcaldía de Envigado | Predicción de gravedad",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1350px;
}
.card {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 0.8rem;
    border: 1px solid #e5e7eb;
}
.metric-card {
    background-color: #f1f5f9;
    padding: 0.9rem;
    border-radius: 0.7rem;
    text-align: center;
    border: 1px solid #e2e8f0;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
}
.metric-label {
    font-size: 0.85rem;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_accidentalidad_envigado.pkl")

modelo = cargar_modelo()

barrios = [
    "ALCALA", "ALTO DE MISAEL", "BOSQUES DE ZUÑIGA", "BUCAREST",
    "EL CHINGUÍ", "EL CHOCHO", "EL DORADO", "EL ESMERALDAL",
    "EL PORTAL", "EL SALADO", "EL TRIANON", "JARDINES",
    "LA INMACULADA", "LA MAGNOLIA", "LA MINA", "LA PAZ",
    "LA PRADERA", "LA SEBASTIANA", "LAS ANTILLAS", "LAS CASITAS",
    "LAS FLORES", "LAS ORQUIDEAS", "LAS VEGAS",
    "LOMA DE LAS BRUJAS", "LOMA DEL ATRAVEZADO", "LOMA DEL BARRO",
    "LOS NARANJOS", "MESA", "MILAN VALLEJUELOS", "NO REPORTA",
    "OBRERO", "PONTE VEDRA", "PRIMAVERA", "SAN JOSE",
    "SAN MARCOS", "SAN RAFAEL", "URIBE ANGEL",
    "VEREDA EL ESCOBERO", "VEREDA EL VALLANO", "VEREDA LA ESPERANZA",
    "VEREDA PALMAS", "VEREDA PANTANILLO", "VEREDA PERICO",
    "VEREDA SANTA CATALINA", "VILLA GRANDE", "ZONA CENTRO", "ZUÑIGA"
]

clases_accidente = [
    "CHOQUE",
    "ATROPELLO",
    "CAIDA OCUPANTE",
    "VOLCAMIENTO",
    "OTRO"
]

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

st.title("🚗 Alcaldía de Envigado | Predicción de gravedad de accidentes de tránsito")

st.markdown("""
Aplicación desarrollada por **Mario Sergio Gómez Rueda** como prototipo académico de analítica predictiva.
El modelo estima la gravedad de un accidente de tránsito a partir de variables históricas del dataset público
**Accidentalidad Municipio de Envigado**, disponible en Datos Abiertos Colombia.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <b>Fuente del dataset</b><br>
    Datos Abiertos Colombia<br>
    Accidentalidad Municipio de Envigado
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <b>Entidad responsable de la Info</b><br>
    Alcaldía de Envigado<br>
    Información pública de accidentalidad
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <b>Autor</b><br>
    Mario Sergio Gómez Rueda<br>
    Aprendizaje de máquinas
    </div>
    """, unsafe_allow_html=True)

st.divider()

col_form, col_guia = st.columns([2.4, 1])

with col_form:
    st.subheader("Características del accidente")

    with st.form("formulario_prediccion"):

        f1c1, f1c2, f1c3 = st.columns(3)

        with f1c1:
            mes_nombre = st.selectbox("Mes", list(meses.keys()))

        with f1c2:
            dia_nombre = st.selectbox("Día de la semana", list(dias.keys()))

        with f1c3:
            hora = st.selectbox("Hora del accidente", list(range(0, 24)))

        f2c1, f2c2, f2c3 = st.columns(3)

        with f2c1:
            resultado_beodez = st.number_input(
                "Resultado de beodez",
                min_value=0,
                step=1
            )

        with f2c2:
            clase_accidente = st.selectbox(
                "Clase de accidente",
                clases_accidente
            )

        with f2c3:
            barrio = st.selectbox(
                "Barrio",
                barrios
            )

        causa = st.text_input(
            "Causa del accidente",
            value="NO REPORTADA"
        )

        enviar = st.form_submit_button("Predecir gravedad")

with col_guia:
    st.subheader("Guía de uso")

    st.markdown("""
    **Mes:** mes en el que ocurrió el accidente.

    **Día de la semana:** día del evento.

    **Hora:** hora aproximada entre 0 y 23.

    **Resultado de beodez:** valor reportado en la base. Si no existe información, dejar 0.

    **Clase de accidente:** tipo de accidente registrado.

    **Barrio:** barrio tomado del dataset original.

    **Causa:** causa reportada. Si no se conoce, usar “NO REPORTADA”.
    """)

if enviar:

    datos_entrada = pd.DataFrame([{
        "mes": meses[mes_nombre],
        "dia_semana": dias[dia_nombre],
        "hora": hora,
        "resultado_beodez": resultado_beodez,
        "clase_accidente": clase_accidente,
        "causa": causa.upper().strip(),
        "barrio": barrio
    }])

    prediccion = modelo.predict(datos_entrada)[0]

    st.divider()

    col_resultado, col_datos = st.columns([1, 1.6])

    with col_resultado:
        st.subheader("Resultado")

        if prediccion == "SOLO DAÑOS":
            st.success(f"Nivel de gravedad estimado: {prediccion}")
            st.write("El modelo estima una afectación principalmente material.")

        elif prediccion == "HERIDOS":
            st.warning(f"Nivel de gravedad estimado: {prediccion}")
            st.write("El modelo estima que el accidente podría estar asociado con personas lesionadas.")

        elif prediccion == "MUERTOS":
            st.error(f"Nivel de gravedad estimado: {prediccion}")
            st.write("El modelo estima un evento de alta gravedad. Esta categoría debe interpretarse con cuidado por el desbalance del dataset.")

        else:
            st.info(f"Nivel de gravedad estimado: {prediccion}")

    with col_datos:
        st.subheader("Datos usados en la predicción")
        st.dataframe(datos_entrada, use_container_width=True)

st.divider()

st.subheader("Desempeño del modelo")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown('<div class="metric-card"><div class="metric-value">0.7742</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

with m2:
    st.markdown('<div class="metric-card"><div class="metric-value">0.5108</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)

with m3:
    st.markdown('<div class="metric-card"><div class="metric-value">0.4959</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)

with m4:
    st.markdown('<div class="metric-card"><div class="metric-value">0.4759</div><div class="metric-label">F1 Macro</div></div>', unsafe_allow_html=True)

st.markdown("""
El modelo final corresponde a un Random Forest evaluado sobre el conjunto de prueba. Aunque el Accuracy muestra
un desempeño general aceptable, el F1 Macro evidencia retos en la clasificación de categorías menos frecuentes.
""")
