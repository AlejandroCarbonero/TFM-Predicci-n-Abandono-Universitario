import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------------
# 1️⃣ Cargar modelo predictivo que hemos obtenido de nuestro notebook 
# ---------------------------
modelo = joblib.load("modelo_logistic.pkl")

# ---------------------------
# 2️⃣ Título - Establecemos un título para la app
# ---------------------------
st.set_page_config(page_title="Predicción de Abandono Académico", layout="wide")
st.title("Predicción de Abandono Académico")
st.markdown("Introduce los datos del estudiante en la barra lateral para obtener la predicción.")

# ---------------------------
# 3️⃣ Inputs en barra lateral - Establecemos un titulo para el lugar donde se introducirán los datos del estudiante
# ---------------------------
st.sidebar.header("📋 Información del estudiante")

# Columnas numéricas
age = st.sidebar.number_input("Edad al matricularse", min_value=15, max_value=80, value=20)

# 1er Semestre
curr_1_credited = st.sidebar.number_input("1er Semestre: Unidades curriculares acreditadas", value=0)
curr_1_enrolled = st.sidebar.number_input("1er Semestre: Unidades curriculares inscritas", value=0)
curr_1_evaluations = st.sidebar.number_input("1er Semestre: Unidades curriculares evaluadas", value=0)
curr_1_approved = st.sidebar.number_input("1er Semestre: Unidades curriculares aprobadas", value=0)
curr_1_grade = st.sidebar.number_input("1er Semestre: Nota promedio de unidades curriculares", min_value=0.0, max_value=20.0, value=10.0)
curr_1_without_eval = st.sidebar.number_input("1er Semestre: Unidades curriculares sin evaluación", value=0)

# 2do Semestre
curr_2_credited = st.sidebar.number_input("2do Semestre: Unidades curriculares acreditadas", value=0)
curr_2_enrolled = st.sidebar.number_input("2do Semestre: Unidades curriculares inscritas", value=0)
curr_2_evaluations = st.sidebar.number_input("2do Semestre: Unidades curriculares evaluadas", value=0)
curr_2_approved = st.sidebar.number_input("2do Semestre: Unidades curriculares aprobadas", value=0)
curr_2_grade = st.sidebar.number_input("2do Semestre: Nota promedio de unidades curriculares", min_value=0.0, max_value=20.0, value=10.0)
curr_2_without_eval = st.sidebar.number_input("2do Semestre: Unidades curriculares sin evaluación", value=0)

# columnas booleanas / dummies
debtor = st.sidebar.selectbox("Deudor", ["No", "Sí"])
tuition_up_to_date = st.sidebar.selectbox("Matrícula al día", ["No", "Sí"])
gender = st.sidebar.selectbox("Género", ["Masculino", "Femenino"])
scholarship = st.sidebar.selectbox("Beca", ["No", "Sí"])

# ---------------------------
# 4️⃣ Creamos DataFrame con inputs (dummies correctas)
# ---------------------------
input_dict = {
    "Age at enrollment": age,
    "Curricular units 1st sem (credited)": curr_1_credited,
    "Curricular units 1st sem (enrolled)": curr_1_enrolled,
    "Curricular units 1st sem (evaluations)": curr_1_evaluations,
    "Curricular units 1st sem (approved)": curr_1_approved,
    "Curricular units 1st sem (grade)": curr_1_grade,
    "Curricular units 1st sem (without evaluations)": curr_1_without_eval,
    "Curricular units 2nd sem (credited)": curr_2_credited,
    "Curricular units 2nd sem (enrolled)": curr_2_enrolled,
    "Curricular units 2nd sem (evaluations)": curr_2_evaluations,
    "Curricular units 2nd sem (approved)": curr_2_approved,
    "Curricular units 2nd sem (grade)": curr_2_grade,
    "Curricular units 2nd sem (without evaluations)": curr_2_without_eval,
    "Debtor_yes": debtor == "Sí",
    "Tuition fees up to date_yes": tuition_up_to_date == "Sí",
    "Gender_male": gender == "Masculino",
    "Scholarship holder_yes": scholarship == "Sí"
}

input_df = pd.DataFrame([input_dict])

# ---------------------------
# 5️⃣ Predecir - Predecimos el modelo
# ---------------------------
pred = modelo.predict(input_df)
prob = modelo.predict_proba(input_df)

# Probabilidad correcta de abandono
dropout_index = list(modelo.classes_).index("dropout")
proba_dropout = prob[0][dropout_index]

# ---------------------------
# 6️⃣ Resultados y Dashboard
# ---------------------------
tab1, tab2 = st.tabs(["Resultado de Predicción", "Dashboard"])

with tab1:
    st.subheader("Resultado de Predicción")

    # Probabilidad y Semáforo
    col1, col2 = st.columns(2)

    # Mostrar predicción
    pred_text = "Abandono" if pred[0] == "dropout" else "No abandono"
    col1.metric("Predicción", pred_text)

    # Semáforo de riesgo
    if proba_dropout < 0.3:
        riesgo = "🟢 Riesgo bajo (<30%)"
    elif proba_dropout < 0.6:
        riesgo = "🟡 Riesgo medio (30-60%)"
    else:
        riesgo = "🔴 Riesgo alto (>60%)"
    col2.metric("Probabilidad de Abandono", f"{proba_dropout:.2%}", delta=riesgo)

    # Barra de progreso visual
    st.progress(float(proba_dropout))

    # Factores más influyentes
    coef = modelo.coef_[0]  
    feature_names = input_df.columns
    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Coeficiente": coef,
        "Valor del estudiante": [input_df[f].iloc[0] for f in feature_names]
    })
    impact_df["Impacto"] = impact_df["Coeficiente"] * impact_df["Valor del estudiante"]
    impact_df = impact_df.sort_values(by="Impacto", key=abs, ascending=False).head(5)

    with st.expander("🔍 Principales factores que influyen en la predicción"):
        st.table(impact_df.rename(columns={
            "Feature": "Variable",
            "Valor del estudiante": "Valor del estudiante",
            "Impacto": "Impacto"
        })[["Variable", "Valor del estudiante", "Impacto"]])

    # Gráfico de probabilidades por clase
    st.subheader("Distribución de Probabilidades por Clase")

    labels_map = {
        "dropout": "Abandono",
        "no_dropout": "No abandono",
        "graduate": "Graduado"
    }

    clases_traducidas = [labels_map.get(c, c) for c in modelo.classes_]

    fig = px.bar(
        x=clases_traducidas,
        y=prob[0],
        labels={"x": "Clase", "y": "Probabilidad"},
        color=clases_traducidas,
        color_discrete_sequence=["red", "green", "blue"]
    )
    fig.update_layout(title="Distribución de Probabilidades por Clase", yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True, key="probabilidades_clase")
with tab2:
    st.subheader("Dashboard de Análisis de Estudiantes")

    @st.cache_data
    def load_data():
        df = pd.read_csv("df_students.csv")
        df["Debtor"] = df["Debtor"].astype(str).str.strip().str.lower()
        df["Tuition fees up to date"] = df["Tuition fees up to date"].astype(str).str.strip().str.lower()
        df["Scholarship holder"] = df["Scholarship holder"].astype(str).str.strip().str.lower()
        df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
        return df
    
    df_dashboard = load_data()

    # Filtros dinámicos
    st.markdown("### 🔎 Filtros dinámicos")
    col1, col2, col3 = st.columns(3)
    selected_gender = col1.multiselect(
        "Selecciona Género", 
        options=df_dashboard["Gender"].unique(), 
        default=df_dashboard["Gender"].unique()
    )
    selected_scholarship = col2.multiselect(
        "Selecciona Beca", 
        options=df_dashboard["Scholarship holder"].unique(), 
        default=df_dashboard["Scholarship holder"].unique()
    )
    selected_debtor = col3.multiselect(
        "Selecciona Deudor", 
        options=df_dashboard["Debtor"].unique(), 
        default=df_dashboard["Debtor"].unique()
    )

    df_filtered = df_dashboard[
        (df_dashboard["Gender"].isin(selected_gender)) &
        (df_dashboard["Scholarship holder"].isin(selected_scholarship)) &
        (df_dashboard["Debtor"].isin(selected_debtor))
    ]

    # KPIs principales
    st.markdown("### KPIs")
    kpi_col1, kpi_col2 = st.columns(2)

    total_students = len(df_filtered)
    avg_age = df_filtered["Age at enrollment"].mean() if total_students > 0 else 0

    kpi_col1.metric("Total Estudiantes", total_students)
    kpi_col2.metric("Edad Promedio", f"{avg_age:.1f}")

    # Gráficos interactivos
    st.markdown("### Distribuciones")

    # Deudores por Género
    fig1 = px.histogram(
        df_filtered,
        x="Gender",
        color="Debtor",
        barmode="group",
        title="Distribución de Deudores por Género"
    )
    st.plotly_chart(fig1, use_container_width=True, key="deudores_genero")

    # Matrícula al día vs Beca
    fig2 = px.histogram(
        df_filtered,
        x="Scholarship holder",
        color="Tuition fees up to date",
        barmode="group",
        title="Relación Beca y Matrícula al Día"
    )
    st.plotly_chart(fig2, use_container_width=True, key="beca_matricula")

    # Edad promedio por Género
    avg_age_df = df_filtered.groupby("Gender", as_index=False)["Age at enrollment"].mean()
    fig3 = px.bar(
        avg_age_df,
        x="Gender",
        y="Age at enrollment",
        color="Gender",
        text="Age at enrollment",
        title="Edad Promedio por Género"
    )
    fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig3, use_container_width=True, key="edad_genero")

    # Histograma de Notas 1er Semestre
    fig4 = px.histogram(
        df_filtered, 
        x="Curricular units 1st sem (grade)", 
        nbins=20, 
        color="Gender", 
        barmode="overlay", 
        title="Distribución de Notas 1er Semestre por Género"
    )
    st.plotly_chart(fig4, use_container_width=True, key="notas_1sem")

    # Boxplot de Notas 2do Semestre
    fig5 = px.box(
        df_filtered, 
        x="Scholarship holder", 
        y="Curricular units 2nd sem (grade)", 
        color="Scholarship holder", 
        points="all", 
        title="Notas 2do Semestre según Beca"
    )
    st.plotly_chart(fig5, use_container_width=True, key="notas_2sem")