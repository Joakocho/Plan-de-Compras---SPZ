import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------------------------------------------
# Funciones auxiliares
# ---------------------------------------------
def clean_date(val):
    """Convierte el valor a fecha o NaT si no es válido"""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, str) and val.strip().upper() in {"OK", "COMPRADO", "-", ""}:
        return pd.NaT
    return pd.to_datetime(val, errors="coerce")

def calcular_kpis(df_plan, df_real, obra):
    # Filtramos por obra
    tareas = df_plan["Tarea"].dropna().unique()
    df_real_obra = df_real[df_real["Obra"] == obra]

    total_tareas = len(tareas)
    en_fecha = 0
    total_dias_retraso = []

    for _, row in df_real_obra.iterrows():
        plan_row = df_plan[df_plan["Tarea"] == row["Tarea"]]
        if not plan_row.empty:
            for fase in ["OBR", "COMPRA", "OCE"]:
                fecha_prev = plan_row[f"Fecha {fase}"].values[0]
                fecha_real = row[f"Real {fase}"]
                if pd.notna(fecha_prev) and pd.notna(fecha_real):
                    dias_dif = (fecha_real - fecha_prev).days
                    if dias_dif <= 0:
                        en_fecha += 1
                    else:
                        total_dias_retraso.append(dias_dif)

    pct_en_fecha = (en_fecha / (total_tareas * 3)) * 100 if total_tareas > 0 else 0
    dias_prom_retraso = sum(total_dias_retraso) / len(total_dias_retraso) if total_dias_retraso else 0
    total_estimado = df_plan["Precio Estimado"].sum()

    return pct_en_fecha, dias_prom_retraso, total_estimado

def plot_gantt(df_plan, df_real, obra):
    df_real_obra = df_real[df_real["Obra"] == obra]
    gantt_data = []

    for _, row in df_plan.iterrows():
        tarea = row["Tarea"]
        precio = row["Precio Estimado"]
        real_row = df_real_obra[df_real_obra["Tarea"] == tarea]

        for fase in ["OBR", "COMPRA", "OCE"]:
            fecha_prev = row[f"Fecha {fase}"]
            fecha_real = None
            if not real_row.empty:
                fecha_real = real_row[f"Real {fase}"].values[0]

            if pd.notna(fecha_prev):
                gantt_data.append({
                    "Tarea": tarea,
                    "Fase": f"{fase} (Prevista)",
                    "Fecha": fecha_prev,
                    "Tipo": "Prevista",
                    "Precio Estimado": precio,
                    "Diferencia (días)": None
                })
            if pd.notna(fecha_real):
                dias_dif = None
                color_tipo = "Adelanto/En fecha"
                if pd.notna(fecha_prev):
                    dias_dif = (fecha_real - fecha_prev).days
                    if dias_dif > 0:
                        color_tipo = "Retraso"
                gantt_data.append({
                    "Tarea": tarea,
                    "Fase": f"{fase} (Real)",
                    "Fecha": fecha_real,
                    "Tipo": color_tipo,
                    "Precio Estimado": precio,
                    "Diferencia (días)": dias_dif
                })

    gantt_df = pd.DataFrame(gantt_data)

    fig = px.scatter(
        gantt_df,
        x="Fecha",
        y="Tarea",
        color="Tipo",
        symbol="Fase",
        hover_data=["Precio Estimado", "Diferencia (días)"],
        color_discrete_map={
            "Prevista": "blue",
            "Adelanto/En fecha": "green",
            "Retraso": "red"
        }
    )

    fig.update_layout(
        title=f"Plan de Compras - {obra}",
        xaxis_title="Fecha",
        yaxis_title="Tarea",
        legend_title="Tipo",
        height=800
    )
    return fig

# ---------------------------------------------
# Interfaz principal
# ---------------------------------------------
st.set_page_config(page_title="Plan de Compras", layout="wide")

st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Selecciona la obra", ["CAPRI", "CHALETS", "SENECA"])

archivo_excel = st.sidebar.file_uploader("Sube el archivo Excel", type=["xlsx"])

if archivo_excel:
    # Leer datos
    df_plan = pd.read_excel(archivo_excel, sheet_name="Plan de Compras")
    df_real = pd.read_excel(archivo_excel, sheet_name="Reales")

    # Limpiar fechas en plan
    for col in df_plan.columns:
        if "Fecha" in str(col):
            df_plan[col] = df_plan[col].apply(clean_date)

    # Limpiar fechas en reales
    for col in df_real.columns:
        if "Real" in str(col):
            df_real[col] = df_real[col].apply(clean_date)

    # Identificar columnas por obra
    idx = list(df_plan.columns).index(pagina)
    prev_cols = [df_plan.columns[idx], df_plan.columns[idx+1], df_plan.columns[idx+2]]
    precio_col = df_plan.columns[idx+3]

    # Crear dataframe formateado
    fechas_prev = df_plan[prev_cols].applymap(clean_date)
    fechas_prev.columns = ["Fecha OBR", "Fecha COMPRA", "Fecha OCE"]
    plan_formateado = pd.concat([
        df_plan["Unnamed: 1"].rename("Tarea"),
        fechas_prev,
        df_plan[precio_col].rename("Precio Estimado")
    ], axis=1)
    plan_formateado = plan_formateado.dropna(subset=["Tarea"])

    # KPIs
    pct, prom_dias, total_est = calcular_kpis(plan_formateado, df_real, pagina)
    col1, col2, col3 = st.columns(3)
    col1.metric("Cumplimiento en fecha", f"{pct:.1f}%")
    col2.metric("Días promedio retraso", f"{prom_dias:.1f} días")
    col3.metric("Total Estimado", f"${total_est:,.0f}")

    # Gantt
    fig = plot_gantt(plan_formateado, df_real, pagina)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("📂 Sube un archivo Excel para comenzar.")
