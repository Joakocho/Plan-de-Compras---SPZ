import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# Utilidades
# -----------------------------
def clean_date(val):
    """Convierte a fecha o NaT, ignorando textos tipo OK/COMPRADO/-"""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, str) and val.strip().upper() in {"OK", "COMPRADO", "-", ""}:
        return pd.NaT
    return pd.to_datetime(val, errors="coerce")

def clean_money(val):
    """Convierte valores como '$ 12.345,67' o 'USD 10.000' a número."""
    if pd.isna(val):
        return pd.NA
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val)
    s = s.replace("USD", "").replace("$", "").strip()
    # quitar separador de miles y normalizar decimales
    s = s.replace(".", "").replace(",", ".")
    return pd.to_numeric(s, errors="coerce")

def fmt_money(num):
    if pd.isna(num):
        return "-"
    try:
        return f"${num:,.0f}".replace(",", ".")
    except Exception:
        return str(num)

def buscar_hoja(xls, candidatos):
    """Devuelve el nombre de hoja que contenga alguno de los candidatos."""
    for s in xls.sheet_names:
        sl = s.strip().lower()
        if any(c in sl for c in candidatos):
            return s
    return None

def columnas_por_obra(df_plan, obra_nombre):
    """
    En la hoja 'Plan de Compras' el patrón es:
    ... 'Unnamed: 1' (tareas), luego 'CAPRI', 'Unnamed: 3', 'Unnamed: 4', <precio>,
    luego 'CHALETS', ... etc. Tomamos 3 fechas + 1 precio a partir del header 'obra'.
    """
    cols = list(df_plan.columns)
    if obra_nombre not in cols:
        raise ValueError(f"No se encontró la columna de inicio para la obra '{obra_nombre}'. "
                         f"Verificá el encabezado exacto en el Excel.")

    idx = cols.index(obra_nombre)
    prev_cols = [cols[idx], cols[idx + 1], cols[idx + 2]]           # Fechas OBR, COMPRA, OCE
    precio_col = cols[idx + 3]                                       # Precio Estimado
    return prev_cols, precio_col

def preparar_plan_por_obra(df_plan, obra_nombre):
    # detectar columna de tareas (según tu archivo suele ser 'Unnamed: 1')
    tareas_col = "Unnamed: 1"
    if tareas_col not in df_plan.columns:
        # fallback si cambiaron encabezados
        for cand in ["Tareas", "Tarea", "tareas", "tarea"]:
            if cand in df_plan.columns:
                tareas_col = cand
                break
    if tareas_col not in df_plan.columns:
        raise ValueError("No encuentro la columna de 'Tarea(s)' en la hoja Plan de Compras.")

    prev_cols, precio_col = columnas_por_obra(df_plan, obra_nombre)

    # limpiar fechas previstas
    fechas_prev = df_plan[prev_cols].applymap(clean_date)
    fechas_prev.columns = ["Fecha OBR", "Fecha COMPRA", "Fecha OCE"]

    plan = pd.concat([
        df_plan[tareas_col].rename("Tarea"),
        fechas_prev,
        df_plan[precio_col].rename("Precio Estimado")
    ], axis=1)

    plan = plan.dropna(subset=["Tarea"])
    plan["Precio Estimado"] = plan["Precio Estimado"].apply(clean_money)
    plan["Precio Estimado (fmt)"] = plan["Precio Estimado"].apply(fmt_money)

    # mantener filas que tengan al menos 1 fecha prevista
    plan = plan[plan[["Fecha OBR", "Fecha COMPRA", "Fecha OCE"]].notna().any(axis=1)].reset_index(drop=True)
    return plan

def calcular_kpis(df_plan, df_real, obra):
    df_real_obra = df_real[df_real["Obra"].astype(str).str.strip().str.upper() == obra.upper()].copy()

    # Normalizar tipos de fecha en reales
    for col in ["Real OBR", "Real COMPRA", "Real OCE"]:
        if col in df_real_obra.columns:
            df_real_obra[col] = df_real_obra[col].apply(clean_date)

    tareas = df_plan["Tarea"].dropna().unique()
    total_fases = len(tareas) * 3 if len(tareas) else 0

    en_fecha = 0
    dias_retraso = []

    for _, rr in df_real_obra.iterrows():
        plan_row = df_plan[df_plan["Tarea"].astype(str).str.strip() == str(rr["Tarea"]).strip()]
        if plan_row.empty:
            continue
        for fase in ["OBR", "COMPRA", "OCE"]:
            prev = plan_row[f"Fecha {fase}"].values[0]
            real = rr.get(f"Real {fase}", pd.NaT)
            if pd.notna(prev) and pd.notna(real):
                diff = (real - prev).days
                if diff <= 0:
                    en_fecha += 1
                else:
                    dias_retraso.append(diff)

    pct_en_fecha = (en_fecha / total_fases * 100) if total_fases else 0.0
    prom_retraso = (sum(dias_retraso) / len(dias_retraso)) if dias_retraso else 0.0
    total_estimado = pd.to_numeric(df_plan["Precio Estimado"], errors="coerce").fillna(0).sum()

    return pct_en_fecha, prom_retraso, total_estimado

def plot_gantt(df_plan, df_real, obra, escala="Mensual (grilla semanal)"):
    df_real_obra = df_real[df_real["Obra"].astype(str).str.strip().str.upper() == obra.upper()].copy()
    for c in ["Real OBR", "Real COMPRA", "Real OCE"]:
        if c in df_real_obra.columns:
            df_real_obra[c] = df_real_obra[c].apply(clean_date)

    rows = []
    for _, pr in df_plan.iterrows():
        tarea = pr["Tarea"]
        precio_num = pr["Precio Estimado"]
        precio_fmt = pr["Precio Estimado (fmt)"]
        rrow = df_real_obra[df_real_obra["Tarea"].astype(str).str.strip() == str(tarea).strip()]

        for fase in ["OBR", "COMPRA", "OCE"]:
            prev = pr[f"Fecha {fase}"]
            real = rrow[f"Real {fase}"].values[0] if (not rrow.empty and f"Real {fase}" in rrow.columns) else pd.NaT

            if pd.notna(prev):
                rows.append({
                    "Tarea": tarea,
                    "Fase": f"{fase} (Prevista)",
                    "Fecha": prev,
                    "Tipo": "Prevista",
                    "Precio Estimado": precio_num,
                    "Precio (texto)": precio_fmt,
                    "Diferencia (días)": None
                })
            if pd.notna(real):
                color = "Adelanto/En fecha"
                dif = None
                if pd.notna(prev):
                    dif = (real - prev).days
                    if dif > 0:
                        color = "Retraso"
                rows.append({
                    "Tarea": tarea,
                    "Fase": f"{fase} (Real)",
                    "Fecha": real,
                    "Tipo": color,
                    "Precio Estimado": precio_num,
                    "Precio (texto)": precio_fmt,
                    "Diferencia (días)": dif
                })

    if not rows:
        return px.scatter(title=f"Plan de Compras - {obra} (sin datos para graficar)")

    gantt_df = pd.DataFrame(rows)

    fig = px.scatter(
        gantt_df,
        x="Fecha",
        y="Tarea",
        color="Tipo",
        symbol="Fase",
        hover_data={
            "Precio (texto)": True,
            "Diferencia (días)": True,
            "Fecha": "|%d-%b-%Y",
            "Precio Estimado": False,   # oculto el num crudo
        },
        color_discrete_map={"Prevista": "blue", "Adelanto/En fecha": "green", "Retraso": "red"},
        height=800,
        title=f"Plan de Compras - {obra}"
    )
    fig.update_layout(legend_title="Tipo")

    # ----- eje X con opción de escala -----
    if escala.startswith("Mensual"):
        # Ticks mensuales, grilla menor semanal
        fig.update_xaxes(
            tickformat="%b %Y",
            dtick="M1",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.15)",
            minor=dict(
                dtick=7 * 24 * 60 * 60 * 1000,  # 7 días en ms
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)"
            )
        )
    else:
        # Ticks semanales
        fig.update_xaxes(
            tickformat="%d-%b\n%Y",
            dtick=7 * 24 * 60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.15)"
        )

    return fig

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Plan de Compras", layout="wide")
st.sidebar.title("Navegación")
obra_sel = st.sidebar.radio("Selecciona la obra", ["CAPRI", "CHALETS", "SENECA"])
escala_sel = st.sidebar.selectbox("Escala de tiempo", ["Mensual (grilla semanal)", "Semanal (ticks semanales)"])
archivo_excel = st.sidebar.file_uploader("Subí el archivo Excel", type=["xlsx"])

if not archivo_excel:
    st.warning("📂 Subí un Excel con la hoja de **Plan de Compras** y la hoja **Reales** para comenzar.")
    st.stop()

# Leer Excel detectando hojas aunque cambien los nombres
xls = pd.ExcelFile(archivo_excel)
plan_sheet = buscar_hoja(xls, ["plan de compras", "plan", "compras"])
reales_sheet = buscar_hoja(xls, ["reales", "real"])

if plan_sheet is None:
    plan_sheet = xls.sheet_names[0]
if reales_sheet is None:
    reales_sheet = xls.sheet_names[-1]

df_plan_raw = pd.read_excel(xls, sheet_name=plan_sheet)
df_real = pd.read_excel(xls, sheet_name=reales_sheet)

# limpiar posibles columnas de fecha en plan
for c in df_plan_raw.columns:
    if "Fecha" in str(c):
        df_plan_raw[c] = df_plan_raw[c].apply(clean_date)

# preparar plan para la obra elegida
try:
    plan_form = preparar_plan_por_obra(df_plan_raw, obra_sel)
except Exception as e:
    st.error(f"Problema leyendo el plan para **{obra_sel}**: {e}")
    st.stop()

# KPIs
pct_ok, prom_ret, total_est = calcular_kpis(plan_form, df_real, obra_sel)
c1, c2, c3 = st.columns(3)
c1.metric("Cumplimiento en fecha", f"{pct_ok:.1f}%")
c2.metric("Días promedio de retraso", f"{prom_ret:.1f} días")
c3.metric("Total Estimado", fmt_money(total_est))

# Gantt
fig = plot_gantt(plan_form, df_real, obra_sel, escala=escala_sel)
st.plotly_chart(fig, use_container_width=True)

# Tabla opcional
with st.expander("Ver tabla base (obra seleccionada)"):
    st.dataframe(plan_form)
