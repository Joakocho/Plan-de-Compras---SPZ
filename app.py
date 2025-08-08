import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# Utilidades
# -----------------------------
def clean_date(val):
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, str) and val.strip().upper() in {"OK", "COMPRADO", "-", ""}:
        return pd.NaT
    return pd.to_datetime(val, errors="coerce")

def clean_money(val):
    if pd.isna(val):
        return pd.NA
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val)
    s = s.replace("USD", "").replace("$", "").strip()
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
    for s in xls.sheet_names:
        sl = s.strip().lower()
        if any(c in sl for c in candidatos):
            return s
    return None

def columnas_por_obra(df_plan, obra_nombre):
    cols = list(df_plan.columns)
    if obra_nombre not in cols:
        raise ValueError(f"No se encontró la columna de inicio para la obra '{obra_nombre}'.")
    idx = cols.index(obra_nombre)
    prev_cols = [cols[idx], cols[idx + 1], cols[idx + 2]]
    precio_col = cols[idx + 3]
    return prev_cols, precio_col

def detectar_columna(df, candidatos):
    for c in candidatos:
        if c in df.columns:
            return c
    # fallback por posición (primera/segunda col)
    if len(df.columns) >= 1 and "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    if len(df.columns) >= 2 and "Unnamed: 1" in df.columns and "Tarea" not in df.columns:
        return "Unnamed: 1"
    return None

def preparar_plan_por_obra(df_plan, obra_nombre):
    # detectar columnas de Rubro y Tarea
    rubro_col = detectar_columna(df_plan, ["Rubro", "rubro", "RUBRO", "Unnamed: 0"])
    tareas_col = detectar_columna(df_plan, ["Tareas", "Tarea", "tareas", "tarea", "Unnamed: 1"])

    if tareas_col is None:
        raise ValueError("No encuentro la columna de 'Tarea(s)' en la hoja Plan de Compras.")
    if rubro_col is None:
        # si no encontramos rubro, creamos uno genérico
        rubro_col = "__Rubro__"
        df_plan[rubro_col] = "Sin rubro"

    prev_cols, precio_col = columnas_por_obra(df_plan, obra_nombre)
    fechas_prev = df_plan[prev_cols].applymap(clean_date)
    fechas_prev.columns = ["Fecha OBR", "Fecha COMPRA", "Fecha OCE"]

    plan = pd.concat([
        df_plan[rubro_col].rename("Rubro"),
        df_plan[tareas_col].rename("Tarea"),
        fechas_prev,
        df_plan[precio_col].rename("Precio Estimado")
    ], axis=1)

    plan = plan.dropna(subset=["Tarea"])
    plan["Rubro"] = plan["Rubro"].fillna("Sin rubro").astype(str).str.strip()
    plan["Tarea"] = plan["Tarea"].astype(str).str.strip()

    plan["Precio Estimado"] = plan["Precio Estimado"].apply(clean_money)
    plan["Precio Estimado (fmt)"] = plan["Precio Estimado"].apply(fmt_money)

    # mantener filas con al menos 1 fecha prevista
    plan = plan[plan[["Fecha OBR", "Fecha COMPRA", "Fecha OCE"]].notna().any(axis=1)].reset_index(drop=True)
    return plan

def calcular_kpis(df_plan, df_real, obra):
    df_real_obra = df_real[df_real["Obra"].astype(str).str.strip().str.upper() == obra.upper()].copy()
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

# --- grilla mensual con subdivisiones 1/8/15/22 ---
def add_month_grid(fig, dates, add_week_subdiv=True):
    if dates.dropna().empty:
        return
    start = pd.to_datetime(dates.min()).to_period("M").start_time
    end = (pd.to_datetime(dates.max()).to_period("M").end_time + pd.Timedelta(days=1))
    month_starts = pd.date_range(start=start, end=end, freq="MS")

    for m0 in month_starts:
        fig.add_vline(x=m0, line_width=1, line_color="rgba(255,255,255,0.18)")

    if add_week_subdiv:
        for i in range(len(month_starts) - 1):
            m0 = month_starts[i]
            m1 = month_starts[i + 1]
            for d in (7, 14, 21):
                x = m0 + pd.Timedelta(days=d)
                if x < m1:
                    fig.add_vline(x=x, line_width=1, line_color="rgba(255,255,255,0.08)")

def construir_rows(df_plan, df_real, obra):
    df_real_obra = df_real[df_real["Obra"].astype(str).str.strip().str.upper() == obra.upper()].copy()
    for c in ["Real OBR", "Real COMPRA", "Real OCE"]:
        if c in df_real_obra.columns:
            df_real_obra[c] = df_real_obra[c].apply(clean_date)

    rows = []
    for _, pr in df_plan.iterrows():
        rubro = pr["Rubro"]
        tarea = pr["Tarea"]
        precio_num = pr["Precio Estimado"]
        precio_fmt = pr["Precio Estimado (fmt)"]
        rrow = df_real_obra[df_real_obra["Tarea"].astype(str).str.strip() == str(tarea).strip()]

        for fase in ["OBR", "COMPRA", "OCE"]:
            prev = pr[f"Fecha {fase}"]
            real = rrow[f"Real {fase}"].values[0] if (not rrow.empty and f"Real {fase}" in rrow.columns) else pd.NaT

            if pd.notna(prev):
                rows.append({
                    "Rubro": rubro, "Tarea": tarea,
                    "Fase": f"{fase} (Prevista)", "Fecha": prev,
                    "Tipo": "Prevista",
                    "Precio Estimado": precio_num, "Precio (texto)": precio_fmt,
                    "Diferencia (días)": None
                })
            if pd.notna(real):
                color = "Adelanto/En fecha"; dif = None
                if pd.notna(prev):
                    dif = (real - prev).days
                    if dif > 0: color = "Retraso"
                rows.append({
                    "Rubro": rubro, "Tarea": tarea,
                    "Fase": f"{fase} (Real)", "Fecha": real,
                    "Tipo": color,
                    "Precio Estimado": precio_num, "Precio (texto)": precio_fmt,
                    "Diferencia (días)": dif
                })
    return pd.DataFrame(rows)

def plot_gantt(gantt_df, titulo, y_col, vista="Mensual", grilla_semanal=True):
    fig = px.scatter(
        gantt_df,
        x="Fecha",
        y=y_col,                  # <-- puede ser "Rubro" o "Tarea"
        color="Tipo",
        symbol="Fase",
        hover_data={
            "Rubro": True,
            "Tarea": True,
            "Precio (texto)": True,
            "Diferencia (días)": True,
            "Fecha": "|%d-%b-%Y",
            "Precio Estimado": False,
        },
        color_discrete_map={"Prevista": "blue", "Adelanto/En fecha": "green", "Retraso": "red"},
        height=800,
        title=titulo
    )
    fig.update_layout(legend_title="Tipo")

    if vista == "Mensual":
        fig.update_xaxes(
            ticklabelmode="period", dtick="M1", tickformat="%b %Y",
            showgrid=False
        )
        add_month_grid(fig, gantt_df["Fecha"], add_week_subdiv=grilla_semanal)
    else:
        fig.update_xaxes(
            tickformat="%d-%b\n%Y",
            dtick=7 * 24 * 60 * 60 * 1000,
            showgrid=True, gridcolor="rgba(255,255,255,0.15)"
        )

    return fig

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Plan de Compras", layout="wide")
st.sidebar.title("Navegación")
obra_sel = st.sidebar.radio("Selecciona la obra", ["CAPRI", "CHALETS", "SENECA"])

# Controles de tiempo
vista_sel = st.sidebar.selectbox("Escala de tiempo", ["Mensual", "Semanal"])
grilla_week = st.sidebar.checkbox("Dividir cada mes en 4 'semanas' (1/8/15/22)", value=True)

# Nuevo: nivel de detalle
nivel = st.sidebar.radio("Nivel de detalle", ["Rubro (colapsado)", "Tareas de un rubro (expandido)"])

archivo_excel = st.sidebar.file_uploader("Subí el archivo Excel", type=["xlsx"])
if not archivo_excel:
    st.warning("📂 Subí un Excel con la hoja de **Plan de Compras** y la hoja **Reales** para comenzar.")
    st.stop()

# Leer Excel detectando hojas
xls = pd.ExcelFile(archivo_excel)
plan_sheet = buscar_hoja(xls, ["plan de compras", "plan", "compras"]) or xls.sheet_names[0]
reales_sheet = buscar_hoja(xls, ["reales", "real"]) or xls.sheet_names[-1]

df_plan_raw = pd.read_excel(xls, sheet_name=plan_sheet)
df_real = pd.read_excel(xls, sheet_name=reales_sheet)

# Limpiar posibles columnas de fecha en plan
for c in df_plan_raw.columns:
    if "Fecha" in str(c):
        df_plan_raw[c] = df_plan_raw[c].apply(clean_date)

# Preparar plan
try:
    plan_form = preparar_plan_por_obra(df_plan_raw, obra_sel)
except Exception as e:
    st.error(f"Problema leyendo el plan para **{obra_sel}**: {e}")
    st.stop()

# KPIs globales de la obra
pct_ok, prom_ret, total_est = calcular_kpis(plan_form, df_real, obra_sel)
c1, c2, c3 = st.columns(3)
c1.metric("Cumplimiento en fecha", f"{pct_ok:.1f}%")
c2.metric("Días promedio de retraso", f"{prom_ret:.1f} días")
c3.metric("Total Estimado", fmt_money(total_est))

# Construir puntos Previsto/Real
gantt_all = construir_rows(plan_form, df_real, obra_sel)

# Vista Rubro o Tarea por rubro
if nivel.startswith("Rubro"):
    # y = Rubro (colapsado). Mostramos todos los puntos de sus tareas en la fila del rubro.
    fig = plot_gantt(gantt_all, f"Plan de Compras - {obra_sel} (vista por Rubro)", y_col="Rubro",
                     vista=vista_sel, grilla_semanal=grilla_week)
    st.plotly_chart(fig, use_container_width=True)

    # Ayuda para expandir
    with st.expander("Ver tareas por rubro"):
        rubro_pick = st.selectbox("Elegí un rubro para expandir", sorted(plan_form["Rubro"].unique()))
        detalle = gantt_all[gantt_all["Rubro"] == rubro_pick].copy()
        if detalle.empty:
            st.info("No hay datos para ese rubro.")
        else:
            fig2 = plot_gantt(detalle, f"{obra_sel} → {rubro_pick} (tareas)", y_col="Tarea",
                              vista=vista_sel, grilla_semanal=grilla_week)
            st.plotly_chart(fig2, use_container_width=True)

else:
    # elegido: Tareas de un rubro (expandido)
    rubro_pick = st.selectbox("Elegí un rubro", sorted(plan_form["Rubro"].unique()))
    detalle = gantt_all[gantt_all["Rubro"] == rubro_pick].copy()
    if detalle.empty:
        st.info("No hay datos para ese rubro.")
    else:
        fig = plot_gantt(detalle, f"Plan de Compras - {obra_sel} → {rubro_pick}", y_col="Tarea",
                         vista=vista_sel, grilla_semanal=grilla_week)
        st.plotly_chart(fig, use_container_width=True)

# Tabla opcional
with st.expander("Ver tabla base (obra seleccionada)"):
    st.dataframe(plan_form)
