import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- import opcional para clicks en el gráfico (expand/collapse) ---
try:
    from streamlit_plotly_events import plotly_events
    _HAS_PLOTLY_EVENTS = True
except Exception:
    _HAS_PLOTLY_EVENTS = False

# ============================
# Utilidades
# ============================
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
    if "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    if "Unnamed: 1" in df.columns:
        return "Unnamed: 1"
    return None

def preparar_plan_por_obra(df_plan, obra_nombre):
    rubro_col = detectar_columna(df_plan, ["Rubro", "rubro", "RUBRO", "Unnamed: 0"])
    tareas_col = detectar_columna(df_plan, ["Tareas", "Tarea", "tareas", "tarea", "Unnamed: 1"])
    if tareas_col is None:
        raise ValueError("No encuentro la columna de 'Tarea(s)' en la hoja Plan de Compras.")
    if rubro_col is None:
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

def construir_rows(plan_form, df_real, obra):
    df_real_obra = df_real[df_real["Obra"].astype(str).str.strip().str.upper() == obra.upper()].copy()
    for c in ["Real OBR", "Real COMPRA", "Real OCE"]:
        if c in df_real_obra.columns:
            df_real_obra[c] = df_real_obra[c].apply(clean_date)

    rows = []
    for _, pr in plan_form.iterrows():
        rubro = pr["Rubro"]; tarea = pr["Tarea"]
        precio_num = pr["Precio Estimado"]; precio_fmt = pr["Precio Estimado (fmt)"]
        rrow = df_real_obra[df_real_obra["Tarea"].astype(str).str.strip() == str(tarea).strip()]

        for fase in ["OBR", "COMPRA", "OCE"]:
            prev = pr[f"Fecha {fase}"]
            real = rrow[f"Real {fase}"].values[0] if (not rrow.empty and f"Real {fase}" in rrow.columns) else pd.NaT
            if pd.notna(prev):
                rows.append({"Rubro": rubro, "Tarea": tarea, "Fase": f"{fase} (Prevista)",
                             "Fecha": prev, "Tipo": "Prevista",
                             "Precio Estimado": precio_num, "Precio (texto)": precio_fmt,
                             "Diferencia (días)": None})
            if pd.notna(real):
                color = "Adelanto/En fecha"; dif = None
                if pd.notna(prev):
                    dif = (real - prev).days
                    if dif > 0: color = "Retraso"
                rows.append({"Rubro": rubro, "Tarea": tarea, "Fase": f"{fase} (Real)",
                             "Fecha": real, "Tipo": color,
                             "Precio Estimado": precio_num, "Precio (texto)": precio_fmt,
                             "Diferencia (días)": dif})
    return pd.DataFrame(rows)

def expand_inline(gantt_df, selected_rubro):
    indent = "  • "
    y_labels = []
    for r in sorted(gantt_df["Rubro"].dropna().unique()):
        y_labels.append(r)
        if selected_rubro and r == selected_rubro:
            tareas = sorted(gantt_df[gantt_df["Rubro"] == r]["Tarea"].unique())
            y_labels.extend([indent + t for t in tareas])

    def map_y(row):
        if selected_rubro and row["Rubro"] == selected_rubro:
            return indent + row["Tarea"]
        else:
            return row["Rubro"]

    df2 = gantt_df.copy()
    df2["Y"] = df2.apply(map_y, axis=1)
    df2["Y"] = pd.Categorical(df2["Y"], categories=y_labels, ordered=True)
    return df2, y_labels

def apply_dark_theme(fig):
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#e5e5e5",
        legend_title_font_color="#e5e5e5"
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.15)")

def make_fig(df_for_plot, title, vista="Mensual", grilla_semanal=True):
    fig = px.scatter(
        df_for_plot,
        x="Fecha",
        y="Y",
        color="Tipo",
        symbol="Fase",
        hover_data={
            "Rubro": True, "Tarea": True,
            "Precio (texto)": True, "Diferencia (días)": True,
            "Fecha": "|%d-%b-%Y",
            "Y": False, "Precio Estimado": False
        },
        color_discrete_map={"Prevista": "blue", "Adelanto/En fecha": "green", "Retraso": "red"},
        height=800,
        title=title
    )
    if vista == "Mensual":
        fig.update_xaxes(ticklabelmode="period", dtick="M1", tickformat="%b %Y", showgrid=False)
        add_month_grid(fig, df_for_plot["Fecha"], add_week_subdiv=grilla_semanal)
    else:
        fig.update_xaxes(tickformat="%d-%b\n%Y", dtick=7*24*60*60*1000, showgrid=True,
                         gridcolor="rgba(255,255,255,0.15)")
    apply_dark_theme(fig)
    return fig

# ============================
# App
# ============================
st.set_page_config(page_title="Plan de Compras", layout="wide")
st.sidebar.title("Navegación")
obra_sel = st.sidebar.radio("Selecciona la obra", ["CAPRI", "CHALETS", "SENECA"])
vista_sel = st.sidebar.selectbox("Escala de tiempo", ["Mensual", "Semanal"])
grilla_week = st.sidebar.checkbox("Dividir cada mes en 4 'semanas' (1/8/15/22)", value=True)

archivo_excel = st.sidebar.file_uploader("Subí el archivo Excel", type=["xlsx"])
if not archivo_excel:
    st.warning("📂 Subí un Excel con la hoja de **Plan de Compras** y la hoja **Reales** para comenzar.")
    st.stop()

xls = pd.ExcelFile(archivo_excel)
plan_sheet = buscar_hoja(xls, ["plan de compras", "plan", "compras"]) or xls.sheet_names[0]
reales_sheet = buscar_hoja(xls, ["reales", "real"]) or xls.sheet_names[-1]

df_plan_raw = pd.read_excel(xls, sheet_name=plan_sheet)
df_real = pd.read_excel(xls, sheet_name=reales_sheet)

for c in df_plan_raw.columns:
    if "Fecha" in str(c):
        df_plan_raw[c] = df_plan_raw[c].apply(clean_date)

try:
    plan_form = preparar_plan_por_obra(df_plan_raw, obra_sel)
except Exception as e:
    st.error(f"Problema leyendo el plan para **{obra_sel}**: {e}")
    st.stop()

# KPIs globales
pct_ok, prom_ret, total_est = calcular_kpis(plan_form, df_real, obra_sel)
c1, c2, c3 = st.columns(3)
c1.metric("Cumplimiento en fecha", f"{pct_ok:.1f}%")
c2.metric("Días promedio de retraso", f"{prom_ret:.1f} días")
c3.metric("Total Estimado", fmt_money(total_est))

# Datos de puntos (Prev/Real)
gantt_all = construir_rows(plan_form, df_real, obra_sel)

# Estado de expansión
if "rubro_expandido" not in st.session_state:
    st.session_state.rubro_expandido = None

# DF para graficar (colapsado/expandido)
df_plot, _ = expand_inline(gantt_all, st.session_state.rubro_expandido)

# Figura
fig = make_fig(df_plot, f"Plan de Compras - {obra_sel}", vista=vista_sel, grilla_semanal=grilla_week)

# Eventos de click para expandir/colapsar
if _HAS_PLOTLY_EVENTS:
    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                           override_height=800, override_width="100%")
else:
    st.info("Para expandir por rubro instala 'streamlit-plotly-events' y reinicia la app.")
    events = []

if events:
    y_clicked = str(events[0].get("y", ""))
    if y_clicked.startswith("  • "):
        tarea = y_clicked.replace("  • ", "", 1)
        match = gantt_all[gantt_all["Tarea"] == tarea]
        rubro = match["Rubro"].iloc[0] if not match.empty else None
    else:
        rubro = y_clicked

    if rubro:
        st.session_state.rubro_expandido = None if st.session_state.rubro_expandido == rubro else rubro
        st.experimental_rerun()

# Botón para colapsar
if st.session_state.rubro_expandido:
    st.info(f"Rubro expandido: **{st.session_state.rubro_expandido}**")
    if st.button("Colapsar rubro"):
        st.session_state.rubro_expandido = None
        st.experimental_rerun()

# Tabla base
with st.expander("Ver tabla base (obra seleccionada)"):
    st.dataframe(plan_form)
