# Plan de Compras - Visualización Interactiva

Esta es una aplicación en Streamlit que muestra el Plan de Compras previsto vs real por obra, con KPIs y un Gantt interactivo.

## 🚀 Cómo usar
1. Crea un repositorio en GitHub y sube estos archivos (`app.py`, `requirements.txt`, `README.md`, y la plantilla Excel si quieres incluirla como ejemplo).
2. Ve a [Streamlit Cloud](https://share.streamlit.io/) e inicia sesión.
3. Conecta tu cuenta de GitHub y selecciona este repositorio.
4. Publica la app.
5. Abre el link que se genere.

## 📂 Estructura del Excel
El archivo debe tener dos hojas:
- **Plan de Compras**: Fechas previstas y precios estimados.
- **Reales**: Fechas reales para cada obra y tarea.

Ejemplo de `Reales`:
| Obra   | Tarea       | Real OBR  | Real COMPRA | Real OCE  |
|--------|-------------|-----------|-------------|-----------|
| CAPRI  | premarcos   | 2025-07-01| 2025-07-10  | 2025-07-25|
| CHALETS| ventanas    | 2025-08-05| 2025-08-15  | 2025-08-30|

## 📦 Dependencias
Se instalan automáticamente en Streamlit Cloud desde `requirements.txt`.
