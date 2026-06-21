"""
Dashboard de calidad del aire de la CDMX (RAMA/SIMAT, 2024-2026)
================================================================
Proyecto de Analítica y Visualización de Datos.

Gráficas interactivas con Altair (incluido en Streamlit). Para ejecutarlo
localmente: coloca este archivo junto a la carpeta `datos/procesados/` y corre
`streamlit run app.py`. En la nube lee `datos/procesados/*.csv` del repo.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2 as chi2_dist
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
DATA_DIR_DEFAULT = "datos/procesados"

POLLUTANTS = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "PMCO", "SO2"]
UNITS = {"CO": "ppm", "NO": "ppb", "NO2": "ppb", "NOX": "ppb", "O3": "ppb",
         "PM10": "µg/m³", "PM2.5": "µg/m³", "PMCO": "µg/m³", "SO2": "ppb"}

ZONE_NAMES = {"NE": "Nororiente", "NO": "Noroeste", "CE": "Centro",
              "SO": "Surponiente", "SE": "Sureste"}
ZONE_COLORS = {"NE": "#EF6C3A", "NO": "#D69A2D", "CE": "#6366F1",
               "SO": "#10B981", "SE": "#38BDF8"}

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASON_ORDER = ["Seca-fría", "Seca-caliente", "Lluvias"]
CAT_ORDER = ["Buena", "Aceptable", "Mala", "Muy mala"]
CAT_COLORS = ["#22C55E", "#FACC15", "#F97316", "#DC2626"]

# Paleta para las series de filtros y acentos (legible en claro y oscuro)
C_CRUDA = "#9CA3AF"
C_MEDIA = "#3B82F6"
C_MEDIANA = "#EF4444"
C_EXPO = "#22C55E"
C_ACCENT = "#3B82F6"


# --------------------------------------------------------------------------
# Funciones de cálculo (química / estadística del proyecto)
# --------------------------------------------------------------------------
def temporada(mes: int) -> str:
    if mes in (11, 12, 1, 2):
        return "Seca-fría"
    if mes in (3, 4, 5):
        return "Seca-caliente"
    return "Lluvias"


def categoria_o3(valor: float):
    if pd.isna(valor):
        return np.nan
    if valor <= 58:
        return "Buena"
    if valor <= 90:
        return "Aceptable"
    if valor <= 135:
        return "Mala"
    return "Muy mala"


def tabla_contingencia(df_o3_estacion: pd.DataFrame) -> pd.DataFrame:
    cat = df_o3_estacion["maximo"].apply(categoria_o3)
    temp = df_o3_estacion["dia"].dt.month.map(temporada)
    tab = pd.crosstab(cat, temp)
    filas = [c for c in CAT_ORDER if c in tab.index]
    cols = [s for s in SEASON_ORDER if s in tab.columns]
    return tab.reindex(index=filas, columns=cols).fillna(0).astype(int)


def chi_desde_tabla(tab: pd.DataFrame):
    t = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return None
    chi2, p, dof, _ = chi2_contingency(t.values, correction=False)
    return chi2, p, dof, t


def espectro(serie: pd.Series):
    """Devuelve (periodos en horas, amplitudes) de la transformada de Fourier."""
    s = serie.sort_index()
    s = s[~s.index.duplicated(keep="first")].asfreq("h")
    s = s.interpolate(method="linear", limit_direction="both").dropna()
    n = len(s)
    if n < 48:
        return None
    x = s.values.astype(float)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)          # quitar tendencia lineal
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)                    # ciclos por hora
    amp = (2.0 / n) * np.abs(fft)
    freqs, amp = freqs[1:], amp[1:]                      # quitar componente 0
    return 1.0 / freqs, amp


def filtro_exponencial(arr: np.ndarray, theta: float = 0.05) -> np.ndarray:
    """y_k = y_{k-1} + theta*(x_k - y_{k-1}), tolerante a NaN."""
    out = np.full(len(arr), np.nan)
    prev = np.nan
    for i, v in enumerate(arr):
        if np.isnan(prev):
            prev = v
        elif not np.isnan(v):
            prev = prev + theta * (v - prev)
        out[i] = prev
    return out


# --------------------------------------------------------------------------
# Constructores de gráficas Altair (interactivas)
# --------------------------------------------------------------------------
def make_cobertura_chart(cob_long: pd.DataFrame):
    """cob_long: columnas [contaminante, anio, pct]."""
    base = alt.Chart(cob_long)
    heat = base.mark_rect().encode(
        x=alt.X("contaminante:N", title=None, sort=POLLUTANTS),
        y=alt.Y("anio:O", title="año"),
        color=alt.Color("pct:Q", title="% válido",
                        scale=alt.Scale(scheme="blues", domain=[0, 100])),
        tooltip=[alt.Tooltip("contaminante:N", title="contaminante"),
                 alt.Tooltip("anio:O", title="año"),
                 alt.Tooltip("pct:Q", title="% válido", format=".1f")],
    )
    texto = base.mark_text(fontSize=11).encode(
        x=alt.X("contaminante:N", sort=POLLUTANTS),
        y="anio:O",
        text=alt.Text("pct:Q", format=".0f"),
        color=alt.condition("datum.pct > 55", alt.value("white"), alt.value("#1E293B")),
    )
    return (heat + texto).properties(height=200)


def make_diurno_chart(dfm: pd.DataFrame, cont: str, unit: str):
    """dfm: columnas [hora, media, q1, q3]."""
    base = alt.Chart(dfm)
    band = base.mark_area(opacity=0.22, color=C_ACCENT).encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 3)))),
        y=alt.Y("q1:Q", title=f"{cont} ({unit})"),
        y2="q3:Q",
    )
    linea = base.mark_line(color=C_ACCENT, strokeWidth=2.5, point=True).encode(
        x="hora:Q",
        y="media:Q",
        tooltip=[alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("media:Q", title="promedio", format=".1f"),
                 alt.Tooltip("q1:Q", title="Q1", format=".1f"),
                 alt.Tooltip("q3:Q", title="Q3", format=".1f")],
    )
    return (band + linea).properties(height=300)


def make_estacional_chart(dfm: pd.DataFrame, cont: str, unit: str):
    """dfm: columnas [mes_idx, mes, valor, seca_caliente(bool)]."""
    return alt.Chart(dfm).mark_bar().encode(
        x=alt.X("mes:N", title=None, sort=MESES),
        y=alt.Y("valor:Q", title=f"{cont} ({unit}) — promedio"),
        color=alt.Color("seca_caliente:N",
                        scale=alt.Scale(domain=["Seca-caliente", "Resto del año"],
                                        range=["#F97316", "#94A3B8"]),
                        legend=alt.Legend(title="temporada")),
        tooltip=[alt.Tooltip("mes:N", title="mes"),
                 alt.Tooltip("valor:Q", title="promedio", format=".1f")],
    ).properties(height=300)


def make_espectro_chart(spec_df: pd.DataFrame, umbral: float, picos_df: pd.DataFrame):
    """spec_df: [periodo, amplitud]; picos_df: [periodo, amplitud]."""
    x = alt.X("periodo:Q", scale=alt.Scale(type="log"),
              title="periodo (horas) — escala logarítmica")
    linea = alt.Chart(spec_df).mark_line(color="#6366F1", strokeWidth=1.4).encode(
        x=x,
        y=alt.Y("amplitud:Q", title="amplitud"),
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", title="amplitud", format=".3f")],
    )
    regla = alt.Chart(pd.DataFrame({"y": [umbral]})).mark_rule(
        color="#DC2626", strokeDash=[5, 4]).encode(y="y:Q")
    marcas = alt.Chart(pd.DataFrame({"periodo": [24, 12], "etq": ["24 h", "12 h"]}))
    reglas_v = marcas.mark_rule(color="#94A3B8", strokeDash=[2, 3]).encode(x=x)
    etiquetas = marcas.mark_text(align="left", dx=4, dy=-6, fontSize=11,
                                 color="#64748B").encode(x=x, text="etq:N")
    puntos = alt.Chart(picos_df).mark_circle(color="#F97316", size=80).encode(
        x=x, y="amplitud:Q",
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", title="amplitud", format=".3f")],
    )
    return (linea + regla + reglas_v + etiquetas + puntos).properties(
        height=340).interactive()


def make_pca_chart(pca_df: pd.DataFrame):
    """pca_df: [estacion, PC1, PC2, zona, zona_nombre]."""
    zonas = [z for z in ZONE_COLORS if z in set(pca_df["zona"])]
    escala = alt.Scale(domain=zonas, range=[ZONE_COLORS[z] for z in zonas])
    base = alt.Chart(pca_df)
    ejes = (alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#CBD5E1").encode(y="z:Q")
            + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#CBD5E1").encode(x="z:Q"))
    puntos = base.mark_circle(size=140, opacity=0.9, stroke="white",
                              strokeWidth=1).encode(
        x=alt.X("PC1:Q", title="PC1 (~73%):  primarios (tráfico/industria)  ←→  ozono"),
        y=alt.Y("PC2:Q", title="PC2 (~14%):  SO₂ ↑  (corredor industrial norte)"),
        color=alt.Color("zona:N", scale=escala,
                        legend=alt.Legend(title="zona")),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("zona_nombre:N", title="zona"),
                 alt.Tooltip("PC1:Q", format=".2f"),
                 alt.Tooltip("PC2:Q", format=".2f")],
    )
    texto = base.mark_text(dy=-12, fontSize=10, color="#475569").encode(
        x="PC1:Q", y="PC2:Q", text="estacion:N")
    return (ejes + puntos + texto).properties(height=460).interactive()


def make_cargas_chart(cargas_long: pd.DataFrame):
    """cargas_long: [parametro, componente, carga]."""
    return alt.Chart(cargas_long).mark_bar().encode(
        x=alt.X("parametro:N", title=None),
        xOffset="componente:N",
        y=alt.Y("carga:Q", title="carga"),
        color=alt.Color("componente:N",
                        scale=alt.Scale(domain=["PC1", "PC2"],
                                        range=["#6366F1", "#10B981"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("parametro:N", title="contaminante"),
                 alt.Tooltip("componente:N", title="componente"),
                 alt.Tooltip("carga:Q", format=".3f")],
    ).properties(height=320)


def make_corr_chart(corr_long: pd.DataFrame, metodo: str):
    """corr_long: [v1, v2, corr]."""
    base = alt.Chart(corr_long)
    heat = base.mark_rect().encode(
        x=alt.X("v1:N", title=None, sort=POLLUTANTS),
        y=alt.Y("v2:N", title=None, sort=POLLUTANTS),
        color=alt.Color("corr:Q", title=f"r ({metodo})",
                        scale=alt.Scale(domain=[-1, 0, 1],
                                        range=["#2166AC", "#F7F7F7", "#B2182B"])),
        tooltip=[alt.Tooltip("v1:N", title=""), alt.Tooltip("v2:N", title=""),
                 alt.Tooltip("corr:Q", title="correlación", format=".2f")],
    )
    texto = base.mark_text(fontSize=10).encode(
        x=alt.X("v1:N", sort=POLLUTANTS),
        y=alt.Y("v2:N", sort=POLLUTANTS),
        text=alt.Text("corr:Q", format=".2f"),
        color=alt.condition("abs(datum.corr) > 0.55",
                            alt.value("white"), alt.value("#1E293B")),
    )
    return (heat + texto).properties(height=420)


def make_chi_prop_chart(prop_long: pd.DataFrame):
    """prop_long: [temporada, categoria, dias]. Barras 100% apiladas."""
    cats = [c for c in CAT_ORDER if c in set(prop_long["categoria"])]
    escala = alt.Scale(domain=cats,
                       range=[CAT_COLORS[CAT_ORDER.index(c)] for c in cats])
    return alt.Chart(prop_long).mark_bar().encode(
        x=alt.X("temporada:N", title=None, sort=SEASON_ORDER),
        y=alt.Y("dias:Q", title="proporción de días", stack="normalize",
                axis=alt.Axis(format="%")),
        color=alt.Color("categoria:N", scale=escala,
                        legend=alt.Legend(title="categoría O₃")),
        order=alt.Order("orden:Q"),
        tooltip=[alt.Tooltip("temporada:N", title="temporada"),
                 alt.Tooltip("categoria:N", title="categoría"),
                 alt.Tooltip("dias:Q", title="días")],
    ).properties(height=320)


# --------------------------------------------------------------------------
# Carga de datos (en caché)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_horario(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "horario_principal.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


@st.cache_data(show_spinner=False)
def load_diario(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "diario_todas.csv")
    df["dia"] = pd.to_datetime(df["dia"])
    return df


@st.cache_data(show_spinner=False)
def load_pca(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "pca_estaciones.csv")


@st.cache_data(show_spinner=False)
def load_cargas(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "cargas_pca.csv")


@st.cache_data(show_spinner=False)
def load_resumen(data_dir: str) -> dict:
    with open(Path(data_dir) / "resumen.json", encoding="utf-8") as fh:
        return json.load(fh)


def resolver_data_dir(propuesta: str):
    base = Path(__file__).resolve().parent
    candidatos = [Path(propuesta), base / propuesta, base,
                  base / "procesados", base.parent / "datos" / "procesados"]
    for c in candidatos:
        if (c / "resumen.json").exists():
            return c
    return None


# --------------------------------------------------------------------------
# Páginas
# --------------------------------------------------------------------------
def pagina_inicio(d):
    st.title(":material/air: Calidad del aire en la Ciudad de México")
    st.markdown(
        "Análisis de la **Red Automática de Monitoreo Atmosférico (RAMA/SIMAT)** "
        "del Valle de México, con datos horarios de 2024 a 2026. Recorre cada "
        "unidad del curso con el menú de la izquierda. "
        ":blue-badge[Unidad I] :blue-badge[Unidad II] :blue-badge[Unidad III]"
    )
    st.caption("Pregunta del proyecto: ¿qué patrones gobiernan la contaminación del "
               "aire en la CDMX —según la hora, la temporada y la zona— y cómo se "
               "relacionan los contaminantes entre sí?")

    r = d["resumen"]
    with st.container(horizontal=True):
        st.metric("Estaciones (PCA)", f"{d['pca']['estacion'].nunique()}", border=True)
        st.metric("Periodo", f"{min(r['anios'])}–{max(r['anios'])}", border=True)
        st.metric("Estación principal",
                  f"{r['nombre_estacion']} ({r['estacion_principal']})", border=True)
        st.metric("Varianza PC1 + PC2",
                  f"{(r['varianza_pc1'] + r['varianza_pc2'])*100:.0f}%", border=True)

    with st.container(border=True):
        st.subheader("Hallazgos principales")
        st.markdown(
            "- **Dos relojes en el aire.** El análisis espectral confirma un ciclo "
            "**diario de 24 h** (y su armónico de 12 h) y una **modulación estacional**.\n"
            "- **El espacio se ordena por química.** El PCA separa un núcleo de "
            "**contaminantes primarios** (tráfico/industria, nororiente y centro) de una "
            "periferia de **ozono** (surponiente elevado), con un eje aparte de **SO₂** "
            "que marca el corredor industrial del norte.\n"
            "- **Relaciones con sentido físico.** O₃ ↔ NO₂ negativa (precursor "
            "consumido), PM10 ↔ NO₂ positiva (tráfico común), O₃ ↔ temperatura positiva.\n"
            "- **Un resultado que invita a pensar.** La prueba χ² **no** detecta "
            "dependencia entre la calidad del aire por ozono y la temporada en la "
            "estación central (ver Unidad III)."
        )


def pagina_datos(d):
    st.header(":material/database: Los datos")
    st.markdown(
        "Cada estación de la RAMA mide los **contaminantes criterio** cada hora. "
        "Los archivos crudos vienen como un Excel por año y contaminante; el "
        "notebook los integró, limpió y exportó a los CSV que alimentan este tablero."
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        with st.container(border=True):
            st.markdown("**Contaminantes medidos**")
            tabla_cont = pd.DataFrame({
                "Contaminante": POLLUTANTS,
                "Unidad": [UNITS[p] for p in POLLUTANTS],
                "Tipo": ["Secundario (sol)" if p == "O3" else "Primario / mezcla"
                         for p in POLLUTANTS],
            })
            st.dataframe(tabla_cont, hide_index=True)
    with col2:
        with st.container(border=True):
            st.markdown(f"**Cobertura en {d['resumen']['nombre_estacion']} "
                        "(% de horas con dato válido)**")
            h = d["horario"].copy()
            h["anio"] = h["fecha"].dt.year
            cob = h.groupby("anio")[POLLUTANTS].apply(lambda x: x.notna().mean() * 100)
            cob_long = (cob.round(1).reset_index()
                        .melt(id_vars="anio", var_name="contaminante", value_name="pct"))
            st.altair_chart(make_cobertura_chart(cob_long))
            st.caption("Los datos de la RAMA tienen huecos reales; 2026 está poco poblado.")

    with st.expander("Ver una muestra de la serie horaria (Merced)",
                     icon=":material/table_chart:"):
        st.dataframe(d["horario"].head(24), hide_index=True)


def pagina_ciclos(d):
    st.header(":material/schedule: Unidad I · Los ciclos del aire")
    cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
    u = UNITS[cont]
    h = d["horario"].copy()
    h["hora"] = h["fecha"].dt.hour
    h["mes"] = h["fecha"].dt.month

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Ciclo diurno** (promedio por hora del día)")
            g = h.groupby("hora")[cont]
            dfm = pd.DataFrame({"hora": g.mean().index, "media": g.mean().values,
                                "q1": g.quantile(0.25).values,
                                "q3": g.quantile(0.75).values})
            st.altair_chart(make_diurno_chart(dfm, cont, u))
            if cont == "O3":
                st.caption("El ozono es secundario: se forma con el sol y llega a su "
                           "máximo a primera hora de la tarde.")
    with col2:
        with st.container(border=True):
            st.markdown("**Ciclo anual** (promedio por mes)")
            media_mes = h.groupby("mes")[cont].mean().reindex(range(1, 13))
            dfm = pd.DataFrame({
                "mes_idx": range(1, 13),
                "mes": MESES,
                "valor": media_mes.values,
                "seca_caliente": ["Seca-caliente" if m in (3, 4, 5) else "Resto del año"
                                  for m in range(1, 13)],
            })
            st.altair_chart(make_estacional_chart(dfm, cont, u))
            if cont == "O3":
                st.caption("En naranja la temporada seca-caliente (mar–may), cuando el "
                           "ozono se dispara por la máxima radiación.")

    with st.container(border=True):
        st.markdown("**Filtros de suavizado**")
        st.caption("La serie horaria es ruidosa; los filtros revelan la tendencia. "
                   "Media y mediana con ventana q = 24 h; exponencial con θ = 0.05.")
        anios = sorted(h["fecha"].dt.year.unique())
        ay = st.select_slider("Año a mostrar", anios, value=anios[0])
        s = d["horario"]
        s = s[s["fecha"].dt.year == ay].set_index("fecha")[cont].sort_index()
        if s.notna().sum() < 100:
            st.warning("Pocos datos válidos en ese año para este contaminante.",
                       icon=":material/warning:")
        else:
            df_f = pd.DataFrame({
                "Señal cruda": s,
                "Media móvil": s.rolling(24, center=True, min_periods=6).mean(),
                "Mediana móvil": s.rolling(24, center=True, min_periods=6).median(),
                "Exponencial": pd.Series(filtro_exponencial(s.values, 0.05), index=s.index),
            })
            st.line_chart(df_f, y_label=f"{cont} ({u})",
                          color=[C_CRUDA, C_MEDIA, C_MEDIANA, C_EXPO])


def pagina_espectro(d):
    st.header(":material/graphic_eq: Unidad II · Análisis espectral")
    st.markdown(
        "La transformada de Fourier descompone la serie en las **ondas** que la "
        "forman y mide cuál es más fuerte. Confirma de forma objetiva los ciclos "
        "vistos en la exploración."
    )
    h = d["horario"]
    anios = sorted(h["fecha"].dt.year.unique())
    c1, c2 = st.columns(2)
    cont = c1.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
    ay = c2.selectbox("Año (ventana de análisis)", anios, index=0)

    serie = h[h["fecha"].dt.year == ay].set_index("fecha")[cont].sort_index()
    res = espectro(serie)
    if res is None:
        st.warning("No hay suficientes datos continuos en esa combinación.",
                   icon=":material/warning:")
        return
    periodos, amp = res
    umbral = float(amp.mean() + 3 * amp.std())
    mask = (periodos >= 2) & (periodos <= 1000)
    spec_df = pd.DataFrame({"periodo": periodos[mask], "amplitud": amp[mask]})
    pmask = mask & (amp > umbral)
    picos_df = pd.DataFrame({"periodo": periodos[pmask], "amplitud": amp[pmask]})

    with st.container(border=True):
        st.markdown(f"**Espectro de {cont} en Merced ({ay})**")
        st.altair_chart(make_espectro_chart(spec_df, umbral, picos_df))

    col1, col2 = st.columns([2, 3])
    with col1:
        with st.container(border=True):
            st.markdown("**Periodos dominantes**")
            top = (picos_df.sort_values("amplitud", ascending=False).head(8)
                   .rename(columns={"periodo": "periodo (h)", "amplitud": "amplitud"}))
            top["periodo (h)"] = top["periodo (h)"].round(1)
            top["amplitud"] = top["amplitud"].round(3)
            st.dataframe(top, hide_index=True)
    with col2:
        if cont == "O3":
            st.info("En el ozono aparecen picos cerca de **24 h** (el ciclo "
                    "fotoquímico diario) y **12 h** (el armónico que corrige la "
                    "forma asimétrica de la curva).", icon=":material/lightbulb:")
        st.caption("Tip: la gráfica es interactiva — arrastra para hacer zoom y pasa "
                   "el cursor sobre los puntos para ver el periodo exacto.")


def pagina_pca(d):
    st.header(":material/scatter_plot: Unidad II · PCA: el mapa de las estaciones")
    r = d["resumen"]

    with st.container(horizontal=True):
        st.metric("Varianza PC1", f"{r['varianza_pc1']*100:.1f}%", border=True)
        st.metric("Varianza PC2", f"{r['varianza_pc2']*100:.1f}%", border=True)
        st.metric("Componentes para 90%", f"{r['q_90']}", border=True)

    pca = d["pca"].copy()
    pca["zona_nombre"] = pca["zona"].map(ZONE_NAMES).fillna(pca["zona"])
    with st.container(border=True):
        st.markdown("**Estaciones en el plano PC1–PC2** (color por zona)")
        st.altair_chart(make_pca_chart(pca))

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            st.markdown("**Cargas: qué pesa en cada componente**")
            cargas_long = d["cargas"].melt(id_vars="parametro",
                                           value_vars=["PC1", "PC2"],
                                           var_name="componente", value_name="carga")
            st.altair_chart(make_cargas_chart(cargas_long))
    with col2:
        with st.container(border=True):
            st.markdown("**Interpretación**")
            st.markdown(
                "- **PC1** (horizontal): los primarios (CO, NO, NO₂, NOX) cargan "
                "negativo y el O₃ positivo. Es un eje **primarios ↔ ozono**, no de "
                "\"nivel general\".\n"
                "- **PC2** (vertical): lo domina el **SO₂**. Aísla al corredor "
                "industrial del norte (Tlalnepantla, Cuautitlán, Atizapán).\n"
                "- El **PM no entró** en este PCA (lo descartó el filtro de cobertura)."
            )
            st.caption("Esta lectura corrige la redacción original del notebook, que "
                       "describía PC1 como 'nivel general' e incluía al PM.")


def pagina_correlacion(d):
    st.header(":material/analytics: Unidad III · Correlación y prueba χ²")

    with st.container(border=True):
        st.markdown("**Correlación entre contaminantes** (serie horaria de Merced)")
        metodo = st.segmented_control(
            "Coeficiente", ["pearson", "spearman", "kendall"], default="pearson")
        if metodo is None:
            metodo = "pearson"
        cols = [c for c in POLLUTANTS if d["horario"][c].notna().sum() > 100]
        corr = d["horario"][cols].corr(method=metodo)
        corr_long = corr.reset_index().melt(id_vars="index", var_name="v2",
                                            value_name="corr").rename(columns={"index": "v1"})
        st.altair_chart(make_corr_chart(corr_long, metodo))
        st.caption("Relaciones clave: O₃ ↔ NO₂ negativa (el NO₂ se consume al formarse "
                   "el ozono) y PM10 ↔ NO₂ positiva (ambos vienen del tráfico). "
                   "Correlación no implica causalidad.")

    diario = d["diario"]
    o3 = diario[diario["parametro"] == "O3"].copy()
    estaciones = sorted(o3["estacion"].unique())
    est = st.selectbox("Estación para la prueba χ²", estaciones,
                       index=estaciones.index("MER") if "MER" in estaciones else 0)
    tab = tabla_contingencia(o3[o3["estacion"] == est])
    res = chi_desde_tabla(tab)

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            st.markdown("**Tabla de contingencia** (días observados)")
            st.dataframe(tab)
            prop_long = (tab.reset_index()
                         .melt(id_vars=tab.index.name or "index",
                               var_name="temporada", value_name="dias"))
            prop_long.columns = ["categoria", "temporada", "dias"]
            prop_long["orden"] = prop_long["categoria"].map(
                {c: i for i, c in enumerate(CAT_ORDER)})
            st.markdown("**Proporción de categorías por temporada**")
            st.altair_chart(make_chi_prop_chart(prop_long))
    with col2:
        with st.container(border=True):
            st.markdown("**Resultado de la prueba**")
            if res is not None:
                chi2, p, dof, _ = res
                st.metric("Estadístico χ²", f"{chi2:.2f}", border=True)
                st.metric("p-valor", f"{p:.3f}", border=True)
                crit = chi2_dist.ppf(0.95, dof)
                if p < 0.05:
                    st.success(f"p < 0.05 → se rechaza H₀ (gl={dof}, crítico={crit:.2f}). "
                               "Hay dependencia con la temporada.",
                               icon=":material/check_circle:")
                else:
                    st.warning(f"p ≥ 0.05 → NO se rechaza H₀ (gl={dof}, "
                               f"crítico={crit:.2f}). No se detecta dependencia.",
                               icon=":material/info:")
            else:
                st.info("Datos insuficientes para la prueba en esta estación.")

    st.markdown(
        "Arriba ves el χ² **recalculado en vivo** sobre la tabla mostrada. El "
        f"notebook había exportado χ² ≈ {d['resumen']['chi2']}, p ≈ {d['resumen']['chi2_p']:.2f}; "
        "los números cambian un poco según el recorte de datos, pero **ambos "
        "coinciden en lo esencial: p ≥ 0.05, no se rechaza H₀ en Merced**. Esto "
        "sorprende, porque la estacionalidad del ozono está documentada. La razón: "
        "Merced es céntrica y su ozono no se concentra tan fuerte por temporada como "
        "el del surponiente; la prueba categórica pierde potencia ahí."
    )

    with st.expander("Ver la prueba χ² en TODAS las estaciones",
                     icon=":material/table_chart:"):
        filas = []
        for e, sub in o3.groupby("estacion"):
            if len(sub) < 120:
                continue
            r2 = chi_desde_tabla(tabla_contingencia(sub))
            if r2 is None:
                continue
            chi2, p, dof, _ = r2
            filas.append({"estación": e, "n_días": len(sub), "χ²": round(chi2, 2),
                          "gl": dof, "p": round(p, 3),
                          "¿rechaza H₀?": "sí" if p < 0.05 else "no"})
        tabla_todas = pd.DataFrame(filas).sort_values("p")
        st.dataframe(tabla_todas, hide_index=True)
        n_rech = int((tabla_todas["¿rechaza H₀?"] == "sí").sum())
        st.caption(f"Solo {n_rech} de {len(tabla_todas)} estaciones rechazan la "
                   "independencia. El patrón estacional existe, pero esta prueba "
                   "categórica solo lo detecta en pocos sitios.")


def pagina_conclusiones(d):
    st.header(":material/flag: Conclusiones")
    with st.container(border=True):
        st.markdown(
            "El proyecto tomó datos reales y con huecos de la RAMA (2024–2026, "
            "horarios) y, aplicando las técnicas del curso **implementadas a mano y "
            "verificadas contra librerías**, mostró que:\n\n"
            "1. **El aire tiene dos relojes.** El análisis espectral confirmó un ciclo "
            "diario de 24 h (con armónico de 12 h) y una modulación estacional. El "
            "ozono es esencialmente periódico y predecible.\n"
            "2. **El espacio se ordena por química.** El PCA (confirmado por el MDS) "
            "separa un núcleo de primarios (tráfico/industria) de una periferia de "
            "ozono, con un eje de SO₂ que marca el corredor industrial del norte.\n"
            "3. **Las relaciones tienen sentido físico.** O₃ ↔ NO₂ negativa, "
            "PM10 ↔ NO₂ positiva, O₃ ↔ temperatura positiva.\n"
            "4. **Un resultado que invita a pensar.** La χ² no detectó dependencia "
            "estacional en la estación central; solo dos estaciones de la red la "
            "rechazan. Es una oportunidad de análisis crítico (potencia de la prueba, "
            "elección de estación y de categorías), no un fracaso."
        )
    with st.container(border=True):
        st.markdown("**Notas de honestidad para la defensa**")
        st.markdown(
            "- La interpretación del PCA se ajustó a las cargas reales: PC1 = "
            "primarios ↔ ozono (no 'nivel general'); PC2 = SO₂; el PM no entró al "
            "análisis.\n"
            "- La conclusión de la χ² se reportó tal cual la dan los números "
            "(p ≈ 0.66 en Merced), explicando por qué no se rechaza H₀."
        )


# --------------------------------------------------------------------------
# App principal
# --------------------------------------------------------------------------
PAGINAS = {
    "Inicio": pagina_inicio,
    "Los datos": pagina_datos,
    "Unidad I · Ciclos y filtros": pagina_ciclos,
    "Unidad II · Análisis espectral": pagina_espectro,
    "Unidad II · PCA (mapa)": pagina_pca,
    "Unidad III · Correlación y χ²": pagina_correlacion,
    "Conclusiones": pagina_conclusiones,
}


def main():
    st.set_page_config(page_title="Calidad del aire CDMX",
                       page_icon=":material/air:", layout="wide")

    st.sidebar.title("Calidad del aire CDMX")
    st.sidebar.caption("RAMA/SIMAT · 2024–2026")
    eleccion = st.sidebar.radio("Secciones", list(PAGINAS.keys()))

    with st.sidebar.expander("Opciones de datos", icon=":material/tune:"):
        propuesta = st.text_input("Ruta de datos", DATA_DIR_DEFAULT)

    data_dir = resolver_data_dir(propuesta)
    if data_dir is None:
        st.error(
            "No encontré los archivos de datos. Asegúrate de que exista "
            f"`{propuesta}/resumen.json` y los demás CSV junto a `app.py`.",
            icon=":material/error:")
        st.stop()

    try:
        d = {
            "horario": load_horario(str(data_dir)),
            "diario": load_diario(str(data_dir)),
            "pca": load_pca(str(data_dir)),
            "cargas": load_cargas(str(data_dir)),
            "resumen": load_resumen(str(data_dir)),
        }
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al leer los datos: {e}", icon=":material/error:")
        st.stop()

    st.sidebar.caption(f"Datos cargados desde: `{data_dir}`")
    PAGINAS[eleccion](d)


if __name__ == "__main__":
    main()
