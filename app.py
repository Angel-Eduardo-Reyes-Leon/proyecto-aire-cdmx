"""
Aire CDMX — ¿qué gobierna la contaminación en la Ciudad de México?
==================================================================
Tablero de datos (RAMA/SIMAT, 2024–2026) construido con Streamlit + Altair.

No está organizado por las unidades del curso, sino por la PREGUNTA: qué
gobierna el aire de la CDMX según la hora, la temporada y la zona, y cómo se
relacionan los contaminantes. Cada vista abre con el hallazgo y muestra solo la
gráfica que lo responde; el análisis (Fourier, PCA, χ²) es el motor detrás, y
el detalle técnico queda accesible bajo demanda.

Rendimiento: cada sección se recalcula y redibuja por separado (@st.fragment),
los cómputos se memorizan (@st.cache_data) y las series densas se submuestrean,
de modo que un clic nunca recarga toda la página.

Local:  streamlit run app.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()

# ==========================================================================
# Sistema de diseño  (paleta editorial; color con propósito, no decorativo)
# ==========================================================================
PAPER = "#FAF8F3"      # fondo papel cálido (estética editorial, no genérica)
INK = "#1C1B19"        # texto principal
SUBTLE = "#76726B"     # texto secundario / ejes
HAIR = "#E5E0D6"       # líneas finas, retícula tenue
ACCENT = "#1D6A75"     # tinta de acento: resalta UNA serie (Tufte)
WARM = "#C2562F"       # acento cálido: énfasis secundario
MUTED = "#CEC8BC"      # gris cálido: series atenuadas

# Escala SEMÁNTICA de calidad del aire (pre-atentiva: verde=bien, rojo=mal)
CAT_ORDER = ["Buena", "Aceptable", "Mala", "Muy mala"]
CAT_COLORS = ["#4C9A5B", "#E0B13C", "#DB7A33", "#B23A2E"]

# Zonas de la ciudad — categórica sobria y cohesiva (CE, el centro, en acento)
ZONE_NAMES = {"NE": "Nororiente", "NO": "Noroeste", "CE": "Centro",
              "SO": "Surponiente", "SE": "Sureste"}
ZONE_COLORS = {"CE": ACCENT, "NE": WARM, "NO": "#C2992B",
               "SO": "#5E8C6A", "SE": "#7E8AA0"}

DIV_RANGE = [ACCENT, "#F2EEE6", WARM]      # divergente centrado en 0 (correlación)

POLLUTANTS = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "PMCO", "SO2"]
UNITS = {"CO": "ppm", "NO": "ppb", "NO2": "ppb", "NOX": "ppb", "O3": "ppb",
         "PM10": "µg/m³", "PM2.5": "µg/m³", "PMCO": "µg/m³", "SO2": "ppb"}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASON_ORDER = ["Seca-fría", "Seca-caliente", "Lluvias"]

DATA_DIR_DEFAULT = "datos/procesados"

# CSS: tipografía editorial (display serif + cuerpo sans), papel, sin "cajas"
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp,
[data-testid="stSidebar"] { background-color: #FAF8F3 !important; }
.stApp, [data-testid="stAppViewContainer"] { color: #1C1B19; }
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

html, body, p, li, div, span, label, .stMarkdown, [data-testid="stWidgetLabel"] {
  font-family: 'Inter', system-ui, sans-serif;
}
h1, h2, h3, h4 {
  font-family: 'Fraunces', Georgia, serif !important;
  color: #1C1B19 !important; letter-spacing: -0.012em; font-weight: 600;
}
h1 { font-size: 2.15rem; line-height: 1.12; margin-bottom: .15rem; }
h2 { font-size: 1.5rem; }

.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1080px; }

/* barra lateral */
[data-testid="stSidebar"] { border-right: 1px solid #E5E0D6; }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
[data-testid="stSidebar"] * { color: #1C1B19 !important; }

/* tipos editoriales propios */
.kicker { font-family:'Inter'; font-size:.74rem; font-weight:600; letter-spacing:.14em;
          text-transform:uppercase; color:#1D6A75; margin-bottom:.35rem; }
.lede   { font-size:1.06rem; line-height:1.62; color:#34322E; max-width:62ch; margin:.2rem 0 1.1rem; }
.note   { font-size:.86rem; line-height:1.55; color:#76726B; max-width:64ch; margin:.3rem 0; }
.rule   { height:1px; background:#E5E0D6; border:0; margin:1.6rem 0; }

/* fila de cifras (en vez de tarjetas KPI decorativas) */
.statrow { display:flex; flex-wrap:wrap; gap:2.4rem; margin:.4rem 0 1.4rem; }
.stat .v { font-family:'Fraunces',serif; font-size:2rem; font-weight:600; color:#1C1B19; line-height:1; }
.stat .l { font-size:.8rem; color:#76726B; margin-top:.3rem; max-width:18ch; }
.stat .v .u { font-size:1rem; color:#76726B; font-weight:500; }

/* leyenda de calidad del aire */
.aqikey { display:flex; flex-wrap:wrap; gap:1.1rem; font-size:.8rem; color:#34322E; margin:.2rem 0; }
.aqikey span { display:inline-flex; align-items:center; gap:.4rem; }
.aqikey i { width:11px; height:11px; border-radius:2px; display:inline-block; }
</style>
"""


# ==========================================================================
# Helpers de presentación
# ==========================================================================
def kicker(text):
    st.markdown(f"<div class='kicker'>{text}</div>", unsafe_allow_html=True)


def lede(text):
    st.markdown(f"<p class='lede'>{text}</p>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<p class='note'>{text}</p>", unsafe_allow_html=True)


def rule():
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)


def stat_row(items):
    """items: lista de (valor, sufijo, etiqueta)."""
    cells = ""
    for v, u, l in items:
        suf = f"<span class='u'> {u}</span>" if u else ""
        cells += (f"<div class='stat'><div class='v'>{v}{suf}</div>"
                  f"<div class='l'>{l}</div></div>")
    st.markdown(f"<div class='statrow'>{cells}</div>", unsafe_allow_html=True)


def aqi_key():
    chips = "".join(
        f"<span><i style='background:{CAT_COLORS[i]}'></i>{c}</span>"
        for i, c in enumerate(CAT_ORDER))
    st.markdown(f"<div class='aqikey'>{chips}</div>", unsafe_allow_html=True)


def themed(ch):
    """Aplica la identidad visual editorial a una gráfica de nivel superior."""
    return (ch.configure(background=PAPER, font="Inter",
                         padding={"top": 6, "left": 4, "right": 4, "bottom": 4})
              .configure_view(stroke=None)
              .configure_axis(labelColor=SUBTLE, titleColor=INK, gridColor=HAIR,
                              domainColor=HAIR, tickColor=HAIR, labelFontSize=11,
                              titleFontSize=12, titleFontWeight=600, grid=True,
                              titlePadding=10)
              .configure_legend(labelColor=INK, titleColor=SUBTLE, labelFontSize=11,
                                titleFontSize=10, titleFontWeight=600, symbolType="square")
              .configure_title(color=INK, fontSize=15, fontWeight=600,
                               anchor="start", font="Fraunces"))


def show(ch):
    st.altair_chart(themed(ch), width='stretch', theme=None)


def pan_x():
    """Arrastrar para desplazar en X; sin zoom de rueda (no secuestra la página)."""
    return alt.selection_interval(bind="scales", encodings=["x"], zoom=False)


def pan_xy():
    return alt.selection_interval(bind="scales", zoom=False)


# ==========================================================================
# Cálculo (química / estadística del proyecto)
# ==========================================================================
def temporada(mes):
    if mes in (11, 12, 1, 2):
        return "Seca-fría"
    if mes in (3, 4, 5):
        return "Seca-caliente"
    return "Lluvias"


def categoria_o3(valor):
    if pd.isna(valor):
        return np.nan
    if valor <= 58:
        return "Buena"
    if valor <= 90:
        return "Aceptable"
    if valor <= 135:
        return "Mala"
    return "Muy mala"


def tabla_contingencia(df_o3):
    cat = df_o3["maximo"].apply(categoria_o3)
    temp = df_o3["dia"].dt.month.map(temporada)
    tab = pd.crosstab(cat, temp)
    filas = [c for c in CAT_ORDER if c in tab.index]
    cols = [s for s in SEASON_ORDER if s in tab.columns]
    return tab.reindex(index=filas, columns=cols).fillna(0).astype(int)


def chi_desde_tabla(tab):
    t = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return None
    chi2, p, dof, _ = chi2_contingency(t.values, correction=False)
    return chi2, p, dof, t


def espectro(serie):
    s = serie.sort_index()
    s = s[~s.index.duplicated(keep="first")].asfreq("h")
    s = s.interpolate(method="linear", limit_direction="both").dropna()
    n = len(s)
    if n < 48:
        return None
    x = s.values.astype(float)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    amp = (2.0 / n) * np.abs(fft)
    return 1.0 / freqs[1:], amp[1:]


# ==========================================================================
# Carga y analítica (memorizadas)
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_horario(d):
    df = pd.read_csv(Path(d) / "horario_principal.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


@st.cache_data(show_spinner=False)
def load_diario(d):
    df = pd.read_csv(Path(d) / "diario_todas.csv")
    df["dia"] = pd.to_datetime(df["dia"])
    return df


@st.cache_data(show_spinner=False)
def load_pca(d):
    return pd.read_csv(Path(d) / "pca_estaciones.csv")


@st.cache_data(show_spinner=False)
def load_cargas(d):
    return pd.read_csv(Path(d) / "cargas_pca.csv")


@st.cache_data(show_spinner=False)
def load_resumen(d):
    with open(Path(d) / "resumen.json", encoding="utf-8") as fh:
        return json.load(fh)


def resolver_data_dir(propuesta):
    base = Path(__file__).resolve().parent
    for c in [Path(propuesta), base / propuesta, base, base / "procesados",
              base.parent / "datos" / "procesados"]:
        if (c / "resumen.json").exists():
            return c
    return None


@st.cache_data(show_spinner=False)
def serie_categoria(d, estacion):
    """Categoría diaria de calidad por ozono (máximo) de una estación."""
    diario = load_diario(d)
    o3 = diario[(diario["parametro"] == "O3") & (diario["estacion"] == estacion)].copy()
    o3["categoria"] = o3["maximo"].apply(categoria_o3)
    o3 = o3.dropna(subset=["categoria"])
    o3["anio"] = o3["dia"].dt.year
    o3["doy"] = o3["dia"].dt.dayofyear
    o3["mes"] = o3["dia"].dt.month
    return o3


@st.cache_data(show_spinner=False)
def panorama_stats(d):
    r = load_resumen(d)
    o3 = serie_categoria(d, r["estacion_principal"])
    pct_buena = (o3["categoria"] == "Buena").mean() * 100 if len(o3) else np.nan
    da = diurno_abs(d, "O3")
    hora_pico = int(da.loc[da["media"].idxmax(), "hora"])
    mn, mx = float(da["media"].min()), float(da["media"].max())
    swing = mx / mn if mn > 0 else np.nan
    n_est = int(load_diario(d)["estacion"].nunique())
    return {"pct_buena": pct_buena, "hora_pico": hora_pico, "swing": swing,
            "o3_min": mn, "o3_max": mx, "n_est": n_est, "principal": r["nombre_estacion"]}


@st.cache_data(show_spinner=False)
def diurno_abs(d, cont):
    h = load_horario(d).copy()
    h["hora"] = h["fecha"].dt.hour
    g = h.groupby("hora")[cont]
    return pd.DataFrame({"hora": g.mean().index, "media": g.mean().values,
                         "q1": g.quantile(0.25).values, "q3": g.quantile(0.75).values})


@st.cache_data(show_spinner=False)
def diurno_norm(d):
    h = load_horario(d).copy()
    h["hora"] = h["fecha"].dt.hour
    out = []
    for c in POLLUTANTS:
        m = h.groupby("hora")[c].mean()
        if m.notna().sum() < 12 or not m.std() > 0:
            continue
        out.append(pd.DataFrame({"hora": m.index, "contaminante": c,
                                 "z": ((m - m.mean()) / m.std()).values}))
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner=False)
def espectro_cache(d, cont, anio):
    h = load_horario(d)
    serie = h[h["fecha"].dt.year == anio].set_index("fecha")[cont].sort_index()
    res = espectro(serie)
    if res is None:
        return None
    per, amp = res
    umbral = float(amp.mean() + 3 * amp.std())
    mask = (per >= 2) & (per <= 1000)
    spec = pd.DataFrame({"periodo": per[mask], "amplitud": amp[mask]})
    pmask = mask & (amp > umbral)
    picos = pd.DataFrame({"periodo": per[pmask], "amplitud": amp[pmask]})
    return spec, picos, umbral


@st.cache_data(show_spinner=False)
def mensual(d, cont):
    h = load_horario(d).copy()
    h["mes"] = h["fecha"].dt.month
    m = h.groupby("mes")[cont].mean().reindex(range(1, 13))
    return pd.DataFrame({"mes": MESES, "valor": m.values,
                         "grupo": ["Seca-caliente" if i in (3, 4, 5) else "Resto del año"
                                   for i in range(1, 13)]})


@st.cache_data(show_spinner=False)
def estacional_categoria(d, estacion):
    o3 = serie_categoria(d, estacion)
    o3["temporada"] = o3["mes"].map(temporada)
    tab = pd.crosstab(o3["categoria"], o3["temporada"])
    tab = tab.reindex(index=[c for c in CAT_ORDER if c in tab.index],
                      columns=[s for s in SEASON_ORDER if s in tab.columns]).fillna(0)
    long = tab.reset_index().melt(id_vars="categoria", var_name="temporada", value_name="dias")
    long["orden"] = long["categoria"].map({c: i for i, c in enumerate(CAT_ORDER)})
    res = chi_desde_tabla(tabla_contingencia(o3[["dia", "maximo"]]))
    return long, res


@st.cache_data(show_spinner=False)
def corr_data(d, metodo):
    h = load_horario(d)
    cols = [c for c in POLLUTANTS if h[c].notna().sum() > 100]
    corr = h[cols].corr(method=metodo)
    if corr.shape[0] >= 3:
        dist = 1.0 - corr.abs().values
        np.fill_diagonal(dist, 0.0)
        orden = [corr.columns[i]
                 for i in leaves_list(linkage(squareform(dist, checks=False), method="average"))]
    else:
        orden = list(corr.columns)
    pos = {v: i for i, v in enumerate(orden)}
    long = (corr.reset_index().melt(id_vars="index", var_name="v2", value_name="corr")
            .rename(columns={"index": "v1"}))
    long = long[long.apply(lambda r: pos[r["v1"]] >= pos[r["v2"]], axis=1)]
    return long, orden


def _downsample(df, n=2500):
    return df.iloc[::int(np.ceil(len(df) / n))] if len(df) > n else df


# ==========================================================================
# Gráficas (solo las que responden la pregunta)
# ==========================================================================
def chart_timeline(o3):
    escala = alt.Scale(domain=CAT_ORDER, range=CAT_COLORS)
    return alt.Chart(o3).mark_rect().encode(
        x=alt.X("doy:Q", title="día del año →", scale=alt.Scale(domain=[1, 366]),
                axis=alt.Axis(values=[1, 60, 120, 182, 244, 305, 366], grid=False)),
        color=alt.Color("categoria:N", scale=escala, sort=CAT_ORDER, legend=None),
        tooltip=[alt.Tooltip("dia:T", title="fecha"),
                 alt.Tooltip("categoria:N", title="calidad (O₃)"),
                 alt.Tooltip("maximo:Q", title="O₃ máx (ppb)", format=".0f")],
    ).properties(width="container", height=44).facet(
        row=alt.Row("anio:O", title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12,
                                      labelColor=INK, labelFontWeight=600))
    ).resolve_scale(x="shared")


def chart_diurno(dfm, cont, unit):
    base = alt.Chart(dfm)
    band = base.mark_area(opacity=0.16, color=ACCENT).encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 3)), grid=False)),
        y=alt.Y("q1:Q", title=f"{cont} ({unit})"), y2="q3:Q")
    ref = alt.Chart(pd.DataFrame({"y": [float(dfm["media"].mean())]})).mark_rule(
        color=SUBTLE, strokeDash=[2, 3], opacity=0.7).encode(y="y:Q")
    linea = base.mark_line(color=ACCENT, strokeWidth=2.6,
                           point=alt.OverlayMarkDef(color=ACCENT, size=26)).encode(
        x="hora:Q", y="media:Q",
        tooltip=[alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("media:Q", title="promedio", format=".1f"),
                 alt.Tooltip("q1:Q", title="Q1", format=".1f"),
                 alt.Tooltip("q3:Q", title="Q3", format=".1f")])
    return (band + ref + linea).properties(height=300)


def chart_diurno_norm(dfn, cont):
    test = f"datum.contaminante === '{cont}'"
    return alt.Chart(dfn).mark_line().encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 6)), grid=False)),
        y=alt.Y("z:Q", title="nivel relativo", axis=alt.Axis(grid=False)),
        detail="contaminante:N",
        color=alt.condition(test, alt.value(ACCENT), alt.value(MUTED)),
        size=alt.condition(test, alt.value(2.8), alt.value(1.0)),
        opacity=alt.condition(test, alt.value(1.0), alt.value(0.55)),
        tooltip=[alt.Tooltip("contaminante:N", title="contaminante"),
                 alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("z:Q", title="nivel (z)", format=".2f")],
    ).properties(height=300)


def chart_espectro(spec, picos, umbral):
    x = alt.X("periodo:Q", scale=alt.Scale(type="log"),
              title="periodo (horas, escala log)")
    linea = alt.Chart(spec).mark_line(color=ACCENT, strokeWidth=1.5).encode(
        x=x, y=alt.Y("amplitud:Q", title="amplitud", axis=alt.Axis(grid=False)),
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", format=".3f")])
    marcas = alt.Chart(pd.DataFrame({"p": [24, 12], "t": ["24 h", "12 h"]}))
    reglas = marcas.mark_rule(color=SUBTLE, strokeDash=[2, 3], opacity=0.8).encode(x="p:Q")
    etq = marcas.mark_text(align="left", dx=4, dy=-6, fontSize=11, fontWeight=600,
                           color=SUBTLE).encode(x="p:Q", text="t:N")
    pts = alt.Chart(picos).mark_point(color=WARM, size=80, filled=True).encode(
        x=x, y="amplitud:Q",
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", format=".3f")])
    return (linea + reglas + etq + pts).properties(height=260).add_params(pan_x())


def chart_mensual(dfm, cont, unit):
    return alt.Chart(dfm).mark_bar().encode(
        x=alt.X("mes:N", title=None, sort=MESES, axis=alt.Axis(grid=False)),
        y=alt.Y("valor:Q", title=f"{cont} ({unit})"),
        color=alt.Color("grupo:N", scale=alt.Scale(
            domain=["Seca-caliente", "Resto del año"], range=[WARM, MUTED]),
            legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("mes:N", title="mes"),
                 alt.Tooltip("valor:Q", title="promedio", format=".1f")],
    ).properties(height=300)


def chart_estacional_cat(long):
    escala = alt.Scale(domain=CAT_ORDER, range=CAT_COLORS)
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("temporada:N", title=None, sort=SEASON_ORDER, axis=alt.Axis(grid=False)),
        y=alt.Y("dias:Q", title="proporción de días", stack="normalize",
                axis=alt.Axis(format="%", grid=False)),
        color=alt.Color("categoria:N", scale=escala, sort=CAT_ORDER, legend=None),
        order=alt.Order("orden:Q", sort="descending"),
        tooltip=[alt.Tooltip("temporada:N", title="temporada"),
                 alt.Tooltip("categoria:N", title="categoría"),
                 alt.Tooltip("dias:Q", title="días")],
    ).properties(height=300)


def chart_mapa(pca, cargas):
    zonas = [z for z in ZONE_COLORS if z in set(pca["zona"])]
    escala = alt.Scale(domain=zonas, range=[ZONE_COLORS[z] for z in zonas])
    rad_p = float(np.nanmax(np.abs(pca[["PC1", "PC2"]].values)))
    rad_l = float(np.nanmax(np.abs(cargas[["PC1", "PC2"]].values)))
    k = (rad_p / rad_l) * 0.72 if rad_l > 0 else 1.0
    fl = cargas.copy()
    fl["x"], fl["y"] = fl["PC1"] * k, fl["PC2"] * k
    ejes = (alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=HAIR).encode(y="z:Q")
            + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=HAIR).encode(x="z:Q"))
    vec = alt.Chart(fl).mark_rule(color=SUBTLE, strokeWidth=1.4, opacity=0.6).encode(
        x=alt.datum(0), y=alt.datum(0), x2="x:Q", y2="y:Q")
    vlab = alt.Chart(fl).mark_text(color=WARM, fontWeight=700, fontSize=12).encode(
        x="x:Q", y="y:Q", text="parametro:N")
    base = alt.Chart(pca)
    pts = base.mark_circle(size=150, opacity=0.92, stroke="white", strokeWidth=1).encode(
        x=alt.X("PC1:Q", title="primarios (tráfico/industria)  ←→  ozono"),
        y=alt.Y("PC2:Q", title="SO₂ (corredor industrial norte)  ↑"),
        color=alt.Color("zona:N", scale=escala,
                        legend=alt.Legend(title="zona", orient="top-right")),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("nombre:N", title="nombre"),
                 alt.Tooltip("zona_nombre:N", title="zona")])
    lab = base.mark_text(dy=-13, fontSize=10, color=INK).encode(
        x="PC1:Q", y="PC2:Q", text="estacion:N")
    return (ejes + vec + vlab + pts + lab).properties(height=480).add_params(pan_xy())


def chart_corr(long, metodo, orden):
    base = alt.Chart(long)
    heat = base.mark_rect().encode(
        x=alt.X("v1:N", title=None, sort=orden, axis=alt.Axis(grid=False)),
        y=alt.Y("v2:N", title=None, sort=orden, axis=alt.Axis(grid=False)),
        color=alt.Color("corr:Q", title=f"r ({metodo})",
                        scale=alt.Scale(domain=[-1, 0, 1], range=DIV_RANGE)),
        tooltip=[alt.Tooltip("v1:N", title=""), alt.Tooltip("v2:N", title=""),
                 alt.Tooltip("corr:Q", title="correlación", format=".2f")])
    txt = base.mark_text(fontSize=10).encode(
        x=alt.X("v1:N", sort=orden), y=alt.Y("v2:N", sort=orden),
        text=alt.Text("corr:Q", format=".2f"),
        color=alt.condition("abs(datum.corr) > 0.55", alt.value("white"), alt.value(INK)))
    return (heat + txt).properties(height=430)


# ==========================================================================
# Secciones (cada una abre con el hallazgo; @st.fragment = redibujo aislado)
# ==========================================================================
def sec_panorama(d):
    s = panorama_stats(d["dir"])
    kicker("Aire de la Ciudad de México · 2024–2026")
    st.title("En la CDMX, la hora del día manda sobre el aire mucho más que la temporada")
    lede("Con datos horarios de la red RAMA/SIMAT, este tablero responde qué gobierna "
         "el ozono de la ciudad. La respuesta tiene tres partes y una sorpresa: el "
         "ozono sigue un <b>ciclo diario brutal</b> —se multiplica por más de veinte "
         "entre el amanecer y media tarde—, las estaciones se separan por su "
         "<b>química</b> más que por su “nivel”, y la <b>temporada</b>, en el centro, "
         "apenas mueve la aguja: el patrón anual que uno esperaría resulta "
         "sorprendentemente débil.")

    stat_row([
        (f"{s['o3_max']/s['o3_min']:.0f}", "×", "sube el ozono del amanecer a media tarde"),
        (f"{s['hora_pico']:02d}", "h", "hora pico del ozono"),
        (f"{s['pct_buena']:.0f}", "%", f"de días con ozono “Buena” en {s['principal']}"),
        (f"{s['n_est']}", "", "estaciones de monitoreo analizadas"),
    ])

    rule()
    kicker("De un vistazo")
    st.markdown(f"#### Tres años de calidad del aire por ozono en {s['principal']}, día a día")
    aqi_key()
    show(chart_timeline(serie_categoria(d["dir"], load_resumen(d["dir"])["estacion_principal"])))
    note("Cada franja es un día; el color, la categoría de calidad del aire por ozono "
         "(del máximo diario). Los días de mala calidad aparecen <b>repartidos a lo "
         "largo del año</b>, con apenas un ligero predominio en la primavera seca: en "
         "el centro, el ozono no tiene una temporada “mala” nítida. El patrón fuerte "
         "no es anual, sino diario —y es lo siguiente que muestra el tablero.")

    with st.expander("Sobre los datos", icon=":material/info:"):
        h = d["horario"].copy()
        h["anio"] = h["fecha"].dt.year
        cob = (h.groupby("anio")[POLLUTANTS].apply(lambda x: x.notna().mean() * 100)
               .round(0).astype(int))
        st.markdown(
            "Fuente: **Red Automática de Monitoreo Atmosférico (RAMA/SIMAT)** del "
            "Valle de México, datos horarios 2024–2026, integrados y limpiados "
            "(interpolación lineal de huecos cortos, marcado de valores extremos). "
            "La estación central de referencia para las series horarias es "
            f"**{s['principal']}**; los mapas usan las {d['pca']['estacion'].nunique()} "
            "estaciones con cobertura suficiente. La cobertura es parcial (sobre todo "
            "en 2026), lo que conviene tener presente al leer las tendencias.")
        st.caption("Cobertura por año (% de horas con dato válido en la estación central):")
        st.dataframe(cob, width='stretch')


def sec_reloj(d):
    s = panorama_stats(d["dir"])
    kicker("¿A qué hora?")
    st.header("El ozono se multiplica por veinte entre el amanecer y media tarde")
    lede(f"Este es el patrón más fuerte de todos. El ozono se forma con el sol: en "
         f"{s['principal']} pasa de apenas {s['o3_min']:.0f} ppb al amanecer a "
         f"{s['o3_max']:.0f} ppb hacia las {s['hora_pico']:02d} h. Los contaminantes "
         f"del tráfico, en cambio, pican en la hora pico de la mañana y se diluyen "
         f"después. Elige un contaminante para ver su pulso diario.")

    @st.fragment
    def cuerpo():
        cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
        u = UNITS[cont]
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown(f"**El día de {cont}** · promedio por hora, banda Q1–Q3")
            show(chart_diurno(diurno_abs(d["dir"], cont), cont, u))
        with c2:
            st.markdown("**¿Quién pica y cuándo?** · todos, a escala comparable")
            show(chart_diurno_norm(d["diurno_norm"], cont))
            note("Cada curva es el ciclo diario de un contaminante normalizado; "
                 "resaltado, el que elegiste. Se ve el desfase entre el tráfico "
                 "(mañana) y el ozono (tarde).")

        with st.expander("La evidencia espectral (transformada de Fourier)",
                         icon=":material/graphic_eq:"):
            anios = sorted(d["horario"]["fecha"].dt.year.unique())
            ay = st.selectbox("Año de análisis", anios, index=0, key="esp_y")
            res = espectro_cache(d["dir"], cont, ay)
            if res is None:
                st.warning("No hay suficientes datos continuos en esa combinación.",
                           icon=":material/warning:")
            else:
                spec, picos, umbral = res
                show(chart_espectro(spec, picos, umbral))
                note("La transformada descompone la serie en sus ondas. Los picos "
                     "en 24 h y 12 h confirman, de forma objetiva, el ciclo diario "
                     "y su armónico (la curva sube rápido y baja lento).")
    cuerpo()


def sec_anio(d):
    r = d["resumen"]
    kicker("¿Y la temporada?")
    st.header("La temporada apenas mueve la aguja en el centro")
    lede("Uno esperaría que la primavera seca —marzo a mayo, con más sol y menos "
         "lluvia— disparara el ozono. En la estación central el efecto es "
         "sorprendentemente tenue: el promedio mensual es casi plano y los días de "
         "mala calidad solo suben un poco en la seca-caliente. Por eso, al probar "
         "formalmente si la temporada decide la categoría de calidad del aire, la "
         "respuesta es que no de forma estadísticamente clara —y no es un error, es "
         "el hallazgo.")

    @st.fragment
    def cuerpo():
        long, res = estacional_categoria(d["dir"], r["estacion_principal"])
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("**Calidad del aire por ozono, según la temporada**")
            aqi_key()
            show(chart_estacional_cat(long))
            note("Las tres temporadas reparten los días de forma parecida; la "
                 "seca-caliente tiene algo más de días “Mala”, pero la mayoría sigue "
                 "siendo “Buena/Aceptable” todo el año.")
        with c2:
            cont = st.selectbox("Promedio mensual de", POLLUTANTS,
                                index=POLLUTANTS.index("O3"))
            st.markdown(f"**{cont} mes a mes** · en naranja, la seca-caliente")
            show(chart_mensual(mensual(d["dir"], cont), cont, UNITS[cont]))
            if cont == "O3":
                note("El promedio mensual de ozono apenas se mueve a lo largo del "
                     "año: no hay una temporada que sobresalga con claridad.")

        with st.expander("¿La temporada decide la calidad? · prueba χ² de independencia",
                         icon=":material/function:"):
            if res is not None:
                chi2, p, dof, _ = res
                cc = st.columns(3)
                cc[0].metric("χ²", f"{chi2:.2f}")
                cc[1].metric("p-valor", f"{p:.3f}")
                cc[2].metric("gl", f"{dof}")
                if p < 0.05:
                    st.success("p < 0.05 → se rechaza la independencia: la categoría "
                               "depende de la temporada.", icon=":material/check_circle:")
                else:
                    st.warning("p ≥ 0.05 → no se rechaza la independencia. En el centro, "
                               "la categoría de calidad por ozono no depende de forma "
                               "estadísticamente clara de la temporada.",
                               icon=":material/info:")
            note("La prueba confirma lo que muestran las barras: las diferencias entre "
                 "temporadas son demasiado pequeñas para ser concluyentes. Además, con "
                 "cobertura parcial (pocos días repartidos en tres años) la prueba "
                 "tiene poca potencia. El efecto estacional del ozono está documentado "
                 "y es más nítido en el surponiente elevado; en el centro, con estos "
                 "datos, no se distingue.")
    cuerpo()


def sec_mapa(d):
    r = d["resumen"]
    var2 = (r["varianza_pc1"] + r["varianza_pc2"]) * 100
    kicker("¿En qué zona?")
    st.header("Las estaciones no se ordenan por “nivel”, sino por química")
    lede("Si se resume a cada estación por su mezcla de contaminantes, el mapa que "
         "emerge no separa “limpias” de “sucias”: separa <b>regímenes químicos</b>. "
         "A la izquierda, las estaciones dominadas por contaminantes primarios del "
         "tráfico e industria; a la derecha, las dominadas por ozono; y un eje "
         "aparte aísla el corredor industrial del norte por su SO₂.")

    @st.fragment
    def cuerpo():
        pca = d["pca"].copy()
        pca["zona_nombre"] = pca["zona"].map(ZONE_NAMES).fillna(pca["zona"])
        show(chart_mapa(pca, d["cargas"]))
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            note("Cada punto es una estación (color por zona); las flechas son los "
                 "contaminantes y apuntan hacia donde ese contaminante crece. "
                 "Estaciones cercanas tienen mezclas parecidas. El eje horizontal "
                 f"explica el grueso de las diferencias entre estaciones "
                 f"({r['varianza_pc1']*100:.0f}%); con el vertical suman {var2:.0f}%.")
        with c2:
            note("Lectura: el eje horizontal opone primarios (CO, NO, NOₓ, a la "
                 "izquierda) contra ozono (derecha) —el ozono se forma corriente "
                 "abajo de donde se emiten sus precursores—. El vertical lo manda "
                 "el SO₂. (El escalamiento multidimensional —MDS— sobre los mismos "
                 "datos reproduce este mapa, lo que respalda la lectura.)")
    cuerpo()


def sec_relaciones(d):
    kicker("¿Cómo se relacionan?")
    st.header("Los contaminantes que comparten origen se mueven juntos")
    lede("La última pieza: ¿qué contaminantes suben y bajan a la vez? Las relaciones "
         "tienen sentido físico —no son coincidencia—. Cambia el coeficiente para "
         "ver relaciones lineales (Pearson) o monótonas y robustas a valores "
         "extremos (Spearman, Kendall).")

    @st.fragment
    def cuerpo():
        metodo = st.segmented_control("Coeficiente",
                                      ["pearson", "spearman", "kendall"], default="pearson")
        if metodo is None:
            metodo = "pearson"
        long, orden = corr_data(d["dir"], metodo)
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            show(chart_corr(long, metodo, orden))
        with c2:
            note("Filas y columnas reordenadas por agrupamiento: los contaminantes "
                 "parecidos quedan juntos y forman bloques. Azul = relación "
                 "negativa, rojo = positiva.")
            note("Dos relaciones que cuentan la historia: <b>O₃ ↔ NO₂ negativa</b> "
                 "(al formarse el ozono se consume su precursor) y <b>PM10 ↔ NO₂ "
                 "positiva</b> (ambos salen del tráfico). Correlación no implica "
                 "causalidad, pero aquí la química la respalda.")
        rule()
        note("<b>En síntesis.</b> El aire de la CDMX se entiende por la <b>hora</b> "
             "(un ciclo diario que multiplica el ozono por más de veinte), por el "
             "<b>espacio</b> (estaciones agrupadas por química, no por nivel) y por "
             "las <b>relaciones</b> físicas entre contaminantes. La <b>temporada</b>, "
             "en cambio, casi no influye en el centro —y la prueba estadística lo "
             "confirma—. Lo que parecía la pieza obvia, la estacionalidad, resultó la "
             "más débil; lo revelador estaba en el reloj diario y en la geografía "
             "química.")
    cuerpo()


# ==========================================================================
# App
# ==========================================================================
SECCIONES = {
    "Panorama": sec_panorama,
    "El reloj del día": sec_reloj,
    "El año del ozono": sec_anio,
    "El mapa químico de la ciudad": sec_mapa,
    "Lo que va junto": sec_relaciones,
}


def main():
    st.set_page_config(page_title="Aire CDMX", page_icon=":material/air:",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown("<div class='kicker'>Aire CDMX</div>", unsafe_allow_html=True)
    st.sidebar.markdown("#### ¿Qué gobierna el aire de la ciudad?")
    st.sidebar.caption("RAMA/SIMAT · 2024–2026")
    st.sidebar.markdown("<hr class='rule'>", unsafe_allow_html=True)
    eleccion = st.sidebar.radio("Recorrido", list(SECCIONES.keys()),
                                label_visibility="collapsed")

    with st.sidebar.expander("Opciones de datos", icon=":material/tune:"):
        propuesta = st.text_input("Ruta de datos", DATA_DIR_DEFAULT)

    data_dir = resolver_data_dir(propuesta)
    if data_dir is None:
        st.error(f"No encontré los datos. Debe existir `{propuesta}/resumen.json` "
                 "y los demás CSV junto a `app.py`.", icon=":material/error:")
        st.stop()

    try:
        d = {"dir": str(data_dir),
             "horario": load_horario(str(data_dir)),
             "diario": load_diario(str(data_dir)),
             "pca": load_pca(str(data_dir)),
             "cargas": load_cargas(str(data_dir)),
             "resumen": load_resumen(str(data_dir)),
             "diurno_norm": diurno_norm(str(data_dir))}
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al leer los datos: {e}", icon=":material/error:")
        st.stop()

    SECCIONES[eleccion](d)


if __name__ == "__main__":
    main()
