"""
Calidad del aire de la Ciudad de México (RAMA/SIMAT, 2024-2026)
================================================================
Proyecto de Analítica y Visualización de Datos.

Tablero interactivo construido con Streamlit + Altair (ambos incluidos, sin
dependencias extra). Lee los productos del análisis desde `datos/procesados/`.

Diseño: estética de producto de datos (gramática de gráficos, color con
propósito, títulos que enuncian el hallazgo). Rendimiento: cada sección
interactiva se recalcula y redibuja por separado con `@st.fragment`, los
cómputos pesados se memorizan con `@st.cache_data` y las series densas se
submuestrean, de modo que un clic NO recarga toda la página.

Para correr localmente:  streamlit run app.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2 as chi2_dist
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()

# --------------------------------------------------------------------------
# Constantes y paleta (color con propósito; categórica accesible Okabe-Ito)
# --------------------------------------------------------------------------
DATA_DIR_DEFAULT = "datos/procesados"

POLLUTANTS = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "PMCO", "SO2"]
GASES = ["CO", "NO", "NO2", "NOX", "O3", "SO2"]          # los que entran al PCA
UNITS = {"CO": "ppm", "NO": "ppb", "NO2": "ppb", "NOX": "ppb", "O3": "ppb",
         "PM10": "µg/m³", "PM2.5": "µg/m³", "PMCO": "µg/m³", "SO2": "ppb"}

ZONE_NAMES = {"NE": "Nororiente", "NO": "Noroeste", "CE": "Centro",
              "SO": "Surponiente", "SE": "Sureste"}
# Okabe-Ito: seguro para daltonismo. CE (centro, foco del análisis) = azul ancla.
ZONE_COLORS = {"CE": "#0072B2", "NE": "#D55E00", "NO": "#E69F00",
               "SO": "#009E73", "SE": "#56B4E9"}

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASON_ORDER = ["Seca-fría", "Seca-caliente", "Lluvias"]
CAT_ORDER = ["Buena", "Aceptable", "Mala", "Muy mala"]
CAT_COLORS = ["#2C7FB8", "#7FCDBB", "#FEB24C", "#BD0026"]   # secuencial cualitativa O3

# Tokens de acento (legibles en claro y oscuro)
ACCENT = "#0072B2"      # azul: serie resaltada / componente 1
ACCENT2 = "#D55E00"     # vermellón: segundo plano / componente 2
HILITE = "#E69F00"      # ámbar: temporada seca-caliente
MUTED = "#C2C9D1"       # gris para series atenuadas (Tufte: resaltar una)
INK = "#5B6675"         # texto de anotación neutro (sirve en claro y oscuro)
GRID = "#CBD5E1"
SEQ_SCHEME = "viridis"  # secuencial perceptualmente uniforme
DIV_RANGE = ["#2166AC", "#F7F7F7", "#B2182B"]   # divergente centrado en 0

# Filtros de suavizado (Okabe-Ito, distinguibles)
F_RAW = "#C2C9D1"
F_MEDIA = "#0072B2"
F_MEDIANA = "#D55E00"
F_EXPO = "#009E73"


# --------------------------------------------------------------------------
# Helpers de interacción: pan SIN zoom de rueda (evita que la rueda mueva la
# página, que era el comportamiento molesto reportado)
# --------------------------------------------------------------------------
def pan_x():
    """Arrastrar para desplazar en X; sin zoom de rueda."""
    return alt.selection_interval(bind="scales", encodings=["x"], zoom=False)


def pan_xy():
    """Arrastrar para desplazar en 2D; sin zoom de rueda."""
    return alt.selection_interval(bind="scales", zoom=False)


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
    chi2, p, dof, esp = chi2_contingency(t.values, correction=False)
    return chi2, p, dof, t, esp


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
# Carga de datos (memorizada)
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
# Analítica memorizada (se computa una vez por entrada y luego es instantánea)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def corr_matrix(data_dir: str, metodo: str) -> pd.DataFrame:
    h = load_horario(data_dir)
    cols = [c for c in POLLUTANTS if h[c].notna().sum() > 100]
    return h[cols].corr(method=metodo)


@st.cache_data(show_spinner=False)
def corr_order(data_dir: str, metodo: str) -> list:
    """Orden de variables por clustering jerárquico (revela bloques)."""
    corr = corr_matrix(data_dir, metodo)
    if corr.shape[0] < 3:
        return list(corr.columns)
    d = 1.0 - corr.abs().values
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    return [corr.columns[i] for i in leaves_list(Z)]


@st.cache_data(show_spinner=False)
def chi_todas_estaciones(data_dir: str) -> pd.DataFrame:
    diario = load_diario(data_dir)
    o3 = diario[diario["parametro"] == "O3"]
    filas = []
    for e, sub in o3.groupby("estacion"):
        if len(sub) < 120:
            continue
        r2 = chi_desde_tabla(tabla_contingencia(sub))
        if r2 is None:
            continue
        chi2, p, dof, _, _ = r2
        filas.append({"estación": e, "n_días": len(sub), "χ²": round(chi2, 2),
                      "gl": dof, "p": round(p, 3),
                      "¿rechaza H₀?": "sí" if p < 0.05 else "no"})
    return pd.DataFrame(filas).sort_values("p")


@st.cache_data(show_spinner=False)
def pca_mds(data_dir: str) -> dict:
    """
    Reconstruye la varianza completa del PCA y calcula el MDS espectral desde
    la matriz estación x gas (mismo diseño del notebook). Para mantener la
    consistencia EXACTA con lo presentado, PC1/PC2 se anclan a resumen.json y
    el biplot usa las coordenadas y cargas exportadas; el MDS se orienta
    (reflexiones) para que sea comparable con el mapa PCA.
    """
    diario = load_diario(data_dir)
    res = load_resumen(data_dir)
    pca_e = load_pca(data_dir).set_index("estacion")

    matriz = diario.pivot_table(index="estacion", columns="parametro",
                                values="promedio", aggfunc="mean")
    matriz = matriz.dropna(axis=1, thresh=int(len(matriz) * 0.8)).dropna(axis=0)
    X = matriz.values.astype(float)
    X_std = (X - X.mean(0)) / X.std(0, ddof=0)
    n = X_std.shape[0]

    # --- varianza por componente (eigenvalores de la covarianza) ---
    cov = (X_std.T @ X_std) / (n - 1)
    lams = np.sort(np.linalg.eigvalsh(cov))[::-1]
    var_exp = lams / lams.sum()
    # anclar PC1, PC2 a lo exportado; repartir el resto por proporción real
    v = var_exp.copy()
    v[0], v[1] = res["varianza_pc1"], res["varianza_pc2"]
    resto = max(0.0, 1.0 - v[0] - v[1])
    cola = var_exp[2:]
    v[2:] = resto * (cola / cola.sum()) if cola.sum() > 0 else 0.0
    acum = np.cumsum(v)
    q90 = int(np.argmax(acum >= 0.9)) + 1
    scree = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(v))],
                          "varianza": v, "acumulada": acum})

    # --- MDS espectral (descomposición de XX^T sobre datos centrados) ---
    Xc = X_std - X_std.mean(0)
    B = Xc @ Xc.T
    vals, vecs = np.linalg.eigh(B)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    Ymds = vecs[:, :2] @ np.diag(np.sqrt(np.maximum(vals[:2], 0)))
    mds = pd.DataFrame(Ymds, index=matriz.index, columns=["M1", "M2"])
    # alinear orientación (signo) con el PCA exportado para lectura comparable
    common = mds.index.intersection(pca_e.index)
    for i, pcc in enumerate(["PC1", "PC2"]):
        a = mds.loc[common, mds.columns[i]].values
        b = pca_e.loc[common, pcc].values
        if np.corrcoef(a, b)[0, 1] < 0:
            mds.iloc[:, i] = -mds.iloc[:, i]
    mds["nombre"] = pca_e["nombre"].reindex(mds.index).values
    mds["zona"] = pca_e["zona"].reindex(mds.index).values

    # --- calidad del MDS: distorsión de distancias y stress de Kruskal ---
    dx = pdist(X_std)
    dy = pdist(mds[["M1", "M2"]].values)
    stress1 = float(np.sqrt(((dy - dx) ** 2).sum() / (dx ** 2).sum()))
    dist = pd.DataFrame({"dx": dx, "delta": dy - dx})
    dist = dist.iloc[:: max(1, len(dist) // 1500)]   # ligero para la gráfica

    return {"scree": scree, "q90": q90, "mds": mds.reset_index(),
            "stress1": stress1, "dist": dist, "n_est": int(n)}


@st.cache_data(show_spinner=False)
def diurno_normalizado(data_dir: str) -> pd.DataFrame:
    """Ciclo diurno z-normalizado de cada contaminante (timing comparable)."""
    h = load_horario(data_dir).copy()
    h["hora"] = h["fecha"].dt.hour
    out = []
    for c in POLLUTANTS:
        m = h.groupby("hora")[c].mean()
        if m.notna().sum() < 12 or m.std(skipna=True) == 0:
            continue
        z = (m - m.mean()) / m.std()
        out.append(pd.DataFrame({"hora": z.index, "contaminante": c, "z": z.values}))
    return pd.concat(out, ignore_index=True)


def _downsample(df: pd.DataFrame, max_points: int = 3000) -> pd.DataFrame:
    if len(df) > max_points:
        step = int(np.ceil(len(df) / max_points))
        return df.iloc[::step]
    return df


# --------------------------------------------------------------------------
# Constructores de gráficas Altair (interactivas, rediseñadas)
# --------------------------------------------------------------------------
def chart_cobertura(cob_long: pd.DataFrame):
    base = alt.Chart(cob_long)
    heat = base.mark_rect().encode(
        x=alt.X("contaminante:N", title=None, sort=POLLUTANTS),
        y=alt.Y("anio:O", title="año"),
        color=alt.Color("pct:Q", title="% válido",
                        scale=alt.Scale(scheme=SEQ_SCHEME, domain=[0, 100])),
        tooltip=[alt.Tooltip("contaminante:N", title="contaminante"),
                 alt.Tooltip("anio:O", title="año"),
                 alt.Tooltip("pct:Q", title="% válido", format=".1f")],
    )
    texto = base.mark_text(fontSize=11, fontWeight=600).encode(
        x=alt.X("contaminante:N", sort=POLLUTANTS), y="anio:O",
        text=alt.Text("pct:Q", format=".0f"),
        color=alt.condition("datum.pct > 55", alt.value("white"), alt.value("#1E293B")),
    )
    return (heat + texto).properties(height=190)


def chart_diurno(dfm: pd.DataFrame, cont: str, unit: str):
    base = alt.Chart(dfm)
    band = base.mark_area(opacity=0.18, color=ACCENT).encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 3)))),
        y=alt.Y("q1:Q", title=f"{cont} ({unit})"), y2="q3:Q")
    media_global = float(dfm["media"].mean())
    ref = alt.Chart(pd.DataFrame({"y": [media_global]})).mark_rule(
        color=INK, strokeDash=[3, 3], opacity=0.7).encode(y="y:Q")
    linea = base.mark_line(color=ACCENT, strokeWidth=2.6,
                           point=alt.OverlayMarkDef(color=ACCENT, size=28)).encode(
        x="hora:Q", y="media:Q",
        tooltip=[alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("media:Q", title="promedio", format=".1f"),
                 alt.Tooltip("q1:Q", title="Q1", format=".1f"),
                 alt.Tooltip("q3:Q", title="Q3", format=".1f")])
    return (band + ref + linea).properties(height=300)


def chart_diurno_norm(dfn: pd.DataFrame, cont: str):
    """Todas las curvas diurnas normalizadas; la elegida resaltada (Tufte)."""
    test = f"datum.contaminante === '{cont}'"
    return alt.Chart(dfn).mark_line().encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 3)))),
        y=alt.Y("z:Q", title="nivel relativo (z)"),
        detail="contaminante:N",
        color=alt.condition(test, alt.value(ACCENT), alt.value(MUTED)),
        size=alt.condition(test, alt.value(2.8), alt.value(1.0)),
        opacity=alt.condition(test, alt.value(1.0), alt.value(0.5)),
        tooltip=[alt.Tooltip("contaminante:N", title="contaminante"),
                 alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("z:Q", title="nivel (z)", format=".2f")],
    ).properties(height=300)


def chart_estacional(dfm: pd.DataFrame, cont: str, unit: str):
    return alt.Chart(dfm).mark_bar().encode(
        x=alt.X("mes:N", title=None, sort=MESES),
        y=alt.Y("valor:Q", title=f"{cont} ({unit}) — promedio"),
        color=alt.Color("seca_caliente:N",
                        scale=alt.Scale(domain=["Seca-caliente", "Resto del año"],
                                        range=[HILITE, MUTED]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("mes:N", title="mes"),
                 alt.Tooltip("valor:Q", title="promedio", format=".1f")],
    ).properties(height=300)


def chart_filtros(df_long: pd.DataFrame, cont: str, unit: str):
    sel = alt.selection_point(fields=["serie"], bind="legend")
    escala = alt.Scale(
        domain=["Señal cruda", "Media móvil", "Mediana móvil", "Exponencial"],
        range=[F_RAW, F_MEDIA, F_MEDIANA, F_EXPO])
    return alt.Chart(df_long).mark_line(clip=True).encode(
        x=alt.X("fecha:T", title=None),
        y=alt.Y("valor:Q", title=f"{cont} ({unit})"),
        color=alt.Color("serie:N", scale=escala,
                        legend=alt.Legend(title="serie (clic para resaltar)", orient="top")),
        size=alt.condition(sel, alt.value(2.2), alt.value(0.9)),
        opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
        tooltip=[alt.Tooltip("fecha:T", title="fecha"),
                 alt.Tooltip("serie:N", title="serie"),
                 alt.Tooltip("valor:Q", title="valor", format=".1f")],
    ).properties(height=330).add_params(sel, pan_x())


def chart_espectro(spec_df, umbral, picos_df):
    x = alt.X("periodo:Q", scale=alt.Scale(type="log"),
              title="periodo (horas) — escala logarítmica")
    linea = alt.Chart(spec_df).mark_line(color=ACCENT, strokeWidth=1.5).encode(
        x=x, y=alt.Y("amplitud:Q", title="amplitud"),
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", title="amplitud", format=".3f")])
    regla = alt.Chart(pd.DataFrame({"y": [umbral]})).mark_rule(
        color=INK, strokeDash=[5, 4]).encode(y="y:Q")
    marcas = alt.Chart(pd.DataFrame({"periodo": [24, 12], "etq": ["24 h", "12 h"]}))
    reglas_v = marcas.mark_rule(color=INK, strokeDash=[2, 3], opacity=0.8).encode(x=x)
    etiquetas = marcas.mark_text(align="left", dx=4, dy=-6, fontSize=11,
                                 fontWeight=600, color=INK).encode(x=x, text="etq:N")
    puntos = alt.Chart(picos_df).mark_point(
        color=ACCENT2, size=90, filled=True).encode(
        x=x, y="amplitud:Q",
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", title="amplitud", format=".3f")])
    return (linea + regla + reglas_v + etiquetas + puntos).properties(
        height=340).add_params(pan_x())


def _zone_scale(zonas_presentes):
    zonas = [z for z in ZONE_COLORS if z in set(zonas_presentes)]
    return alt.Scale(domain=zonas, range=[ZONE_COLORS[z] for z in zonas])


def chart_biplot(pca_df, cargas_df):
    escala = _zone_scale(pca_df["zona"])
    # escalar los vectores de carga para que llenen ~70% de la nube de puntos
    rad_pts = float(np.nanmax(np.abs(pca_df[["PC1", "PC2"]].values)))
    rad_load = float(np.nanmax(np.abs(cargas_df[["PC1", "PC2"]].values)))
    k = (rad_pts / rad_load) * 0.72 if rad_load > 0 else 1.0
    flechas = cargas_df.copy()
    flechas["x"] = flechas["PC1"] * k
    flechas["y"] = flechas["PC2"] * k

    ejes = (alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=GRID).encode(y="z:Q")
            + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=GRID).encode(x="z:Q"))
    # vector de carga: segmento del origen (0,0) a (x,y)
    vec = alt.Chart(flechas).mark_rule(color=INK, strokeWidth=1.5, opacity=0.65).encode(
        x=alt.datum(0), y=alt.datum(0), x2="x:Q", y2="y:Q")
    vlab = alt.Chart(flechas).mark_text(color=ACCENT2, fontWeight=700, fontSize=12).encode(
        x="x:Q", y="y:Q", text="parametro:N")
    base = alt.Chart(pca_df)
    pts = base.mark_circle(size=150, opacity=0.92, stroke="white", strokeWidth=1).encode(
        x=alt.X("PC1:Q", title="PC1 (~73%):  primarios (tráfico/industria)  ←→  ozono"),
        y=alt.Y("PC2:Q", title="PC2 (~14%):  SO₂ ↑  (corredor industrial norte)"),
        color=alt.Color("zona:N", scale=escala, legend=alt.Legend(title="zona")),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("zona_nombre:N", title="zona"),
                 alt.Tooltip("PC1:Q", format=".2f"),
                 alt.Tooltip("PC2:Q", format=".2f")])
    plab = base.mark_text(dy=-13, fontSize=10, color=INK).encode(
        x="PC1:Q", y="PC2:Q", text="estacion:N")
    return (ejes + vec + vlab + pts + plab).properties(height=470).add_params(pan_xy())


def chart_scree(scree, q90):
    base = alt.Chart(scree).encode(
        x=alt.X("PC:N", sort=list(scree["PC"]), title=None))
    barras = base.mark_bar(color=ACCENT, opacity=0.85, size=34).encode(
        y=alt.Y("varianza:Q", axis=alt.Axis(format="%", title="varianza explicada")),
        tooltip=[alt.Tooltip("PC:N"),
                 alt.Tooltip("varianza:Q", title="varianza", format=".1%"),
                 alt.Tooltip("acumulada:Q", title="acumulada", format=".1%")])
    val_text = base.mark_text(dy=-6, fontSize=10, color=INK).encode(
        y="varianza:Q", text=alt.Text("varianza:Q", format=".0%"))
    linea = base.mark_line(color=ACCENT2, strokeWidth=2,
                           point=alt.OverlayMarkDef(color=ACCENT2)).encode(
        y=alt.Y("acumulada:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="acumulada")))
    regla90 = alt.Chart(pd.DataFrame({"y": [0.9]})).mark_rule(
        color=INK, strokeDash=[4, 4]).encode(y="y:Q")
    t90 = alt.Chart(pd.DataFrame({"y": [0.9], "t": ["90%"]})).mark_text(
        align="left", dx=4, dy=-4, color=INK, fontSize=11).encode(y="y:Q", text="t:N")
    izq = (barras + val_text)
    der = (linea + regla90 + t90)
    return alt.layer(izq, der).resolve_scale(y="independent").properties(height=320)


def chart_cargas(cargas_long):
    sign_scale = alt.Scale(domain=[-1, 0, 1], range=DIV_RANGE)
    paneles = []
    for comp in ["PC1", "PC2"]:
        sub = cargas_long[cargas_long["componente"] == comp]
        barras = alt.Chart(sub).mark_bar().encode(
            y=alt.Y("parametro:N", title=None, sort=GASES),
            x=alt.X("carga:Q", title="carga", scale=alt.Scale(domain=[-0.7, 1.0])),
            color=alt.Color("carga:Q", scale=sign_scale, legend=None),
            tooltip=[alt.Tooltip("parametro:N", title="contaminante"),
                     alt.Tooltip("componente:N", title="componente"),
                     alt.Tooltip("carga:Q", format=".3f")])
        cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GRID).encode(x="x:Q")
        umb = alt.Chart(pd.DataFrame({"x": [-0.3, 0.3]})).mark_rule(
            color=INK, strokeDash=[2, 3], opacity=0.6).encode(x="x:Q")
        paneles.append((cero + umb + barras).properties(height=210, width=280, title=comp))
    return alt.hconcat(*paneles)


def chart_mds(mds_df):
    escala = _zone_scale(mds_df["zona"])
    ejes = (alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=GRID).encode(y="z:Q")
            + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=GRID).encode(x="z:Q"))
    base = alt.Chart(mds_df)
    pts = base.mark_circle(size=150, opacity=0.92, stroke="white", strokeWidth=1).encode(
        x=alt.X("M1:Q", title="dimensión 1 del MDS"),
        y=alt.Y("M2:Q", title="dimensión 2 del MDS"),
        color=alt.Color("zona:N", scale=escala, legend=alt.Legend(title="zona")),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("nombre:N", title="nombre"),
                 alt.Tooltip("M1:Q", format=".2f"),
                 alt.Tooltip("M2:Q", format=".2f")])
    lab = base.mark_text(dy=-13, fontSize=10, color=INK).encode(
        x="M1:Q", y="M2:Q", text="estacion:N")
    return (ejes + pts + lab).properties(height=460).add_params(pan_xy())


def chart_distorsion(dist_df):
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=ACCENT2, strokeDash=[4, 3]).encode(y="y:Q")
    pts = alt.Chart(dist_df).mark_circle(size=22, opacity=0.4, color=ACCENT).encode(
        x=alt.X("dx:Q", title="distancia real entre estaciones (dˣ)"),
        y=alt.Y("delta:Q", title="distorsión al proyectar (dʸ − dˣ)"),
        tooltip=[alt.Tooltip("dx:Q", format=".2f"),
                 alt.Tooltip("delta:Q", format=".2f")])
    return (cero + pts).properties(height=300)


def chart_corr(corr_long, metodo, orden):
    base = alt.Chart(corr_long)
    heat = base.mark_rect().encode(
        x=alt.X("v1:N", title=None, sort=orden),
        y=alt.Y("v2:N", title=None, sort=orden),
        color=alt.Color("corr:Q", title=f"r ({metodo})",
                        scale=alt.Scale(domain=[-1, 0, 1], range=DIV_RANGE)),
        tooltip=[alt.Tooltip("v1:N", title=""), alt.Tooltip("v2:N", title=""),
                 alt.Tooltip("corr:Q", title="correlación", format=".2f")])
    texto = base.mark_text(fontSize=10).encode(
        x=alt.X("v1:N", sort=orden), y=alt.Y("v2:N", sort=orden),
        text=alt.Text("corr:Q", format=".2f"),
        color=alt.condition("abs(datum.corr) > 0.55",
                            alt.value("white"), alt.value("#1E293B")))
    return (heat + texto).properties(height=430)


def chart_splom(df_pts, variables):
    return alt.Chart(df_pts).mark_circle(size=14, opacity=0.35, color=ACCENT).encode(
        x=alt.X(alt.repeat("column"), type="quantitative"),
        y=alt.Y(alt.repeat("row"), type="quantitative"),
    ).properties(width=120, height=120).repeat(row=variables, column=variables)


def chart_mosaico(mos_df):
    rects = alt.Chart(mos_df).mark_rect(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("x0:Q", title="temporada  (ancho ∝ nº de días)",
                axis=alt.Axis(labels=False, ticks=False)),
        x2="x1:Q",
        y=alt.Y("y0:Q", title="categoría de calidad por O₃  (alto ∝ proporción)",
                axis=alt.Axis(labels=False, ticks=False)),
        y2="y1:Q",
        color=alt.Color("residuo:Q", title="residuo estandarizado",
                        scale=alt.Scale(domain=[-3, 0, 3], range=DIV_RANGE)),
        tooltip=[alt.Tooltip("temporada:N", title="temporada"),
                 alt.Tooltip("categoria:N", title="categoría"),
                 alt.Tooltip("observado:Q", title="días observados"),
                 alt.Tooltip("esperado:Q", title="esperados (si H₀)", format=".1f"),
                 alt.Tooltip("residuo:Q", title="residuo", format=".2f")])
    etq_temp = alt.Chart(mos_df.drop_duplicates("temporada")).mark_text(
        baseline="top", dy=2, fontSize=11, fontWeight=600, color=INK).encode(
        x=alt.X("xmid:Q"), y=alt.datum(1.0), text="temporada:N")
    return (rects + etq_temp).properties(height=360)


def chart_chi_prop(prop_long):
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


def mosaico_df(tab: pd.DataFrame, esperado: np.ndarray) -> pd.DataFrame:
    """Construye los rectángulos del mosaico con residuos estandarizados."""
    obs = tab.values.astype(float)
    col_tot = obs.sum(axis=0)
    grand = obs.sum()
    filas, cols = list(tab.index), list(tab.columns)
    x_cursor = 0.0
    registros = []
    for j, c in enumerate(cols):
        w = col_tot[j] / grand if grand else 0
        x0, x1 = x_cursor, x_cursor + w
        x_cursor = x1
        col_sum = obs[:, j].sum()
        y_cursor = 0.0
        for i, f in enumerate(filas):
            h = obs[i, j] / col_sum if col_sum else 0
            y0, y1 = y_cursor, y_cursor + h
            y_cursor = y1
            e = esperado[i, j]
            r = (obs[i, j] - e) / np.sqrt(e) if e > 0 else 0.0
            registros.append({"temporada": c, "categoria": f,
                              "x0": x0, "x1": x1, "xmid": (x0 + x1) / 2,
                              "y0": 1 - y0, "y1": 1 - y1,   # invertir para que "Buena" quede arriba
                              "observado": int(obs[i, j]), "esperado": float(e),
                              "residuo": float(r)})
    return pd.DataFrame(registros)


# --------------------------------------------------------------------------
# Páginas
# --------------------------------------------------------------------------
def pagina_inicio(d):
    r = d["resumen"]
    st.title("El aire de la Ciudad de México tiene patrones, y aquí se ven")
    st.markdown(
        "Datos horarios de la **Red Automática de Monitoreo Atmosférico "
        "(RAMA/SIMAT)** del Valle de México, 2024–2026. Este tablero recorre las "
        "tres unidades del curso para responder una sola pregunta: "
        ":blue-badge[Unidad I] :blue-badge[Unidad II] :blue-badge[Unidad III]"
    )
    st.markdown(
        "> **¿Qué gobierna la contaminación del aire en la CDMX —según la hora, la "
        "temporada y la zona— y cómo se relacionan los contaminantes entre sí?**")

    with st.container(horizontal=True):
        st.metric("Estaciones analizadas", f"{d['pca']['estacion'].nunique()}")
        st.metric("Periodo", f"{min(r['anios'])}–{max(r['anios'])}")
        st.metric("Estación central", f"{r['nombre_estacion']} ({r['estacion_principal']})")
        st.metric("Varianza en 2 ejes (PCA)",
                  f"{(r['varianza_pc1'] + r['varianza_pc2'])*100:.0f}%")

    st.subheader("Cuatro respuestas, una por técnica")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**El aire corre con dos relojes.** El análisis espectral (Fourier) "
            "confirma un ciclo **diario de 24 h** —y su armónico de 12 h— más una "
            "**modulación estacional**. El ozono es, en esencia, periódico.")
        st.markdown(
            "**El espacio se ordena por química, no por nivel.** El PCA separa un "
            "núcleo de **primarios** (tráfico/industria) de una periferia de "
            "**ozono**, con un eje aparte de **SO₂** que marca el corredor "
            "industrial del norte. El MDS llega a la misma estructura.")
    with c2:
        st.markdown(
            "**Las relaciones tienen sentido físico.** O₃ ↔ NO₂ negativa (el "
            "precursor se consume al formarse el ozono); PM10 ↔ NO₂ positiva "
            "(ambos vienen del tráfico).")
        st.markdown(
            "**Un resultado que invita a pensar.** La prueba χ² **no** detecta "
            "dependencia entre la calidad del aire por ozono y la temporada en la "
            "estación central. No es un error: es un hallazgo que se explica "
            "(ver Unidad III).")
    st.caption("Usa el menú de la izquierda para entrar a cada unidad. Casi todas "
               "las gráficas son interactivas: pasa el cursor para ver valores y "
               "haz clic en las leyendas para resaltar.")


def pagina_datos(d):
    st.header(":material/database: Los datos")
    st.markdown(
        "Cada estación de la RAMA mide los **contaminantes criterio** cada hora. "
        "El notebook integró los archivos crudos (un Excel por año y contaminante), "
        "imputó huecos cortos por interpolación lineal, marcó valores extremos y "
        "exportó los CSV que alimentan este tablero.")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**Qué se mide**")
        tabla_cont = pd.DataFrame({
            "Contaminante": POLLUTANTS,
            "Unidad": [UNITS[p] for p in POLLUTANTS],
            "Tipo": ["Secundario (sol)" if p == "O3" else "Primario / mezcla"
                     for p in POLLUTANTS]})
        st.dataframe(tabla_cont, hide_index=True, height=350)
    with col2:
        st.markdown(f"**Hay huecos reales: cobertura en {d['resumen']['nombre_estacion']}**")
        h = d["horario"].copy()
        h["anio"] = h["fecha"].dt.year
        cob = h.groupby("anio")[POLLUTANTS].apply(lambda x: x.notna().mean() * 100)
        cob_long = (cob.round(1).reset_index()
                    .melt(id_vars="anio", var_name="contaminante", value_name="pct"))
        st.altair_chart(chart_cobertura(cob_long), use_container_width=True)
        st.caption("El color es el % de horas con dato válido. 2026 está poco "
                   "poblado y el PM tiene menos cobertura: por eso no entró al PCA.")

    with st.expander("Ver una muestra de la serie horaria (Merced)",
                     icon=":material/table_chart:"):
        st.dataframe(d["horario"].head(24), hide_index=True)


@st.fragment
def _ciclos_fragment(d):
    cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
    u = UNITS[cont]
    h = d["horario"].copy()
    h["hora"] = h["fecha"].dt.hour
    h["mes"] = h["fecha"].dt.month

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**El ciclo diario de {cont}** (promedio por hora, banda Q1–Q3)")
        g = h.groupby("hora")[cont]
        dfm = pd.DataFrame({"hora": g.mean().index, "media": g.mean().values,
                            "q1": g.quantile(0.25).values, "q3": g.quantile(0.75).values})
        st.altair_chart(chart_diurno(dfm, cont, u), use_container_width=True)
        if cont == "O3":
            st.caption("El ozono es secundario: se forma con el sol y llega a su "
                       "máximo a primera hora de la tarde.")
    with col2:
        st.markdown(f"**¿Cuándo manda {cont} frente a los demás?** (curvas normalizadas)")
        st.altair_chart(chart_diurno_norm(d["diurno_norm"], cont),
                        use_container_width=True)
        st.caption("Cada curva es el ciclo diario de un contaminante llevado a la "
                   "misma escala. Resaltado, el que elegiste; en gris, el resto. "
                   "Así se ve el desfase: los primarios pican en la hora pico "
                   "matutina; el ozono, a media tarde.")

    st.markdown(f"**La serie es ruidosa; los filtros revelan la tendencia de {cont}**")
    st.caption("Media y mediana con ventana q = 24 h; exponencial con θ = 0.05. "
               "Clic en la leyenda para resaltar una serie. Arrastra para "
               "desplazar en el tiempo (la rueda del ratón ya no mueve la página).")
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
            "Exponencial": pd.Series(filtro_exponencial(s.values, 0.05), index=s.index)})
        df_f = _downsample(df_f).reset_index().melt(
            id_vars="fecha", var_name="serie", value_name="valor").dropna()
        st.altair_chart(chart_filtros(df_f, cont, u), use_container_width=True)


def pagina_ciclos(d):
    st.header(":material/schedule: Unidad I · Los ciclos del aire")
    st.markdown("Antes de cualquier modelo, mirar el dato: ¿a qué hora y en qué "
                "meses sube cada contaminante, y qué tendencia queda al filtrar el ruido?")
    _ciclos_fragment(d)


@st.fragment
def _espectro_fragment(d):
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

    st.markdown(f"**El espectro de {cont} en Merced ({ay})** — amplitud por periodo")
    st.altair_chart(chart_espectro(spec_df, umbral, picos_df), use_container_width=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**Periodos dominantes**")
        top = (picos_df.sort_values("amplitud", ascending=False).head(8)
               .rename(columns={"periodo": "periodo (h)", "amplitud": "amplitud"}))
        top["periodo (h)"] = top["periodo (h)"].round(1)
        top["amplitud"] = top["amplitud"].round(3)
        st.dataframe(top, hide_index=True)
    with col2:
        if cont == "O3":
            st.info("En el ozono el pico está en **24 h** (el ciclo fotoquímico "
                    "diario) y reaparece en **12 h** (el armónico que corrige la "
                    "forma asimétrica de la curva: sube rápido, baja lento).",
                    icon=":material/lightbulb:")
        st.caption("Eje X en escala logarítmica. Las líneas punteadas marcan 24 h "
                   "y 12 h; los puntos naranjas son picos sobre el umbral μ+3σ.")


def pagina_espectro(d):
    st.header(":material/graphic_eq: Unidad II · Análisis espectral")
    st.markdown("La transformada de Fourier descompone la serie en las **ondas** que "
                "la forman y mide cuál pesa más. Confirma de forma objetiva los "
                "ciclos que se intuían en la exploración.")
    _espectro_fragment(d)


@st.fragment
def _pca_fragment(d):
    pm = d["pca_mds"]
    r = d["resumen"]
    pca = d["pca"].copy()
    pca["zona_nombre"] = pca["zona"].map(ZONE_NAMES).fillna(pca["zona"])
    cargas = d["cargas"]

    with st.container(horizontal=True):
        st.metric("Varianza PC1", f"{r['varianza_pc1']*100:.1f}%")
        st.metric("Varianza PC2", f"{r['varianza_pc2']*100:.1f}%")
        st.metric("Componentes para 90%", f"{pm['q90']}")
        st.metric("Stress del MDS", f"{pm['stress1']:.2f}")

    t_mapa, t_scree, t_cargas, t_mds = st.tabs(
        ["Mapa (biplot)", "Varianza (scree)", "Cargas", "MDS"])

    with t_mapa:
        st.markdown("**Las estaciones se agrupan por régimen químico**")
        st.altair_chart(chart_biplot(pca, cargas), use_container_width=True)
        st.caption("Puntos = estaciones (color por zona). Flechas = cargas de los "
                   "contaminantes: apuntan hacia donde ese contaminante crece. "
                   "Izquierda (primarios: CO, NO, NOX) ↔ derecha (ozono). "
                   "Arrastra para desplazar.")

    with t_scree:
        st.markdown("**Con tres componentes basta para el 90% de la varianza**")
        st.altair_chart(chart_scree(pm["scree"], pm["q90"]), use_container_width=True)
        st.caption("Barras = varianza de cada componente; línea = acumulada. El "
                   "plano PC1–PC2 ya resume ~87%; cruzar el 90% (línea punteada) "
                   "exige la tercera componente.")

    with t_cargas:
        st.markdown("**Qué contaminante pesa en cada eje**")
        cargas_long = cargas.melt(id_vars="parametro", value_vars=["PC1", "PC2"],
                                  var_name="componente", value_name="carga")
        st.altair_chart(chart_cargas(cargas_long))
        st.markdown(
            "- **PC1**: los primarios (CO, NO, NO₂, NOX) cargan **negativo** y el "
            "O₃ **positivo** → es un eje **primarios ↔ ozono**, no de \"nivel general\".\n"
            "- **PC2**: lo domina el **SO₂** → aísla el corredor industrial del "
            "norte (Tlalnepantla, Cuautitlán, Atizapán).\n"
            "- El **PM no entró** al PCA (lo descartó el filtro de cobertura).")
        st.caption("Líneas punteadas en ±0.3: umbral de carga 'importante' usado en "
                   "el análisis. Esta lectura corrige la redacción original del "
                   "notebook, que hablaba de 'nivel general' e incluía al PM.")

    with t_mds:
        st.markdown("**El MDS reconstruye el mismo mapa desde las distancias**")
        cmds1, cmds2 = st.columns([3, 2])
        with cmds1:
            st.altair_chart(chart_mds(pm["mds"]), use_container_width=True)
            st.caption("El MDS parte de las distancias entre estaciones, no de la "
                       "varianza. Aun así reproduce el agrupamiento del PCA: el MDS "
                       "espectral sobre una matriz de características es idéntico al "
                       "PCA salvo rotación/reflexión —y eso valida ambas implementaciones.")
        with cmds2:
            st.markdown("**¿Cuánto se distorsiona al proyectar a 2D?**")
            st.altair_chart(chart_distorsion(pm["dist"]), use_container_width=True)
            st.caption(f"Stress de Kruskal ≈ {pm['stress1']:.2f}. Δd ≤ 0 casi "
                       "siempre: al perder dimensiones las distancias solo pueden "
                       "encogerse, y las estaciones más distintas son las que más "
                       "se acortan.")


def pagina_pca(d):
    st.header(":material/scatter_plot: Unidad II · PCA y MDS: el mapa de las estaciones")
    st.markdown("Misma pregunta —¿qué estaciones se parecen?— por dos caminos: el "
                "PCA (desde la varianza) y el MDS (desde las distancias).")
    _pca_fragment(d)


@st.fragment
def _corr_fragment(d):
    st.markdown("**Cómo se relacionan los contaminantes** (serie horaria de Merced)")
    metodo = st.segmented_control(
        "Coeficiente", ["pearson", "spearman", "kendall"], default="pearson")
    if metodo is None:
        metodo = "pearson"
    corr = corr_matrix(d["dir"], metodo)
    orden = corr_order(d["dir"], metodo)
    # triángulo inferior (incl. diagonal) según el orden por clustering
    pos = {v: i for i, v in enumerate(orden)}
    corr_long = (corr.reset_index().melt(id_vars="index", var_name="v2",
                                         value_name="corr").rename(columns={"index": "v1"}))
    corr_long = corr_long[corr_long.apply(
        lambda r: pos.get(r["v1"], 0) >= pos.get(r["v2"], 0), axis=1)]
    st.altair_chart(chart_corr(corr_long, metodo, orden), use_container_width=True)
    st.caption("Filas y columnas reordenadas por agrupamiento jerárquico (los "
               "contaminantes parecidos quedan juntos) y solo el triángulo inferior "
               "para no repetir. Azul = relación negativa, rojo = positiva. "
               "Relaciones clave: O₃ ↔ NO₂ negativa; PM10 ↔ NO₂ positiva. "
               "Correlación no implica causalidad.")

    with st.expander("Ver la forma de cada relación (dispersión por pares)",
                     icon=":material/grain:"):
        cols_disp = [c for c in POLLUTANTS if d["horario"][c].notna().sum() > 100]
        elegidos = st.multiselect(
            "Contaminantes a cruzar (2 a 5)", cols_disp,
            default=[c for c in ["O3", "NO2", "PM10", "CO"] if c in cols_disp],
            max_selections=5)
        if len(elegidos) >= 2:
            pts = _downsample(d["horario"][elegidos].dropna(), 1500)
            st.altair_chart(chart_splom(pts, elegidos))
            st.caption("Cada panel es un par de contaminantes. El heatmap resume "
                       "la fuerza; aquí se ve la forma (lineal, curva, dispersa).")
        else:
            st.info("Elige al menos dos contaminantes.")


@st.fragment
def _chi_fragment(d):
    o3 = d["diario"][d["diario"]["parametro"] == "O3"]
    estaciones = sorted(o3["estacion"].unique())
    est = st.selectbox("Estación para la prueba χ²", estaciones,
                       index=estaciones.index("MER") if "MER" in estaciones else 0)
    tab = tabla_contingencia(o3[o3["estacion"] == est])
    res = chi_desde_tabla(tab)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**¿Depende la calidad por ozono de la temporada?**")
        if res is not None:
            chi2, p, dof, t, esp = res
            mos = mosaico_df(t, esp)
            st.altair_chart(chart_mosaico(mos), use_container_width=True)
            st.caption("Mosaico: el ancho de cada columna es proporcional a los días "
                       "de esa temporada; el alto, a la proporción de cada categoría. "
                       "El color es el **residuo estandarizado**: gris ≈ lo esperado "
                       "si fueran independientes; azul/rojo = se desvía. Un mosaico "
                       "casi sin color significa **poca asociación**.")
        else:
            st.info("Datos insuficientes para la prueba en esta estación.")
    with col2:
        st.markdown("**Resultado de la prueba**")
        if res is not None:
            chi2, p, dof, t, esp = res
            st.metric("Estadístico χ²", f"{chi2:.2f}")
            st.metric("p-valor", f"{p:.3f}")
            crit = chi2_dist.ppf(0.95, dof)
            if p < 0.05:
                st.success(f"p < 0.05 → se rechaza H₀ (gl={dof}, crítico={crit:.2f}). "
                           "Hay dependencia con la temporada.",
                           icon=":material/check_circle:")
            else:
                st.warning(f"p ≥ 0.05 → NO se rechaza H₀ (gl={dof}, crítico={crit:.2f}). "
                           "No se detecta dependencia.", icon=":material/info:")
            with st.expander("Ver tabla y proporciones"):
                st.dataframe(t)
                prop_long = (t.reset_index().melt(
                    id_vars=t.index.name or "index", var_name="temporada",
                    value_name="dias"))
                prop_long.columns = ["categoria", "temporada", "dias"]
                prop_long["orden"] = prop_long["categoria"].map(
                    {c: i for i, c in enumerate(CAT_ORDER)})
                st.altair_chart(chart_chi_prop(prop_long), use_container_width=True)


def pagina_correlacion(d):
    st.header(":material/analytics: Unidad III · Correlación y prueba χ²")
    st.markdown("Dos preguntas de relación: entre **contaminantes** (¿se mueven "
                "juntos?) y entre **calidad del aire y temporada** (¿son independientes?).")
    _corr_fragment(d)
    st.divider()
    _chi_fragment(d)
    st.markdown(
        "**Lectura honesta del resultado.** El χ² se recalcula en vivo sobre la tabla "
        f"mostrada; el notebook había exportado χ² ≈ {d['resumen']['chi2']}, "
        f"p ≈ {d['resumen']['chi2_p']:.2f}. Los números cambian un poco según el "
        "recorte de datos, pero **coinciden en lo esencial: en Merced no se rechaza "
        "H₀**. Sorprende, porque la estacionalidad del ozono está documentada. ¿Por "
        "qué pasa? Merced es céntrica y su ozono no se concentra tan fuerte por "
        "temporada como el del surponiente; además la prueba categórica (con cuatro "
        "categorías y tres temporadas) **pierde potencia**. Es una invitación a "
        "pensar en la prueba, no un fracaso del análisis.")

    with st.expander("Ver la prueba χ² en TODAS las estaciones",
                     icon=":material/table_chart:"):
        tabla_todas = chi_todas_estaciones(d["dir"])
        st.dataframe(tabla_todas, hide_index=True)
        n_rech = int((tabla_todas["¿rechaza H₀?"] == "sí").sum())
        st.caption(f"Solo {n_rech} de {len(tabla_todas)} estaciones rechazan la "
                   "independencia. El patrón estacional existe, pero esta prueba "
                   "categórica solo lo capta en pocos sitios.")


def pagina_conclusiones(d):
    st.header(":material/flag: Conclusiones")
    st.markdown(
        "Con datos reales y con huecos de la RAMA (2024–2026, horarios) y las "
        "técnicas del curso **implementadas a mano y verificadas contra librerías**, "
        "el proyecto muestra una sola historia coherente:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "1. **El aire tiene dos relojes.** Fourier confirma el ciclo diario de "
            "24 h (con armónico de 12 h) y una modulación estacional. El ozono es "
            "esencialmente periódico y predecible.\n\n"
            "2. **El espacio se ordena por química.** PCA y MDS coinciden: un núcleo "
            "de primarios (tráfico/industria) frente a una periferia de ozono, con "
            "un eje de SO₂ que marca el corredor industrial del norte.")
    with c2:
        st.markdown(
            "3. **Las relaciones tienen sentido físico.** O₃ ↔ NO₂ negativa, "
            "PM10 ↔ NO₂ positiva.\n\n"
            "4. **Un resultado que invita a pensar.** La χ² no detecta dependencia "
            "estacional en la estación central; solo dos estaciones de la red la "
            "rechazan. Es una oportunidad de análisis crítico (potencia de la "
            "prueba, elección de estación y de categorías), no un fracaso.")
    st.divider()
    st.markdown("**Notas de honestidad para la defensa**")
    st.markdown(
        "- La interpretación del PCA se ajustó a las cargas reales: PC1 = primarios "
        "↔ ozono (no 'nivel general'); PC2 = SO₂; el PM no entró al análisis.\n"
        "- La conclusión de la χ² se reporta tal cual la dan los números "
        "(p ≈ 0.66 en Merced), explicando por qué no se rechaza H₀.")


# --------------------------------------------------------------------------
# App principal
# --------------------------------------------------------------------------
PAGINAS = {
    "Inicio": pagina_inicio,
    "Los datos": pagina_datos,
    "Unidad I · Ciclos y filtros": pagina_ciclos,
    "Unidad II · Análisis espectral": pagina_espectro,
    "Unidad II · PCA y MDS": pagina_pca,
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
        st.error("No encontré los archivos de datos. Asegúrate de que exista "
                 f"`{propuesta}/resumen.json` y los demás CSV junto a `app.py`.",
                 icon=":material/error:")
        st.stop()

    try:
        d = {
            "dir": str(data_dir),
            "horario": load_horario(str(data_dir)),
            "diario": load_diario(str(data_dir)),
            "pca": load_pca(str(data_dir)),
            "cargas": load_cargas(str(data_dir)),
            "resumen": load_resumen(str(data_dir)),
            "pca_mds": pca_mds(str(data_dir)),
            "diurno_norm": diurno_normalizado(str(data_dir)),
        }
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al leer los datos: {e}", icon=":material/error:")
        st.stop()

    st.sidebar.caption(f"Datos cargados desde: `{data_dir}`")
    PAGINAS[eleccion](d)


if __name__ == "__main__":
    main()
