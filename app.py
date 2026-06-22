"""
Aire CDMX — ¿qué gobierna la contaminación en la Ciudad de México?
==================================================================
Tablero de datos (RAMA/SIMAT, 2024–2026) con Streamlit + Altair.

Está organizado por la PREGUNTA (la hora, la temporada, la zona, las relaciones),
no por las unidades del curso. Diseño:
  · Identidad visual y color viven en .streamlit/config.toml (tema del aire:
    escala del índice de calidad del aire + cielo del altiplano). Sin CSS frágil.
  · Navegación por pestañas con carga perezosa: solo se ejecuta la pestaña activa.
  · Una vista a la vez (selector de vista) para evitar el scroll interminable.
  · Cómputos memorizados (@st.cache_data); el detalle técnico va "bajo demanda".

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

# ---------------------------------------------------------------------------
# Colores (espejo de config.toml) para las escalas de las gráficas
# ---------------------------------------------------------------------------
INK = "#16242B"
SUBTLE = "#5B6B74"
ACCENT = "#1F7A8C"        # azul-ozono: resalta una serie
WARM = "#E07B2E"          # ámbar-smog: énfasis secundario
MUTED = "#C4CED2"         # gris-cielo: series atenuadas
HAIR = "#DDE5E7"

CAT_ORDER = ["Buena", "Aceptable", "Mala", "Muy mala"]
CAT_COLORS = ["#3F9E5A", "#E1B530", "#E07B2E", "#C8392E"]

ZONE_NAMES = {"NE": "Nororiente", "NO": "Noroeste", "CE": "Centro",
              "SO": "Surponiente", "SE": "Sureste"}
ZONE_COLORS = {"CE": "#1F7A8C", "NE": "#E07B2E", "NO": "#C99A2E",
               "SO": "#5E8C6A", "SE": "#7E8AA0"}

DIV_RANGE = ["#1F7A8C", "#EFF1F0", "#E07B2E"]   # cielo ↔ neutro ↔ smog

POLLUTANTS = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "PMCO", "SO2"]
UNITS = {"CO": "ppm", "NO": "ppb", "NO2": "ppb", "NOX": "ppb", "O3": "ppb",
         "PM10": "µg/m³", "PM2.5": "µg/m³", "PMCO": "µg/m³", "SO2": "ppb"}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASON_ORDER = ["Seca-fría", "Seca-caliente", "Lluvias"]
DATA_DIR_DEFAULT = "datos/procesados"


# ---------------------------------------------------------------------------
# Cálculo (química / estadística del proyecto)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Carga y analítica (memorizadas)
# ---------------------------------------------------------------------------
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


def resolver_data_dir():
    base = Path(__file__).resolve().parent
    for c in [Path(DATA_DIR_DEFAULT), base / DATA_DIR_DEFAULT, base,
              base / "procesados", base.parent / "datos" / "procesados"]:
        if (c / "resumen.json").exists():
            return str(c)
    return None


@st.cache_data(show_spinner=False)
def serie_categoria(d, estacion):
    diario = load_diario(d)
    o3 = diario[(diario["parametro"] == "O3") & (diario["estacion"] == estacion)].copy()
    o3["categoria"] = o3["maximo"].apply(categoria_o3)
    o3 = o3.dropna(subset=["categoria"])
    o3["anio"] = o3["dia"].dt.year
    o3["doy"] = o3["dia"].dt.dayofyear
    o3["mes"] = o3["dia"].dt.month
    return o3


@st.cache_data(show_spinner=False)
def estaciones_o3(d):
    diario = load_diario(d)
    ests = sorted(diario.loc[diario["parametro"] == "O3", "estacion"].unique())
    pca = load_pca(d)
    nombres = dict(zip(pca["estacion"], pca["nombre"]))
    return ests, nombres


@st.cache_data(show_spinner=False)
def panorama_stats(d):
    r = load_resumen(d)
    o3 = serie_categoria(d, r["estacion_principal"])
    pct_buena = (o3["categoria"] == "Buena").mean() * 100 if len(o3) else np.nan
    da = diurno_abs(d, "O3")
    hora_pico = int(da.loc[da["media"].idxmax(), "hora"])
    mn, mx = float(da["media"].min()), float(da["media"].max())
    n_est = int(load_diario(d)["estacion"].nunique())
    return {"pct_buena": pct_buena, "hora_pico": hora_pico, "o3_min": mn,
            "o3_max": mx, "n_est": n_est, "principal": r["nombre_estacion"]}


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
    return spec, picos


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


# ---------------------------------------------------------------------------
# Gráficas  (el tema -fuentes, fondo, ejes- lo aplica Streamlit desde config.toml;
# aquí solo se fija el COLOR de los datos y se limpia la retícula)
# ---------------------------------------------------------------------------
def show(ch):
    st.altair_chart(ch, width="stretch")


def chart_timeline(o3):
    escala = alt.Scale(domain=CAT_ORDER, range=CAT_COLORS)
    return alt.Chart(o3).mark_rect().encode(
        x=alt.X("doy:Q", title="día del año →", scale=alt.Scale(domain=[1, 366]),
                axis=alt.Axis(values=[1, 60, 120, 182, 244, 305, 366], grid=False)),
        color=alt.Color("categoria:N", scale=escala, sort=CAT_ORDER, legend=None),
        tooltip=[alt.Tooltip("dia:T", title="fecha"),
                 alt.Tooltip("categoria:N", title="calidad (O₃)"),
                 alt.Tooltip("maximo:Q", title="O₃ máx (ppb)", format=".0f")],
    ).properties(height=42).facet(
        row=alt.Row("anio:O", title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12)))


def chart_diurno(dfm, cont, unit):
    base = alt.Chart(dfm)
    band = base.mark_area(opacity=0.18, color=ACCENT).encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 3)), grid=False)),
        y=alt.Y("q1:Q", title=f"{cont} ({unit})"), y2="q3:Q")
    linea = base.mark_line(color=ACCENT, strokeWidth=2.6,
                           point=alt.OverlayMarkDef(color=ACCENT, size=24)).encode(
        x="hora:Q", y="media:Q",
        tooltip=[alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("media:Q", title="promedio", format=".1f"),
                 alt.Tooltip("q1:Q", title="Q1", format=".1f"),
                 alt.Tooltip("q3:Q", title="Q3", format=".1f")])
    return (band + linea).properties(height=320)


def chart_diurno_norm(dfn, cont):
    test = f"datum.contaminante === '{cont}'"
    return alt.Chart(dfn).mark_line().encode(
        x=alt.X("hora:Q", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24, 6)), grid=False)),
        y=alt.Y("z:Q", title="nivel relativo", axis=alt.Axis(grid=False)),
        detail="contaminante:N",
        color=alt.condition(test, alt.value(ACCENT), alt.value(MUTED)),
        size=alt.condition(test, alt.value(2.8), alt.value(1.0)),
        opacity=alt.condition(test, alt.value(1.0), alt.value(0.5)),
        tooltip=[alt.Tooltip("contaminante:N", title="contaminante"),
                 alt.Tooltip("hora:Q", title="hora"),
                 alt.Tooltip("z:Q", title="nivel (z)", format=".2f")],
    ).properties(height=320)


def chart_espectro(spec, picos):
    x = alt.X("periodo:Q", scale=alt.Scale(type="log"), title="periodo (horas, escala log)")
    linea = alt.Chart(spec).mark_line(color=ACCENT, strokeWidth=1.5).encode(
        x=x, y=alt.Y("amplitud:Q", title="amplitud", axis=alt.Axis(grid=False)),
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", format=".3f")])
    marcas = alt.Chart(pd.DataFrame({"p": [24, 12], "t": ["24 h", "12 h"]}))
    reglas = marcas.mark_rule(color=SUBTLE, strokeDash=[2, 3], opacity=0.8).encode(x="p:Q")
    etq = marcas.mark_text(align="left", dx=4, dy=-6, fontSize=11, fontWeight=600,
                           color=SUBTLE).encode(x="p:Q", text="t:N")
    pts = alt.Chart(picos).mark_point(color=WARM, size=70, filled=True).encode(
        x=x, y="amplitud:Q",
        tooltip=[alt.Tooltip("periodo:Q", title="periodo (h)", format=".1f"),
                 alt.Tooltip("amplitud:Q", format=".3f")])
    return (linea + reglas + etq + pts).properties(height=300)


def chart_mensual(dfm, cont, unit):
    return alt.Chart(dfm).mark_bar().encode(
        x=alt.X("mes:N", title=None, sort=MESES, axis=alt.Axis(grid=False)),
        y=alt.Y("valor:Q", title=f"{cont} ({unit})"),
        color=alt.Color("grupo:N", scale=alt.Scale(
            domain=["Seca-caliente", "Resto del año"], range=[WARM, MUTED]),
            legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("mes:N", title="mes"),
                 alt.Tooltip("valor:Q", title="promedio", format=".1f")],
    ).properties(height=320)


def chart_estacional_cat(long):
    escala = alt.Scale(domain=CAT_ORDER, range=CAT_COLORS)
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("temporada:N", title=None, sort=SEASON_ORDER, axis=alt.Axis(grid=False)),
        y=alt.Y("dias:Q", title="proporción de días", stack="normalize",
                axis=alt.Axis(format="%", grid=False)),
        color=alt.Color("categoria:N", scale=escala, sort=CAT_ORDER,
                        legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("orden:Q", sort="descending"),
        tooltip=[alt.Tooltip("temporada:N", title="temporada"),
                 alt.Tooltip("categoria:N", title="categoría"),
                 alt.Tooltip("dias:Q", title="días")],
    ).properties(height=320)


def chart_mapa(pca, cargas):
    zonas = [z for z in ZONE_COLORS if z in set(pca["zona"])]
    escala = alt.Scale(domain=zonas, range=[ZONE_COLORS[z] for z in zonas])
    rad_p = float(np.nanmax(np.abs(pca[["PC1", "PC2"]].values)))
    rad_l = float(np.nanmax(np.abs(cargas[["PC1", "PC2"]].values)))
    k = (rad_p / rad_l) * 0.78 if rad_l > 0 else 1.0
    # flechas de carga: dos filas por contaminante (origen -> punta)
    filas = []
    for _, r in cargas.iterrows():
        filas.append({"parametro": r["parametro"], "o": 0, "x": 0.0, "y": 0.0})
        filas.append({"parametro": r["parametro"], "o": 1,
                      "x": float(r["PC1"]) * k, "y": float(r["PC2"]) * k})
    arr = pd.DataFrame(filas)
    tips = arr[arr["o"] == 1]
    zero_v = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=HAIR).encode(x="x:Q")
    zero_h = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=HAIR).encode(y="y:Q")
    flechas = alt.Chart(arr).mark_line(color=SUBTLE, strokeWidth=1.4, opacity=0.55).encode(
        x=alt.X("x:Q", title="primarios (tráfico/industria)  ←→  ozono"),
        y=alt.Y("y:Q", title="SO₂ (corredor industrial norte)  ↑"),
        detail="parametro:N", order="o:Q")
    flbl = alt.Chart(tips).mark_text(color=WARM, fontWeight=700, fontSize=12, dy=-3).encode(
        x="x:Q", y="y:Q", text="parametro:N")
    pts = alt.Chart(pca).mark_circle(size=170, opacity=0.92, stroke="white",
                                     strokeWidth=1.2).encode(
        x="PC1:Q", y="PC2:Q",
        color=alt.Color("zona:N", scale=escala, legend=alt.Legend(title="zona", orient="top")),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("nombre:N", title="nombre"),
                 alt.Tooltip("zona_nombre:N", title="zona")])
    elbl = alt.Chart(pca).mark_text(dy=-14, fontSize=10, color=INK).encode(
        x="PC1:Q", y="PC2:Q", text="estacion:N")
    return alt.layer(zero_v, zero_h, flechas, flbl, pts, elbl).properties(height=520)


def chart_corr(long, metodo, orden):
    base = alt.Chart(long)
    heat = base.mark_rect().encode(
        x=alt.X("v1:N", title=None, sort=orden, axis=alt.Axis(grid=False)),
        y=alt.Y("v2:N", title=None, sort=orden, axis=alt.Axis(grid=False)),
        color=alt.Color("corr:Q", title=f"r ({metodo})",
                        scale=alt.Scale(domain=[-1, 0, 1], range=DIV_RANGE),
                        legend=alt.Legend(orient="right")),
        tooltip=[alt.Tooltip("v1:N", title=""), alt.Tooltip("v2:N", title=""),
                 alt.Tooltip("corr:Q", title="correlación", format=".2f")])
    txt = base.mark_text(fontSize=10).encode(
        x=alt.X("v1:N", sort=orden), y=alt.Y("v2:N", sort=orden),
        text=alt.Text("corr:Q", format=".2f"),
        color=alt.condition("abs(datum.corr) > 0.55", alt.value("white"), alt.value(INK)))
    return (heat + txt).properties(height=430)


# ---------------------------------------------------------------------------
# Secciones  (cada una: contexto breve + 1 vista a la vez)
# ---------------------------------------------------------------------------
def sec_panorama(d):
    s = panorama_stats(d["dir"])
    st.caption("PANORAMA · ¿qué gobierna el aire?")
    st.title("En la CDMX, la hora del día manda sobre el aire más que la temporada")
    st.markdown(
        "Con datos horarios de la red **RAMA/SIMAT**, este tablero responde qué "
        "gobierna el ozono de la ciudad: la **hora** (un ciclo diario enorme), la "
        "**zona** (química, no “nivel”) y, mucho menos, la **temporada**.")

    c = st.columns(4)
    c[0].metric("Swing diario del O₃", f"×{s['o3_max'] / s['o3_min']:.0f}",
                help=f"De {s['o3_min']:.0f} ppb al amanecer a {s['o3_max']:.0f} a media tarde, en {s['principal']}.")
    c[1].metric("Hora pico", f"{s['hora_pico']:02d} h")
    c[2].metric("Días “Buena”", f"{s['pct_buena']:.0f} %", help=f"por ozono, en {s['principal']}")
    c[3].metric("Estaciones", f"{s['n_est']}")

    ests, nombres = estaciones_o3(d["dir"])
    idx = ests.index(d["resumen"]["estacion_principal"]) if d["resumen"]["estacion_principal"] in ests else 0
    est = st.selectbox("Estación", ests, index=idx,
                       format_func=lambda e: f"{e} — {nombres.get(e, 'estación')}")
    st.markdown(f"**Tres años de calidad del aire por ozono en {nombres.get(est, est)}, día a día**")
    st.markdown(
        ":green-badge[Buena] :orange-badge[Mala] :red-badge[Muy mala] &nbsp; "
        "(amarillo = Aceptable)")
    show(chart_timeline(serie_categoria(d["dir"], est)))
    st.caption("Cada franja es un día (color = calidad por ozono del máximo diario). "
               "Los días malos se reparten todo el año: el patrón fuerte no es anual, es diario.")

    with st.popover("Sobre los datos", icon=":material/info:"):
        h = d["horario"].copy()
        h["anio"] = h["fecha"].dt.year
        cob = (h.groupby("anio")[POLLUTANTS].apply(lambda x: x.notna().mean() * 100)
               .round(0).astype(int))
        st.markdown(
            "Fuente: **RAMA/SIMAT** del Valle de México, datos horarios 2024–2026, "
            f"limpiados (interpolación de huecos cortos). Estación central: "
            f"**{s['principal']}**; el mapa usa {d['pca']['estacion'].nunique()} "
            "estaciones con cobertura suficiente. La cobertura es parcial (sobre todo 2026).")
        st.dataframe(cob, width="stretch")


def sec_reloj(d):
    st.caption("EL RELOJ DEL DÍA · ¿a qué hora?")
    st.subheader("El ozono se multiplica por veinte entre el amanecer y media tarde")
    st.markdown("El ozono se forma con el sol y revienta a media tarde; el tráfico pica "
                "en la mañana. Elige un contaminante y una vista.")
    cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
    vista = st.segmented_control(
        "Vista", ["Ciclo del día", "Comparar todos", "Espectro (Fourier)"],
        default="Ciclo del día", label_visibility="collapsed")

    if vista == "Comparar todos":
        show(chart_diurno_norm(d["diurno_norm"], cont))
        st.caption("Cada curva es el ciclo diario de un contaminante a escala comparable; "
                   "resaltado, el que elegiste. Se ve el desfase tráfico (mañana) → ozono (tarde).")
    elif vista == "Espectro (Fourier)":
        anios = sorted(d["horario"]["fecha"].dt.year.unique())
        ay = st.selectbox("Año", anios, index=0, key="anio_esp")
        res = espectro_cache(d["dir"], cont, ay)
        if res is None:
            st.info("No hay suficientes datos continuos en esa combinación.", icon=":material/info:")
        else:
            show(chart_espectro(*res))
            st.caption("La transformada de Fourier descompone la serie en ondas. Los picos "
                       "en 24 h y 12 h confirman, de forma objetiva, el ciclo diario.")
    else:
        show(chart_diurno(diurno_abs(d["dir"], cont), cont, UNITS[cont]))
        st.caption(f"Promedio de {cont} por hora, con banda Q1–Q3 (la mitad central de los días).")


def sec_anio(d):
    r = d["resumen"]
    st.caption("EL AÑO DEL OZONO · ¿y la temporada?")
    st.subheader("La temporada apenas mueve la aguja en el centro")
    st.markdown("Uno esperaría que la primavera seca disparara el ozono; en la estación "
                "central el efecto es tenue, y la prueba estadística lo confirma.")
    vista = st.segmented_control(
        "Vista", ["Por temporada", "Mes a mes", "Prueba χ²"],
        default="Por temporada", label_visibility="collapsed")
    long, res = estacional_categoria(d["dir"], r["estacion_principal"])

    if vista == "Mes a mes":
        cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
        show(chart_mensual(mensual(d["dir"], cont), cont, UNITS[cont]))
        st.caption("Promedio mensual; en ámbar, la temporada seca-caliente (mar–may). "
                   "El ozono apenas se mueve: ninguna temporada sobresale con claridad.")
    elif vista == "Prueba χ²":
        if res is not None:
            chi2, p, dof, _ = res
            cc = st.columns(3)
            cc[0].metric("χ²", f"{chi2:.2f}")
            cc[1].metric("p-valor", f"{p:.3f}")
            cc[2].metric("grados de libertad", f"{dof}")
            if p < 0.05:
                st.success("p < 0.05 → la categoría depende de la temporada.",
                           icon=":material/check_circle:")
            else:
                st.warning("p ≥ 0.05 → no se rechaza la independencia: en el centro, la "
                           "categoría de calidad por ozono no depende de forma "
                           "estadísticamente clara de la temporada.", icon=":material/info:")
        st.caption("La prueba confirma las barras: las diferencias entre temporadas son "
                   "demasiado pequeñas (y la cobertura, parcial). El efecto estacional es "
                   "más nítido en el surponiente elevado, no en el centro.")
    else:
        st.markdown(":green-badge[Buena] :orange-badge[Mala] :red-badge[Muy mala]")
        show(chart_estacional_cat(long))
        st.caption("Las tres temporadas reparten los días de forma parecida; la seca-caliente "
                   "tiene algo más de días “Mala”, pero “Buena/Aceptable” domina todo el año.")


def sec_mapa(d):
    r = d["resumen"]
    var2 = (r["varianza_pc1"] + r["varianza_pc2"]) * 100
    st.caption("EL MAPA QUÍMICO · ¿en qué zona?")
    st.subheader("Las estaciones no se ordenan por “nivel”, sino por química")
    st.markdown("Resumida cada estación por su mezcla de contaminantes, el mapa separa "
                "**regímenes químicos**: primarios del tráfico (izq.) ↔ ozono (der.), "
                "y el SO₂ del corredor industrial norte en el eje vertical.")
    pca = d["pca"].copy()
    pca["zona_nombre"] = pca["zona"].map(ZONE_NAMES).fillna(pca["zona"])
    show(chart_mapa(pca, d["cargas"]))
    st.caption(f"Cada punto, una estación (color por zona); las flechas, contaminantes que "
               f"apuntan hacia donde crecen. Estaciones cercanas = mezclas parecidas. "
               f"El eje horizontal explica el {r['varianza_pc1'] * 100:.0f}% de las "
               f"diferencias; con el vertical, {var2:.0f}%.")
    with st.popover("Cómo leer el mapa", icon=":material/help:"):
        st.markdown(
            "- El eje **horizontal** opone **primarios** (CO, NO, NOₓ — izquierda) contra "
            "**ozono** (derecha): el ozono se forma corriente abajo de donde se emiten sus "
            "precursores.\n"
            "- El eje **vertical** lo manda el **SO₂** (industria del norte).\n"
            "- El escalamiento multidimensional (**MDS**) sobre los mismos datos reproduce "
            "este mapa, lo que respalda la lectura.")


def sec_relaciones(d):
    st.caption("LO QUE VA JUNTO · ¿cómo se relacionan?")
    st.subheader("Los contaminantes que comparten origen se mueven juntos")
    st.markdown("¿Qué contaminantes suben y bajan a la vez? Cambia el coeficiente: lineal "
                "(Pearson) o monótono y robusto a valores extremos (Spearman, Kendall).")
    metodo = st.segmented_control("Coeficiente", ["pearson", "spearman", "kendall"],
                                  default="pearson", label_visibility="collapsed")
    metodo = metodo or "pearson"
    long, orden = corr_data(d["dir"], metodo)
    show(chart_corr(long, metodo, orden))
    st.caption("Filas/columnas reordenadas por agrupamiento (los parecidos quedan juntos). "
               "Azul = relación negativa, ámbar = positiva. Dos que cuentan la historia: "
               "**O₃ ↔ NO₂ negativa** (al formarse el ozono se consume su precursor) y "
               "**PM10 ↔ NO₂ positiva** (ambos del tráfico).")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
SECCIONES = [
    ("Panorama", sec_panorama),
    ("El reloj del día", sec_reloj),
    ("El año del ozono", sec_anio),
    ("El mapa químico", sec_mapa),
    ("Lo que va junto", sec_relaciones),
]


def main():
    st.set_page_config(page_title="Aire CDMX", page_icon=":material/airwave:",
                       layout="centered", initial_sidebar_state="collapsed")

    data_dir = resolver_data_dir()
    if data_dir is None:
        st.error("No encontré los datos. Debe existir `datos/procesados/resumen.json` "
                 "y los demás CSV junto a `app.py`.", icon=":material/error:")
        st.stop()
    try:
        d = {"dir": data_dir,
             "horario": load_horario(data_dir),
             "diario": load_diario(data_dir),
             "pca": load_pca(data_dir),
             "cargas": load_cargas(data_dir),
             "resumen": load_resumen(data_dir),
             "diurno_norm": diurno_norm(data_dir)}
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al leer los datos: {e}", icon=":material/error:")
        st.stop()

    tabs = st.tabs([t[0] for t in SECCIONES], on_change="rerun", key="nav")
    for (titulo, render), tab in zip(SECCIONES, tabs):
        if tab.open:
            with tab:
                render(d)


if __name__ == "__main__":
    main()
