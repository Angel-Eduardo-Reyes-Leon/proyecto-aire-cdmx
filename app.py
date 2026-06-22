"""
Aire CDMX: instrumento narrativo de calidad del aire (RAMA/SIMAT 2024-2026).
Cinco vistas organizadas por la pregunta y tres latidos (tiempo, espacio, causa).
Identidad visual y color en .streamlit/config.toml. Navegacion por pestanas con
carga perezosa; una vista a la vez. Local: streamlit run app.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

alt.data_transformers.disable_max_rows()

# --- colores (espejo de config.toml) -------------------------------------
INK = "#16242B"
SUBTLE = "#5B6B74"
ACCENT = "#1F7A8C"     # azul-ozono
WARM = "#E07B2E"       # ambar-smog
MUTED = "#C4CED2"
HAIR = "#DDE5E7"

CAT_ORDER = ["Buena", "Aceptable", "Mala", "Muy mala"]
CAT_COLORS = ["#3F9E5A", "#E1B530", "#E07B2E", "#C8392E"]
HEAT = ["#F2F7F8", "#CFE6DA", "#F0D88C", "#E8A04B", "#E07B2E", "#C8392E", "#7A1E18"]

ZONE_NAMES = {"NE": "Nororiente", "NO": "Noroeste", "CE": "Centro",
              "SO": "Surponiente", "SE": "Sureste"}
ZONE_COLORS = {"CE": "#1F7A8C", "NE": "#E07B2E", "NO": "#C99A2E",
               "SO": "#5E8C6A", "SE": "#7E8AA0"}
SEASON_COLORS = {"Seca-fría": "#4D9BA8", "Seca-caliente": "#E07B2E", "Lluvias": "#3F9E5A"}

POLLUTANTS = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "PMCO", "SO2"]
UNITS = {"CO": "ppm", "NO": "ppb", "NO2": "ppb", "NOX": "ppb", "O3": "ppb",
         "PM10": "µg/m³", "PM2.5": "µg/m³", "PMCO": "µg/m³", "SO2": "ppb"}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASON_ORDER = ["Seca-fría", "Seca-caliente", "Lluvias"]
DATA_DIR_DEFAULT = "datos/procesados"


# --- calculo --------------------------------------------------------------
def temporada(mes):
    if mes in (11, 12, 1, 2):
        return "Seca-fría"
    if mes in (3, 4, 5):
        return "Seca-caliente"
    return "Lluvias"


def categoria_o3(v):
    if pd.isna(v):
        return np.nan
    if v <= 58:
        return "Buena"
    if v <= 90:
        return "Aceptable"
    if v <= 135:
        return "Mala"
    return "Muy mala"


# --- carga (memorizada) ---------------------------------------------------
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


@st.cache_data(show_spinner=False)
def load_coords(d):
    p = Path(d) / "coordenadas.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, comment="#")
    if not {"estacion", "lat", "lon"}.issubset(df.columns):
        return None
    return df[["estacion", "lat", "lon"]].dropna()


@st.cache_data(show_spinner=False)
def load_basemap(d):
    p = Path(d) / "mapa_base.geojson"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def resolver_data_dir():
    base = Path(__file__).resolve().parent
    for c in [Path(DATA_DIR_DEFAULT), base / DATA_DIR_DEFAULT, base,
              base / "procesados", base.parent / "datos" / "procesados"]:
        if (c / "resumen.json").exists():
            return str(c)
    return None


# --- analitica derivada (memorizada) --------------------------------------
@st.cache_data(show_spinner=False)
def estaciones_o3(d):
    di = load_diario(d)
    ests = sorted(di.loc[di["parametro"] == "O3", "estacion"].unique())
    nombres = dict(zip(load_pca(d)["estacion"], load_pca(d)["nombre"]))
    return ests, nombres


@st.cache_data(show_spinner=False)
def serie_diaria(d, estacion, cont):
    di = load_diario(d)
    s = di[(di["parametro"] == cont) & (di["estacion"] == estacion)].copy()
    s = s.dropna(subset=["maximo"])
    s["anio"] = s["dia"].dt.year
    s["doy"] = s["dia"].dt.dayofyear
    s["doy2"] = s["doy"] + 1
    if cont == "O3":
        s["categoria"] = s["maximo"].apply(categoria_o3)
        s = s.dropna(subset=["categoria"])
    return s


@st.cache_data(show_spinner=False)
def diurno_pico(d):
    h = load_horario(d).copy()
    g = h.groupby(h["fecha"].dt.hour)["O3"].mean()
    return int(g.idxmax()), float(g.min()), float(g.max())


@st.cache_data(show_spinner=False)
def heatmap_hora_mes(d, cont, anio, agg):
    h = load_horario(d).copy()
    if anio != "Todos":
        h = h[h["fecha"].dt.year == int(anio)]
    h["hora"] = h["fecha"].dt.hour
    h["mes"] = h["fecha"].dt.month
    g = h.groupby(["hora", "mes"])[cont]
    val = (g.mean() if agg == "promedio" else g.max()).reset_index(name="valor")
    val["n"] = g.count().values
    val["valor"] = val["valor"].where(val["n"] >= 3)
    val["mes_nom"] = val["mes"].map(lambda m: MESES[m - 1])
    return val


@st.cache_data(show_spinner=False)
def mapa_data(d, cont, valor_col):
    di = load_diario(d)
    sub = di[di["parametro"] == cont].copy()
    sub["mes"] = sub["dia"].dt.month
    agg = sub.groupby(["estacion", "mes"])[valor_col].mean().reset_index(name="valor")
    return agg


@st.cache_data(show_spinner=False)
def kpis(d, estacion):
    di = load_diario(d)
    o3 = di[di["parametro"] == "O3"]
    est_o3 = o3[o3["estacion"] == estacion]
    dias = int(est_o3["maximo"].notna().sum())
    mala = int((est_o3["maximo"] > 90).sum())
    medias = o3.groupby("estacion")["maximo"].mean()
    return {"dias": dias, "mala": mala,
            "limpia": medias.idxmin(), "sucia": medias.idxmax()}


@st.cache_data(show_spinner=False)
def lab_merced(d, X, Y):
    h = load_horario(d).copy()
    h["hora"] = h["fecha"].dt.hour
    h["temporada"] = h["fecha"].dt.month.map(temporada)
    cols = list({X, Y})
    df = h[cols + ["hora", "temporada", "fecha"]].dropna(subset=cols)
    return df


@st.cache_data(show_spinner=False)
def lab_estacion(d, est, X, Y, valor_col):
    di = load_diario(d)
    sub = di[(di["estacion"] == est) & (di["parametro"].isin([X, Y]))]
    piv = sub.pivot_table(index="dia", columns="parametro", values=valor_col)
    for c in (X, Y):
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv.dropna(subset=[X, Y]).reset_index()
    piv["temporada"] = piv["dia"].dt.month.map(temporada)
    return piv


# --- graficas (el tema lo aplica Streamlit; aqui solo color de datos) ------
def show(ch):
    st.altair_chart(ch, width="stretch")


def show_fixed(ch):
    st.altair_chart(ch)


def chart_timeline(df, cont):
    base = alt.Chart(df).mark_rect()
    x = alt.X("doy:O", title="día del año",
              axis=alt.Axis(values=[1, 60, 120, 182, 244, 305, 366], grid=False, labelAngle=0))
    if cont == "O3":
        color = alt.Color("categoria:N", sort=CAT_ORDER,
                          scale=alt.Scale(domain=CAT_ORDER, range=CAT_COLORS),
                          legend=alt.Legend(title="calidad del aire por O₃", orient="top"))
        tip = [alt.Tooltip("dia:T", title="fecha"),
               alt.Tooltip("categoria:N", title="calidad"),
               alt.Tooltip("maximo:Q", title="O₃ máx (ppb)", format=".0f")]
    else:
        color = alt.Color("maximo:Q", scale=alt.Scale(range=HEAT),
                          legend=alt.Legend(title=f"{cont} máx ({UNITS[cont]})",
                                            orient="top", gradientLength=170))
        tip = [alt.Tooltip("dia:T", title="fecha"),
               alt.Tooltip("maximo:Q", title=f"{cont} máx ({UNITS[cont]})", format=".1f")]
    return base.encode(x=x, color=color, tooltip=tip).properties(height=42).facet(
        row=alt.Row("anio:O", title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12)))


def chart_heatmap(df, cont):
    return alt.Chart(df).mark_rect().encode(
        x=alt.X("mes_nom:O", title=None, sort=MESES, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("hora:O", title="hora del día",
                axis=alt.Axis(values=list(range(0, 24)), labelFontSize=10)),
        color=alt.Color("valor:Q", title=f"{cont} ({UNITS[cont]})",
                        scale=alt.Scale(range=HEAT),
                        legend=alt.Legend(orient="right", gradientLength=200)),
        tooltip=[alt.Tooltip("mes_nom:O", title="mes"),
                 alt.Tooltip("hora:O", title="hora"),
                 alt.Tooltip("valor:Q", title=f"{cont}", format=".1f"),
                 alt.Tooltip("n:Q", title="obs.")],
    ).properties(height=470)


def chart_mapa_geo(basemap, dff, cont, sel=None):
    layers = []
    if basemap is not None:
        layers.append(alt.Chart(alt.Data(values=basemap["features"])).mark_geoshape(
            fill="#E9EFF1", stroke="#FFFFFF", strokeWidth=1.1))
    # viento dominante diurno (referencia, no calculado): del NE hacia el SW
    wind_line = pd.DataFrame({"lat": [19.72, 19.31], "lon": [-98.93, -99.19], "o": [0, 1]})
    layers.append(alt.Chart(wind_line).mark_line(
        color=ACCENT, strokeWidth=2.2, opacity=0.38, strokeDash=[6, 4]).encode(
        longitude="lon:Q", latitude="lat:Q", order="o:Q"))
    head = pd.DataFrame({"lat": [19.31], "lon": [-99.19]})
    layers.append(alt.Chart(head).mark_point(
        shape="triangle", angle=212, size=150, color=ACCENT, opacity=0.5, filled=True).encode(
        longitude="lon:Q", latitude="lat:Q"))
    wlbl = pd.DataFrame({"lat": [19.70], "lon": [-99.00], "t": ["viento dominante diurno"]})
    layers.append(alt.Chart(wlbl).mark_text(
        color=ACCENT, opacity=0.85, fontSize=10, align="left", dx=4).encode(
        longitude="lon:Q", latitude="lat:Q", text="t:N"))
    pts = alt.Chart(dff).mark_circle(size=215, stroke=INK, strokeWidth=0.9, opacity=0.92).encode(
        longitude="lon:Q", latitude="lat:Q",
        color=alt.Color("valor:Q", scale=alt.Scale(range=HEAT),
                        legend=alt.Legend(title=f"{cont} ({UNITS[cont]})",
                                          orient="right", gradientLength=200)),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("valor:Q", title=cont, format=".1f"),
                 alt.Tooltip("zona_nombre:N", title="zona")])
    layers.append(pts)
    if sel is not None and (dff["estacion"] == sel).any():
        layers.append(alt.Chart(dff[dff["estacion"] == sel]).mark_point(
            size=520, color=INK, strokeWidth=2.2, shape="circle").encode(
            longitude="lon:Q", latitude="lat:Q"))
    layers.append(alt.Chart(dff).mark_text(dy=-13, fontSize=9, color=INK).encode(
        longitude="lon:Q", latitude="lat:Q", text="estacion:N"))
    return alt.layer(*layers).properties(width=600, height=560).project(type="mercator")


def chart_mapa_rank(dff):
    zonas = [z for z in ZONE_COLORS if z in set(dff["zona"])]
    escala = alt.Scale(domain=zonas, range=[ZONE_COLORS[z] for z in zonas])
    return alt.Chart(dff).mark_circle(size=150, stroke="white", strokeWidth=0.8).encode(
        x=alt.X("valor:Q", title="concentración (promedio del mes)"),
        y=alt.Y("estacion:N", sort="-x", title=None),
        color=alt.Color("zona:N", scale=escala, legend=alt.Legend(title="zona", orient="top")),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("valor:Q", title="valor", format=".1f")],
    ).properties(height=560)


def chart_pca(pca, cargas):
    zonas = [z for z in ZONE_COLORS if z in set(pca["zona"])]
    escala = alt.Scale(domain=zonas, range=[ZONE_COLORS[z] for z in zonas])
    rad_p = float(np.nanmax(np.abs(pca[["PC1", "PC2"]].values)))
    rad_l = float(np.nanmax(np.abs(cargas[["PC1", "PC2"]].values)))
    k = (rad_p / rad_l) * 0.78 if rad_l > 0 else 1.0
    filas = []
    for _, r in cargas.iterrows():
        filas.append({"parametro": r["parametro"], "o": 0, "x": 0.0, "y": 0.0})
        filas.append({"parametro": r["parametro"], "o": 1,
                      "x": float(r["PC1"]) * k, "y": float(r["PC2"]) * k})
    arr = pd.DataFrame(filas)
    tips = arr[arr["o"] == 1].copy()
    tips["xl"] = tips["x"] * 1.14
    tips["yl"] = tips["y"] * 1.14
    zero_v = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=HAIR).encode(x="x:Q")
    zero_h = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=HAIR).encode(y="y:Q")
    flechas = alt.Chart(arr).mark_line(color=SUBTLE, strokeWidth=1.3, opacity=0.5).encode(
        x=alt.X("x:Q", title="primarios (tráfico e industria)  ←→  ozono"),
        y=alt.Y("y:Q", title="SO₂ (corredor industrial del norte)"),
        detail="parametro:N", order="o:Q")
    fhalo = alt.Chart(tips).mark_text(stroke="white", strokeWidth=3,
                                      fontWeight=700, fontSize=12).encode(
        x="xl:Q", y="yl:Q", text="parametro:N")
    flbl = alt.Chart(tips).mark_text(color=WARM, fontWeight=700, fontSize=12).encode(
        x="xl:Q", y="yl:Q", text="parametro:N")
    sel = alt.selection_point(name="sel_estacion", fields=["estacion"], on="click",
                              toggle=False, empty=False)
    pts = alt.Chart(pca).mark_circle(size=180, stroke="white").encode(
        x="PC1:Q", y="PC2:Q",
        color=alt.Color("zona:N", scale=escala, legend=alt.Legend(title="zona", orient="top")),
        opacity=alt.condition(sel, alt.value(1.0), alt.value(0.85)),
        strokeWidth=alt.condition(sel, alt.value(2.6), alt.value(1.1)),
        tooltip=[alt.Tooltip("estacion:N", title="estación"),
                 alt.Tooltip("nombre:N", title="nombre"),
                 alt.Tooltip("zona_nombre:N", title="zona")]).add_params(sel)
    ehalo = alt.Chart(pca).mark_text(dx=7, align="left", fontSize=9,
                                     stroke="white", strokeWidth=3).encode(
        x="PC1:Q", y="PC2:Q", text="estacion:N")
    elbl = alt.Chart(pca).mark_text(dx=7, align="left", fontSize=9, color=INK).encode(
        x="PC1:Q", y="PC2:Q", text="estacion:N")
    return alt.layer(zero_v, zero_h, flechas, fhalo, flbl, pts, ehalo, elbl).properties(height=480)


def chart_cargas(cargas):
    long = cargas.melt(id_vars="parametro", value_vars=["PC1", "PC2"],
                       var_name="comp", value_name="carga")
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("carga:Q", title="carga"),
        y=alt.Y("parametro:N", title=None, sort=alt.EncodingSortField("carga", op="min")),
        yOffset="comp:N",
        color=alt.Color("comp:N", scale=alt.Scale(domain=["PC1", "PC2"], range=[ACCENT, WARM]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("parametro:N"), alt.Tooltip("comp:N"),
                 alt.Tooltip("carga:Q", format=".2f")],
    ).properties(height=240)


def chart_lab(df, X, Y, color_field):
    if color_field == "temporada":
        color = alt.Color("temporada:N", title="temporada",
                          scale=alt.Scale(domain=SEASON_ORDER,
                                          range=[SEASON_COLORS[s] for s in SEASON_ORDER]),
                          legend=alt.Legend(orient="top"))
    else:
        color = alt.Color("hora:Q", title="hora",
                          scale=alt.Scale(scheme="sinebow"),
                          legend=alt.Legend(orient="top"))
    pts = alt.Chart(df).mark_circle(size=34, opacity=0.45).encode(
        x=alt.X(f"{X}:Q", title=f"{X} ({UNITS[X]})", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{Y}:Q", title=f"{Y} ({UNITS[Y]})", scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[alt.Tooltip(f"{X}:Q", format=".1f"), alt.Tooltip(f"{Y}:Q", format=".1f")])
    reg = alt.Chart(df).transform_regression(X, Y).mark_line(
        color=INK, strokeWidth=2.4).encode(x=f"{X}:Q", y=f"{Y}:Q")
    return (pts + reg).properties(height=400)


# --- secciones ------------------------------------------------------------
def sec_panorama(d):
    dd = d["dir"]
    ests, nombres = estaciones_o3(dd)
    st.title("¿De dónde viene la contaminación de la CDMX y cómo se forma?")
    st.markdown("La respuesta tiene tres latidos: el **tiempo** (cuándo), el **espacio** "
                "(dónde) y la **causa** (cómo se forma). Cada pestaña responde uno.")
    est = st.selectbox("Estación", ests,
                       index=ests.index("MER") if "MER" in ests else 0,
                       format_func=lambda e: f"{e} ({nombres.get(e, 'estación')})")
    k = kpis(dd, est)
    pico, _, _ = diurno_pico(dd)
    c = st.columns(4)
    c[0].metric("Días sobre la norma de O₃", f"{k['mala']}",
                help=f"días con O₃ máximo mayor a 90 ppb en {est}, de {k['dias']} con dato")
    c[1].metric("Hora pico del ozono", f"{pico:02d} h")
    c[2].metric("Estación más limpia", k["limpia"])
    c[3].metric("Estación más cargada", k["sucia"])
    st.markdown("**Calendario de calidad del aire, día a día**")
    cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"),
                        key="cont_panorama")
    serie = serie_diaria(dd, est, cont)
    if len(serie):
        show(chart_timeline(serie, cont))
    else:
        st.info(f"No hay datos diarios de {cont} en esta estación.")
    if cont == "O3":
        st.markdown("Cada franja es un día. La mayoría son verdes: el ozono no suele superar "
                    "la norma. Cambia de contaminante o de estación para ver otra cara.")
    else:
        st.markdown(f"Cada franja es un día, sombreada por el {cont} máximo. "
                    "Compara estaciones y notarás que el color cambia mucho.")


@st.fragment
def sec_reloj(d):
    dd = d["dir"]
    st.caption("Cuándo pasa")
    st.subheader("El reloj doble")
    st.markdown("Una sola imagen con los dos ciclos del aire: la hora del día (vertical) "
                "y el mes del año (horizontal).")
    c1, c2, c3 = st.columns([2, 1.4, 1.4])
    cont = c1.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"),
                        key="cont_global")
    anio = c2.selectbox("Año", ["Todos", "2024", "2025", "2026"], index=0)
    agg = c3.segmented_control("Valor", ["promedio", "máximo"], default="promedio") or "promedio"
    show(chart_heatmap(heatmap_hora_mes(dd, cont, anio, agg), cont))
    msg = {
        "O3": "El ozono se enciende a media tarde y sube en la primavera seca.",
        "NO2": "El NO₂ marca las horas pico del tráfico, por la mañana y la noche.",
        "NO": "El NO se dispara en la hora pico de la mañana y casi desaparece de tarde.",
        "NOX": "Los NOₓ siguen al tráfico: máximos en hora pico, valle de madrugada.",
        "CO": "El CO sigue al tráfico: picos de mañana y noche, valle de madrugada.",
        "PM2.5": "Las PM2.5 suben de noche y madrugada, cuando la atmósfera se estabiliza.",
        "PM10": "Las PM10 crecen con la actividad diurna y el viento de la tarde.",
        "SO2": "El SO₂ aparece en pulsos ligados a la actividad industrial.",
        "PMCO": "La fracción gruesa (PMCO) sigue al viento y la actividad diurna.",
    }.get(cont, f"Patrón de {cont} por hora del día y mes del año.")
    st.markdown(msg)


def sec_mapa(d):
    dd = d["dir"]
    st.caption("Dónde pasa")
    st.subheader("El mapa que respira")
    st.markdown("Cada punto es una estación en su lugar real. Mueve el mes para ver dónde "
                "está alta la contaminación y cómo cambia a lo largo del año.")
    basemap = load_basemap(dd)
    coords = load_coords(dd)
    c1, c2 = st.columns([2, 1.5])
    cont = c1.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"),
                        key="cont_global")
    valor = c2.segmented_control("Valor", ["máximo", "promedio"], default="máximo") or "máximo"
    valor_col = "maximo" if valor == "máximo" else "promedio"
    mes = st.select_slider("Mes", options=list(range(1, 13)), value=5,
                           format_func=lambda m: MESES[m - 1])
    agg = mapa_data(dd, cont, valor_col)
    dff = agg[agg["mes"] == mes].copy()
    zmap = dict(zip(d["pca"]["estacion"], d["pca"]["zona"]))
    dff["zona"] = dff["estacion"].map(zmap).fillna("ND")
    dff["zona_nombre"] = dff["zona"].map(ZONE_NAMES).fillna("sin zona")
    sel = st.session_state.get("estacion_sel")
    if coords is not None:
        g = dff.merge(coords, on="estacion", how="inner")
        if len(g):
            show_fixed(chart_mapa_geo(basemap, g, cont, sel))
            st.markdown(f"El color marca el {cont} del mes: verde es bajo y rojo es alto "
                        "(la escala está a la derecha). La flecha señala el viento dominante "
                        "diurno, puesta como referencia.")
        else:
            st.info("No hay estaciones con ubicación para este mes.")
    else:
        show(chart_mapa_rank(dff))
        st.markdown("Estaciones ordenadas por concentración del mes.")


def sec_quimico(d):
    st.caption("Por qué se agrupan las estaciones")
    st.subheader("El mapa químico")
    st.markdown("Cada punto es una estación, colocada según su mezcla de contaminantes. "
                "Haz clic en una para resaltarla en el mapa y fijarla en el laboratorio.")
    pca = d["pca"].copy()
    pca["zona_nombre"] = pca["zona"].map(ZONE_NAMES).fillna(pca["zona"])
    ev = st.altair_chart(chart_pca(pca, d["cargas"]), on_select="rerun", key="pca_evt")
    try:
        picks = ev.selection.get("sel_estacion") if ev and ev.selection else None
        if picks:
            st.session_state["estacion_sel"] = picks[0]["estacion"]
    except Exception:  # noqa: BLE001
        pass
    sel = st.session_state.get("estacion_sel")
    cc = st.columns([3, 1], vertical_alignment="center")
    if sel:
        cc[0].markdown(f"Seleccionada: **{sel}**. Resaltada en el mapa y en el laboratorio.")
    else:
        cc[0].markdown("Haz clic en una estación para resaltarla.")
    if cc[1].button("Limpiar"):
        st.session_state.pop("estacion_sel", None)
        st.rerun()
    with st.expander("Qué define cada eje"):
        show(chart_cargas(d["cargas"]))
        st.markdown("Eje horizontal: a la izquierda los primarios del tráfico (CO, NO, NOₓ) "
                    "y a la derecha el ozono. Eje vertical: el SO₂, propio del corredor "
                    "industrial del norte.")
    r = d["resumen"]
    st.markdown(f"El eje horizontal explica el {r['varianza_pc1'] * 100:.0f}% de las "
                f"diferencias entre estaciones; sumando el vertical, "
                f"{(r['varianza_pc1'] + r['varianza_pc2']) * 100:.0f}%.")


@st.fragment
def sec_lab(d):
    dd = d["dir"]
    ests, nombres = estaciones_o3(dd)
    principal = d["resumen"]["estacion_principal"]
    st.caption("Cómo se relacionan")
    st.subheader("El laboratorio de relaciones")
    st.markdown("Elige una estación y dos contaminantes para ver si se mueven juntos. "
                "Solo Merced tiene datos por hora; el resto, por día.")
    sel = st.session_state.get("estacion_sel")
    est_def = sel if sel in ests else (principal if principal in ests else ests[0])
    c1, c2, c3 = st.columns(3)
    est = c1.selectbox("Estación", ests, index=ests.index(est_def),
                       format_func=lambda e: f"{e} ({nombres.get(e, 'estación')})")
    X = c2.selectbox("Eje X", POLLUTANTS, index=POLLUTANTS.index("NO2"))
    Y = c3.selectbox("Eje Y", POLLUTANTS, index=POLLUTANTS.index("O3"))
    es_merced = (est == principal)
    c4, c5 = st.columns(2)
    op_color = ["temporada", "hora"] if es_merced else ["temporada"]
    color_field = c4.segmented_control("Color por", op_color, default="temporada") or "temporada"
    metodo = c5.segmented_control("Correlación", ["pearson", "spearman", "kendall"],
                                  default="pearson") or "pearson"
    if X == Y:
        st.info("Elige dos contaminantes distintos.")
        return
    if es_merced:
        df = lab_merced(dd, X, Y)
        modo = "pares horarios"
    else:
        df = lab_estacion(dd, est, X, Y, "promedio")
        modo = "pares diarios"
        color_field = "temporada"
    if len(df) < 5:
        st.info(f"Hay muy pocos datos de {X} y {Y} en {est}.")
        return
    coef = df[X].corr(df[Y], method=metodo)
    cc = st.columns([1, 3], vertical_alignment="center")
    cc[0].metric(f"r ({metodo})", f"{coef:+.2f}")
    cc[1].markdown(f"{X} frente a {Y} en {nombres.get(est, est)}. {len(df)} {modo}. "
                   "La recta muestra la tendencia y el coeficiente, su fuerza.")
    plot = df.sample(4000, random_state=0) if len(df) > 4000 else df
    show(chart_lab(plot, X, Y, color_field))
    st.markdown("Por ejemplo: O₃ y NO₂ bajan juntos, porque el ozono consume su precursor; "
                "PM10 y NO₂ suben juntos, porque ambos vienen del tráfico.")


# --- app ------------------------------------------------------------------
SECCIONES = [
    ("Panorama", sec_panorama),
    ("El reloj doble", sec_reloj),
    ("El mapa que respira", sec_mapa),
    ("El mapa químico", sec_quimico),
    ("El laboratorio", sec_lab),
]


def main():
    st.set_page_config(page_title="Aire CDMX", layout="centered",
                       initial_sidebar_state="collapsed")
    dd = resolver_data_dir()
    if dd is None:
        st.error("No encontré los datos. Debe existir datos/procesados/resumen.json "
                 "junto a app.py.")
        st.stop()
    try:
        d = {"dir": dd, "pca": load_pca(dd), "cargas": load_cargas(dd),
             "resumen": load_resumen(dd)}
    except Exception as e:  # noqa: BLE001
        st.error(f"Error al leer los datos: {e}")
        st.stop()
    tabs = st.tabs([s[0] for s in SECCIONES], on_change="rerun", key="nav")
    for (titulo, render), tab in zip(SECCIONES, tabs):
        if tab.open:
            with tab:
                render(d)


if __name__ == "__main__":
    main()
