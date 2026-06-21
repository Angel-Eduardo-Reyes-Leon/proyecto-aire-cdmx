"""
Dashboard de calidad del aire de la CDMX (RAMA/SIMAT, 2024-2026)
================================================================
Proyecto de Analítica y Visualización de Datos.

CÓMO EJECUTARLO
---------------
1. Coloca este archivo (app.py) en la carpeta del proyecto, junto a la
   carpeta `datos/` (la que contiene datos/procesados/*.csv).
   La estructura esperada es:
       proyecto/
         app.py            <-- este archivo
         datos/
           procesados/
             horario_principal.csv
             diario_todas.csv
             pca_estaciones.csv
             cargas_pca.csv
             resumen.json
   Si tus archivos están en otra ruta, cámbiala en la barra lateral
   (campo "Ruta de datos") o edita DATA_DIR_DEFAULT abajo.

2. Activa el entorno que ya tiene streamlit (el de environment.yml):
       conda activate aire-cdmx

3. Ejecuta:
       streamlit run app.py

   Se abrirá solo en el navegador. Si no, entra a http://localhost:8501
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2 as chi2_dist
import streamlit as st

# --------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------
st.set_page_config(page_title="Calidad del aire CDMX", page_icon="🌫️", layout="wide")

DATA_DIR_DEFAULT = "datos/procesados"

POLLUTANTS = ["CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5", "PMCO", "SO2"]
UNITS = {"CO": "ppm", "NO": "ppb", "NO2": "ppb", "NOX": "ppb", "O3": "ppb",
         "PM10": "µg/m³", "PM2.5": "µg/m³", "PMCO": "µg/m³", "SO2": "ppb"}

ZONE_NAMES = {"NE": "Nororiente", "NO": "Noroeste", "CE": "Centro",
              "SO": "Surponiente", "SE": "Sureste"}
ZONE_COLORS = {"NE": "#D85A30", "NO": "#BA7517", "CE": "#534AB7",
               "SO": "#1D9E75", "SE": "#378ADD"}

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SEASON_ORDER = ["Seca-fría", "Seca-caliente", "Lluvias"]
CAT_ORDER = ["Buena", "Aceptable", "Mala", "Muy mala"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13,
})


# --------------------------------------------------------------------------
# Funciones auxiliares (química / estadística del proyecto)
# --------------------------------------------------------------------------
def temporada(mes: int) -> str:
    if mes in (11, 12, 1, 2):
        return "Seca-fría"
    if mes in (3, 4, 5):
        return "Seca-caliente"
    return "Lluvias"


def categoria_o3(valor: float) -> str:
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


def resolver_data_dir(propuesta: str) -> Path | None:
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
    st.title("🌫️ Calidad del aire en la Ciudad de México")
    st.markdown(
        "Análisis de la **Red Automática de Monitoreo Atmosférico (RAMA/SIMAT)** "
        "del Valle de México, con datos horarios de 2024 a 2026. Este tablero "
        "presenta los resultados del proyecto, organizados según las tres unidades "
        "del curso."
    )
    st.markdown(
        "> **Pregunta del proyecto:** ¿qué patrones gobiernan la contaminación del "
        "aire en la CDMX —según la hora, la temporada y la zona— y cómo se "
        "relacionan los contaminantes entre sí?"
    )

    r = d["resumen"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estaciones (PCA)", f"{d['pca']['estacion'].nunique()}")
    c2.metric("Periodo", f"{min(r['anios'])}–{max(r['anios'])}")
    c3.metric("Estación principal", f"{r['nombre_estacion']} ({r['estacion_principal']})")
    c4.metric("Varianza PC1 + PC2", f"{(r['varianza_pc1'] + r['varianza_pc2'])*100:.0f}%")

    st.divider()
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
    st.caption("Usa el menú de la izquierda para recorrer cada unidad.")


def pagina_datos(d):
    st.header("Los datos")
    st.markdown(
        "Cada estación de la RAMA mide los **contaminantes criterio** cada hora. "
        "Los archivos crudos vienen como un Excel por año y contaminante; el "
        "notebook los integró, limpió y exportó a los CSV que alimentan este "
        "tablero."
    )
    tabla_cont = pd.DataFrame({
        "Contaminante": POLLUTANTS,
        "Unidad": [UNITS[p] for p in POLLUTANTS],
        "Tipo": ["Secundario (se forma con sol)" if p == "O3" else "Primario / mezcla"
                 for p in POLLUTANTS],
    })
    st.dataframe(tabla_cont, hide_index=True, use_container_width=True)

    st.subheader(f"Cobertura de datos en {d['resumen']['nombre_estacion']} (estación principal)")
    st.caption("Porcentaje de horas con dato válido. Los datos de la RAMA tienen "
               "huecos reales; 2026 está poco poblado.")
    h = d["horario"].copy()
    h["anio"] = h["fecha"].dt.year
    cob = (h.groupby("anio")[POLLUTANTS].apply(lambda x: x.notna().mean() * 100)).round(1)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(cob.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(POLLUTANTS)), POLLUTANTS)
    ax.set_yticks(range(len(cob.index)), cob.index)
    for i in range(cob.shape[0]):
        for j in range(cob.shape[1]):
            ax.text(j, i, f"{cob.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("% de horas válidas por año y contaminante")
    fig.colorbar(im, ax=ax, label="% válido")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Ver una muestra de la serie horaria (Merced)"):
        st.dataframe(d["horario"].head(24), use_container_width=True)


def pagina_ciclos(d):
    st.header("Unidad I · Exploración: los ciclos del aire")
    st.caption("Series de la estación principal (Merced).")
    cont = st.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
    h = d["horario"].copy()
    h["hora"] = h["fecha"].dt.hour
    h["mes"] = h["fecha"].dt.month
    u = UNITS[cont]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ciclo diurno (por hora del día)")
        g = h.groupby("hora")[cont]
        media, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.fill_between(media.index, q1, q3, color="#9FE1CB", alpha=0.5, label="rango intercuartílico")
        ax.plot(media.index, media.values, color="#0F6E56", lw=2, label="promedio")
        ax.set_xlabel("hora del día"); ax.set_ylabel(f"{cont} ({u})")
        ax.set_xticks(range(0, 24, 3)); ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        if cont == "O3":
            st.caption("El ozono es secundario: se forma con el sol y alcanza su "
                       "máximo a primera hora de la tarde.")

    with col2:
        st.subheader("Ciclo anual (por mes)")
        media_mes = h.groupby("mes")[cont].mean().reindex(range(1, 13))
        fig, ax = plt.subplots(figsize=(6, 3.8))
        colores = ["#BA7517" if m in (3, 4, 5) else "#888780" for m in range(1, 13)]
        ax.bar(range(1, 13), media_mes.values, color=colores)
        ax.set_xticks(range(1, 13), MESES, rotation=0, fontsize=8)
        ax.set_ylabel(f"{cont} ({u}) — promedio")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        if cont == "O3":
            st.caption("En naranja la temporada seca-caliente (mar–may), cuando el "
                       "ozono se dispara por la máxima radiación.")

    st.divider()
    st.subheader("Filtros de suavizado (Unidad I)")
    st.caption("La serie horaria es ruidosa; los filtros revelan la tendencia. "
               "Media y mediana con ventana q = 24 h; exponencial con θ = 0.05.")
    anios = sorted(h["fecha"].dt.year.unique())
    ay = st.select_slider("Año a mostrar", anios, value=anios[0])
    hh = d["horario"]
    s = hh[hh["fecha"].dt.year == ay].set_index("fecha")[cont].sort_index()
    if s.notna().sum() < 100:
        st.warning("Pocos datos válidos en ese año para este contaminante.")
    else:
        media = s.rolling(24, center=True, min_periods=6).mean()
        mediana = s.rolling(24, center=True, min_periods=6).median()
        expo = pd.Series(filtro_exponencial(s.values, 0.05), index=s.index)
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(s.index, s.values, color="#D3D1C7", lw=0.6, label="señal cruda")
        ax.plot(media.index, media.values, color="#185FA5", lw=1.4, label="media móvil")
        ax.plot(mediana.index, mediana.values, color="#993C1D", lw=1.4, label="mediana móvil")
        ax.plot(expo.index, expo.values, color="#3B6D11", lw=1.4, label="exponencial")
        ax.set_ylabel(f"{cont} ({u})"); ax.legend(ncol=4, fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


def pagina_espectro(d):
    st.header("Unidad II · Análisis espectral")
    st.markdown(
        "La transformada de Fourier descompone la serie en las **ondas** que la "
        "forman y mide cuál es más fuerte. Esperamos que confirme, de forma "
        "objetiva, los ciclos vistos en la exploración."
    )
    h = d["horario"]
    anios = sorted(h["fecha"].dt.year.unique())
    c1, c2 = st.columns(2)
    cont = c1.selectbox("Contaminante", POLLUTANTS, index=POLLUTANTS.index("O3"))
    ay = c2.selectbox("Año (ventana de análisis)", anios, index=0)

    serie = h[h["fecha"].dt.year == ay].set_index("fecha")[cont].sort_index()
    res = espectro(serie)
    if res is None:
        st.warning("No hay suficientes datos continuos en esa combinación.")
        return
    periodos, amp = res
    umbral = amp.mean() + 3 * amp.std()

    mask = (periodos >= 2) & (periodos <= 1000)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(periodos[mask], amp[mask], color="#534AB7", lw=1.2)
    ax.axhline(umbral, color="#A32D2D", ls="--", lw=1, label="umbral (media + 3σ)")
    for ph in (24, 12):
        ax.axvline(ph, color="#888780", ls=":", lw=1)
        ax.text(ph, ax.get_ylim()[1] * 0.95, f"{ph} h", rotation=90,
                va="top", ha="right", fontsize=8, color="#5F5E5A")
    picos = mask & (amp > umbral)
    ax.scatter(periodos[picos], amp[picos], color="#D85A30", zorder=5, s=30,
               label="picos dominantes")
    ax.set_xscale("log"); ax.set_xlabel("periodo (horas) — escala log")
    ax.set_ylabel("amplitud"); ax.legend(fontsize=8)
    ax.set_title(f"Espectro de {cont} en Merced ({ay})")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    top = (pd.DataFrame({"periodo_h": periodos[picos], "amplitud": amp[picos]})
           .sort_values("amplitud", ascending=False).head(8).round(2))
    top["periodo_h"] = top["periodo_h"].round(1)
    st.caption("Periodos detectados como dominantes (amplitud por encima del umbral):")
    st.dataframe(top, hide_index=True, use_container_width=True)
    if cont == "O3":
        st.info("En el ozono aparecen picos cerca de **24 h** (el ciclo "
                "fotoquímico diario) y **12 h** (el armónico que corrige la forma "
                "asimétrica de la curva).")


def pagina_pca(d):
    st.header("Unidad II · PCA: el mapa de las estaciones")
    r = d["resumen"]
    pca, cargas = d["pca"], d["cargas"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Varianza PC1", f"{r['varianza_pc1']*100:.1f}%")
    c2.metric("Varianza PC2", f"{r['varianza_pc2']*100:.1f}%")
    c3.metric("Componentes para 90%", f"{r['q_90']}")

    st.subheader("Estaciones en el plano PC1–PC2 (color por zona)")
    fig, ax = plt.subplots(figsize=(10, 6))
    for z, sub in pca.groupby("zona"):
        ax.scatter(sub["PC1"], sub["PC2"], s=70, color=ZONE_COLORS.get(z, "#888780"),
                   label=f"{z} · {ZONE_NAMES.get(z, z)}", edgecolor="white", zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["estacion"], (row["PC1"], row["PC2"]),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#444441")
    ax.axhline(0, color="#B4B2A9", lw=0.8); ax.axvline(0, color="#B4B2A9", lw=0.8)
    ax.set_xlabel("PC1 (~73%):  primarios (tráfico/industria)  ←→  ozono")
    ax.set_ylabel("PC2 (~14%):  SO₂ ↑  (corredor industrial norte)")
    ax.legend(fontsize=8, loc="best")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Cargas: qué pesa en cada componente")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(cargas)); w = 0.38
        ax.bar(x - w/2, cargas["PC1"], w, label="PC1", color="#534AB7")
        ax.bar(x + w/2, cargas["PC2"], w, label="PC2", color="#1D9E75")
        ax.axhline(0, color="#888780", lw=0.8)
        ax.set_xticks(x, cargas["parametro"]); ax.legend(fontsize=8)
        ax.set_ylabel("carga")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col2:
        st.subheader("Interpretación")
        st.markdown(
            "- **PC1** (eje horizontal): los primarios (CO, NO, NO₂, NOX) cargan "
            "negativo y el O₃ positivo. Es un eje **primarios ↔ ozono**, no un "
            "eje de \"nivel general\".\n"
            "- **PC2** (eje vertical): lo domina el **SO₂**. Aísla al corredor "
            "industrial del norte (Tlalnepantla, Cuautitlán, Atizapán).\n"
            "- El **PM no entró** en este PCA (lo descartó el filtro de cobertura)."
        )
    st.caption("Nota: esta lectura corrige la redacción original del notebook, que "
               "describía PC1 como 'nivel general' e incluía al PM.")


def pagina_correlacion(d):
    st.header("Unidad III · Correlación entre contaminantes")
    st.caption("Series horarias de la estación principal (Merced).")
    metodo = st.radio("Coeficiente", ["pearson", "spearman", "kendall"], horizontal=True)
    cols = [c for c in POLLUTANTS if d["horario"][c].notna().sum() > 100]
    corr = d["horario"][cols].corr(method=metodo)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr.values[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label=f"correlación ({metodo})")
    ax.set_title(f"Matriz de correlación — {metodo}")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown(
        "Relaciones clave: **O₃ ↔ NO₂ negativa** (el NO₂ se consume al formarse "
        "el ozono) y **PM10 ↔ NO₂ positiva** (ambos vienen del tráfico). "
        "Recuerda: correlación no implica causalidad."
    )

    st.divider()
    st.subheader("Prueba χ²: ¿la calidad del aire por ozono depende de la temporada?")
    diario = d["diario"]
    o3 = diario[diario["parametro"] == "O3"].copy()
    estaciones = sorted(o3["estacion"].unique())
    est = st.selectbox("Estación", estaciones,
                       index=estaciones.index("MER") if "MER" in estaciones else 0)

    tab = tabla_contingencia(o3[o3["estacion"] == est])
    res = chi_desde_tabla(tab)
    cA, cB = st.columns([3, 2])
    with cA:
        st.markdown("**Tabla de contingencia** (días observados):")
        st.dataframe(tab, use_container_width=True)
    with cB:
        if res is not None:
            chi2, p, dof, _ = res
            st.metric("Estadístico χ²", f"{chi2:.2f}")
            st.metric("p-valor", f"{p:.3f}")
            crit = chi2_dist.ppf(0.95, dof)
            if p < 0.05:
                st.success(f"p < 0.05 → se rechaza H₀ (gl={dof}, crítico={crit:.2f}). "
                           "Hay dependencia con la temporada.")
            else:
                st.warning(f"p ≥ 0.05 → NO se rechaza H₀ (gl={dof}, crítico={crit:.2f}). "
                           "No se detecta dependencia.")
        else:
            st.info("Datos insuficientes para la prueba en esta estación.")

    st.markdown(
        f"El valor exportado por el notebook para **Merced** fue χ² = {d['resumen']['chi2']}, "
        f"p = {d['resumen']['chi2_p']:.2f} → **no se rechaza H₀**. Esto sorprende, "
        "porque la estacionalidad del ozono está documentada. La razón: Merced es "
        "céntrica y su ozono no se concentra tan fuerte por temporada como el del "
        "surponiente; la prueba categórica pierde potencia ahí."
    )

    with st.expander("Ver la prueba χ² en TODAS las estaciones"):
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
        st.dataframe(tabla_todas, hide_index=True, use_container_width=True)
        n_rech = (tabla_todas["¿rechaza H₀?"] == "sí").sum()
        st.caption(f"Solo **{n_rech}** de {len(tabla_todas)} estaciones rechazan la "
                   "independencia. El patrón estacional existe, pero esta prueba "
                   "categórica solo lo detecta en pocos sitios.")


def pagina_conclusiones(d):
    st.header("Conclusiones")
    st.markdown(
        "El proyecto tomó datos reales y con huecos de la RAMA (2024–2026, "
        "horarios) y, aplicando las técnicas del curso **implementadas a mano y "
        "verificadas contra librerías**, mostró que:\n\n"
        "1. **El aire tiene dos relojes.** El análisis espectral confirmó de forma "
        "objetiva un ciclo diario de 24 h (con armónico de 12 h) y una modulación "
        "estacional. El ozono es esencialmente periódico y predecible.\n"
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
    st.divider()
    st.subheader("Notas de honestidad para la defensa")
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
def main():
    st.sidebar.title("Calidad del aire CDMX")
    st.sidebar.caption("RAMA/SIMAT · 2024–2026")

    propuesta = st.sidebar.text_input("Ruta de datos", DATA_DIR_DEFAULT)
    data_dir = resolver_data_dir(propuesta)
    if data_dir is None:
        st.error(
            "No encontré los archivos de datos. Asegúrate de que existan "
            f"`{propuesta}/resumen.json` y los demás CSV, o corrige la 'Ruta de "
            "datos' en la barra lateral.\n\nEstructura esperada: una carpeta "
            "`datos/procesados/` junto a `app.py`."
        )
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
        st.error(f"Error al leer los datos: {e}")
        st.stop()

    paginas = {
        "Inicio": pagina_inicio,
        "Los datos": pagina_datos,
        "Unidad I · Ciclos y filtros": pagina_ciclos,
        "Unidad II · Análisis espectral": pagina_espectro,
        "Unidad II · PCA (mapa)": pagina_pca,
        "Unidad III · Correlación y χ²": pagina_correlacion,
        "Conclusiones": pagina_conclusiones,
    }
    eleccion = st.sidebar.radio("Secciones", list(paginas.keys()))
    st.sidebar.divider()
    st.sidebar.caption(f"Datos cargados desde:\n`{data_dir}`")
    paginas[eleccion](d)


if __name__ == "__main__":
    main()
