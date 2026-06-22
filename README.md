# Análisis de calidad del aire de la CDMX (RAMA/SIMAT, 2024–2026)

*Por García Juanillo Alan y Reyes León Angel Eduardo · Grupo 5AM1*

Proyecto final de **Analítica y Visualización de Datos**. Se analizan los datos
horarios de calidad del aire del Valle de México aplicando las técnicas del
curso —preprocesamiento, análisis espectral, PCA/MDS y correlación— y se
presenta todo en un **dashboard interactivo** hecho con Streamlit.

## 🔗 Dashboard interactivo

(https://proyecto-aire-cdmx.streamlit.app/)

## Contenido del repositorio

- `app.py` — dashboard interactivo (Streamlit), organizado por las tres unidades del curso.
- `proyecto_final_aire_cdmx.ipynb` — notebook con todo el análisis.
- `datos/crudos/` — datos originales de la RAMA (un Excel por año y contaminante).
- `datos/procesados/` — resultados procesados que alimentan el dashboard.
- `requirements.txt` — dependencias para ejecutar el dashboard.
- `environment.yml` — entorno conda para reproducir el notebook.

## Cómo ejecutar el dashboard localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```



## Datos

Fuente: Sistema de Monitoreo Atmosférico (SIMAT) de la Ciudad de México —
Red Automática de Monitoreo Atmosférico (RAMA). Periodo 2024–2026, frecuencia
horaria. Contaminantes: O₃, NO, NO₂, NOX, CO, SO₂, PM10, PM2.5 y PMCO.

## Autores

- **Alan García Juanillo**
- **Angel Eduardo Reyes León**

Grupo: **5AM1** · Materia: Analítica y Visualización de Datos
