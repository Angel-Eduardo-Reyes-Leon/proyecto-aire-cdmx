# Análisis de calidad del aire de la CDMX (RAMA/SIMAT, 2024–2026)

*Por Angel Eduardo Reyes León y Alan García Juanillo · Grupo 5AM1*

Proyecto final de **Analítica y Visualización de Datos**. Se analizan los datos
horarios de calidad del aire del Valle de México aplicando las técnicas del
curso —preprocesamiento, análisis espectral, PCA/MDS y correlación— y se
presenta todo en un **dashboard interactivo** hecho con Streamlit.

## 🔗 Dashboard interactivo

(pega aquí el enlace de Streamlit cuando lo despliegues)

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

Se abre solo en el navegador (o entra a http://localhost:8501).

## Datos

Fuente: Sistema de Monitoreo Atmosférico (SIMAT) de la Ciudad de México —
Red Automática de Monitoreo Atmosférico (RAMA). Periodo 2024–2026, frecuencia
horaria. Contaminantes: O₃, NO, NO₂, NOX, CO, SO₂, PM10, PM2.5 y PMCO.

## Autores

- **Angel Eduardo Reyes León**
- **Alan García Juanillo**

Grupo: **5AM1** · Materia: Analítica y Visualización de Datos
