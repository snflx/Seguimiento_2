# 📊 Databehavior Insights

![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Pandas_|_Scikit_Learn-blue?style=for-the-badge&logo=pandas&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github&logoColor=white)

Bienvenido a **Databehavior Insights**, una solución completa para el análisis estadístico y predictivo del comportamiento de los usuarios basada en datos crudos limpios. 

El proyecto consta de una Landing Page interactiva, y una App (Dashboard) full-stack desarrollada en Python con **Streamlit**, la cual procesa datos usando múltiples modelos matemáticos y de Machine Learning en *tiempo real*.

## 🚀 Características Principales

- **Landing Page Interactiva:** (`index.html`) Muestra la explicación paso a paso de los análisis realizados, empleando elementos de diseño visual premium y UI/UX moderno.
- **Análisis Descriptivos y Frecuencias:** Cálculo dinámico de medias, medianas, varianzas, asimetría, curtosis y su representación mediante boxplots e histogramas interactivos.
- **Predicción en Tiempo Real (Machine Learning):** Se implementó y entrenó un modelo de **Regresión Múltiple**. Acepta "Valor de Clics", "Páginas Vistas" y "Tiempo", para proyectar el **Valor Estimado de Compra**.
- **Análisis Geográfico:** Mapa tipo "Coropleta" que identifica mediante variaciones de color las zonas globales con el valor promedio de compra más alto.
- **Filtros Globales Dinámicos:** Permite cruzar en vivo y en directo datos por Producto, Dispositivos y Fuentes de Tráfico, para analizar de manera muy minuciosa los resultados sin usar un motor de base de datos voluminoso.
- **Buenas Prácticas Aplicadas:** Código altamente comentado, modular (`app.py`), evitando el anti-patrón de 'Spaghetti Code', asegurado mediante el uso de caché local (`@st.cache_data`).

## ⚙️ Estructura del Proyecto

```text
📁 Proyecto - DATABEHAVIOR INSIGHTS
│
├── Data_Limpio.xlsx           # Dataset con las métricas depuradas
├── Code_Complete.py           # Código base secuencial del modelo original
├── app.py                     # Streamlit App - El Dashboard Analítico (Modular)
├── index.html                 # Landing Page explicativa
├── requirements.txt           # Dependencias para correr la App
└── README.md                  # Este documento
```

## 🛠️ Instalación y Uso Local

Sigue los pasos a continuación para correr este proyecto en tu entorno local.

1. **Abre tu terminal favorita (Cmd o Powershell).**

2. **Ve al directorio del proyecto y asegúrate de tener un entorno virtual para no entrar en conflicto con tus librerías base (opcional, pero recomendado).**
```bash
python -m venv mi_entorno
mi_entorno\Scripts\activate
```

3. **Instala las dependencias necesarias:**
```bash
pip install -r requirements.txt
```

4. **Corre la aplicación localmente en Streamlit:**
```bash
streamlit run app.py
```

5. **Abre tu Navegador en la Landing Page principal (`index.html`) para apreciar el flujo de entrada a la App.**

---
*Hecho por [Databehavior Insights]*
