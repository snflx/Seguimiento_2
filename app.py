import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Databehavior Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# FUNCIONES AUXILIARES & CACHE
# ==========================================
@st.cache_data
def load_data():
    """Carga los datos limpios desde el archivo Excel."""
    try:
        df = pd.read_excel('Data_Limpio.xlsx', sheet_name='Sheet1')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

def format_large_number(n):
    """Formatea números grandes para mostrar en pantalla."""
    if pd.isna(n):
        return "N/A"
    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}T"
    elif abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:.2f}"

def calculate_descriptive_stats(df, column):
    """Calcula estadísticas descriptivas para una columna dada."""
    stats_dict = {
        'Media': df[column].mean(),
        'Mediana': df[column].median(),
        'Moda': df[column].mode().iloc[0] if not df[column].mode().empty else np.nan,
        'Desv. Est.': df[column].std(),
        'Varianza': df[column].var(),
        'Asimetría': stats.skew(df[column].dropna()),
        'Curtosis': stats.kurtosis(df[column].dropna())
    }
    return stats_dict

# ==========================================
# CARGA DE DATOS
# ==========================================
df_raw = load_data()

if df_raw.empty:
    st.stop()

# ==========================================
# SIDEBAR - FILTROS GLOBALES
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/google-analytics.png", width=60)
st.sidebar.title("Filtros Globales")

# Convertir o asegurar tipos si existieran fechas u otros casos
categorical_cols = ['producto', 'fuente_trafico', 'dispositivo', 'pais']
numeric_cols = ['clics', 'paginas_vistas', 'tiempo_en_pagina_seg', 'tasa_rebote', 'valor_compra_usd']

# Filtros
selected_products = st.sidebar.multiselect("Producto", options=df_raw['producto'].unique(), default=df_raw['producto'].unique())
selected_sources = st.sidebar.multiselect("Fuente de Tráfico", options=df_raw['fuente_trafico'].unique(), default=df_raw['fuente_trafico'].unique())
selected_devices = st.sidebar.multiselect("Dispositivo", options=df_raw['dispositivo'].unique(), default=df_raw['dispositivo'].unique())

# Aplicar filtros
df_filtered = df_raw[
    (df_raw['producto'].isin(selected_products)) &
    (df_raw['fuente_trafico'].isin(selected_sources)) &
    (df_raw['dispositivo'].isin(selected_devices))
]

st.sidebar.markdown("---")
st.sidebar.info(f"Registros filtrados: **{len(df_filtered)}** / {len(df_raw)}")

# ==========================================
# ESTRUCTURA PRINCIPAL - TABS (FICHAS)
# ==========================================
st.title("📊 Databehavior Insights")
st.markdown("Analítica y predicciones sobre el comportamiento de usuarios.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Estadísticas Descriptivas",
    "🌍 Mapa Interactivo",
    "📊 Gráficos de Frecuencia",
    "🤖 Predicción en Tiempo Real",
    "📁 Datos Crudos"
])

# ------------------------------------------
# TAB 1: ESTADÍSTICAS DESCRIPTIVAS
# ------------------------------------------
with tab1:
    st.header("Estadísticas Descriptivas y Percentiles")
    
    if df_filtered.empty:
        st.warning("No hay datos disponibles con los filtros actuales.")
    else:
        var_to_analyze = st.selectbox("Seleccione la variable a analizar:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Medidas Centrales e Indicadores")
            stats_vals = calculate_descriptive_stats(df_filtered, var_to_analyze)
            
            # Formatear el diccionario para mostrarlo en DF
            stats_df = pd.DataFrame(list(stats_vals.items()), columns=["Métrica", "Valor"])
            # Aplicamos formato especial para mostrarlas legibles
            stats_df['Valor Formateado'] = stats_df['Valor'].apply(lambda x: format_large_number(x) if isinstance(x, (int, float)) else x)
            st.dataframe(stats_df[['Métrica', 'Valor Formateado']], use_container_width=True, hide_index=True)
            
            # Interpretación rápida Asimetría
            asim = stats_vals['Asimetría']
            if abs(asim) < 0.5: asim_texto = "Aproximadamente simétrica"
            elif asim > 0: asim_texto = "Sesgo positivo (cola a la derecha)"
            else: asim_texto = "Sesgo negativo (cola a la izquierda)"
            st.caption(f"**Interpretación Asimetría:** {asim_texto}")
            
        with col2:
            st.subheader("Percentiles")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            perc_values = np.percentile(df_filtered[var_to_analyze].dropna(), percentiles)
            perc_df = pd.DataFrame({
                "Percentil": [f"{p}%" for p in percentiles],
                "Valor": [format_large_number(v) for v in perc_values]
            })
            st.dataframe(perc_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader(f"Distribución de {var_to_analyze}")
        fig_dist = px.histogram(df_filtered, x=var_to_analyze, marginal="box", nbins=40, color_discrete_sequence=['#3b82f6'])
        st.plotly_chart(fig_dist, use_container_width=True)


# ------------------------------------------
# TAB 2: MAPA INTERACTIVO
# ------------------------------------------
with tab2:
    st.header("Análisis Geográfico de Compras")
    
    if df_filtered.empty:
        st.warning("No hay datos disponibles para el mapa.")
    elif 'pais' not in df_filtered.columns:
        st.warning("La columna 'pais' no está en el conjunto de datos.")
    else:
        st.write("Mapa coroplético mostrando el valor promedio de compra por país, permitiendo visualizar la distribución global del gasto de usuarios.")
        
        # Agrupar por país
        df_geo = df_filtered.groupby('pais', as_index=False)['valor_compra_usd'].mean()
        
        # Mapa usando Plotly (Choropleth) asume que la columna de país tiene nombres reconocibles (ej: United States, Spain, etc.)
        fig_map = px.choropleth(
            df_geo,
            locations="pais",
            locationmode="country names",
            color="valor_compra_usd",
            hover_name="pais",
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Valor Promedio de Compra (USD) por País"
        )
        fig_map.update_layout(geo=dict(showcoastlines=True, projection_type="equirectangular"))
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.subheader("Top Países por Valor Promedio de Compra")
        st.dataframe(df_geo.sort_values(by='valor_compra_usd', ascending=False).style.format({'valor_compra_usd': '${:.2f}'}), use_container_width=True, hide_index=True)


# ------------------------------------------
# TAB 3: GRÁFICOS Y TABLAS DE FRECUENCIA
# ------------------------------------------
with tab3:
    st.header("Análisis por Categorías y Frecuencias")
    
    cat_col = st.selectbox("Seleccione la variable categórica:", categorical_cols, index=0)
    
    col_a, col_b = st.columns(2)
    
    freq_data = df_filtered[cat_col].value_counts().reset_index()
    freq_data.columns = [cat_col, 'Frecuencia absoluta']
    freq_data['Frecuencia relativa (%)'] = (freq_data['Frecuencia absoluta'] / len(df_filtered) * 100).round(2)
    
    with col_a:
        st.subheader(f"Tabla de Frecuencias: {cat_col.title()}")
        st.dataframe(freq_data, hide_index=True, use_container_width=True)
        
    with col_b:
        st.subheader("Distribución Gráfica")
        fig_pie = px.pie(freq_data, names=cat_col, values='Frecuencia absoluta', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.subheader(f"Valor de Compra Distribuido por {cat_col.title()}")
    fig_box = px.box(df_filtered, x=cat_col, y='valor_compra_usd', color=cat_col)
    st.plotly_chart(fig_box, use_container_width=True)


# ------------------------------------------
# TAB 4: PREDICCIÓN EN TIEMPO REAL
# ------------------------------------------
with tab4:
    st.header("🔮 Predicción Continua: Valor de Compra (USD)")
    st.markdown("Modelo de **Regresión Múltiple** entrenado en tiempo real con los datos filtrados para estimar el valor de la compra según el comportamiento del usuario.")
    
    if len(df_filtered) < 10:
        st.error("No hay suficientes datos para entrenar el modelo (mínimo 10). Ajusta los filtros.")
    else:
        # Preparación del modelo
        features = ['clics', 'paginas_vistas', 'tiempo_en_pagina_seg']
        
        # Rellenar nulos con la media de sus columnas (prevención)
        X = df_filtered[features].fillna(df_filtered[features].mean())
        y = df_filtered['valor_compra_usd'].fillna(df_filtered['valor_compra_usd'].mean())
        
        try:
            # Entrenamiento en tiempo real
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluación rápida
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # UI
            st.success(f"Modelo Entrenado con Éxito. R² del modelo en Test: **{r2:.4f}**")
            
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                in_clics = st.number_input("Número de Clics", min_value=0, value=int(X['clics'].mean()))
            with col_in2:
                in_paginas = st.number_input("Páginas Vistas", min_value=0, value=int(X['paginas_vistas'].mean()))
            with col_in3:
                in_tiempo = st.number_input("Tiempo (segundos)", min_value=0.0, value=float(X['tiempo_en_pagina_seg'].mean()))
                
            if st.button("Predecir Valor de Compra", type="primary"):
                pred_input = np.array([[in_clics, in_paginas, in_tiempo]])
                prediction = model.predict(pred_input)[0]
                
                # Mostrar resultado destacado
                st.metric(label="Valor Estimado (USD)", value=f"${prediction:,.2f}")
                
            with st.expander("Ver Coeficientes del Modelo"):
                coef_dict = dict(zip(features, model.coef_))
                coef_df = pd.DataFrame(list(coef_dict.items()), columns=["Variable", "Coeficiente (Peso)"])
                st.table(coef_df)
                st.write(f"**Intercepto:** {model.intercept_:.4f}")
                
        except Exception as e:
            st.error(f"Error entrenando el modelo: {e}")

# ------------------------------------------
# TAB 5: EXPLORADOR DE DATOS CRUDOS
# ------------------------------------------
with tab5:
    st.header("Datos Crudos Filtrados")
    st.dataframe(df_filtered, use_container_width=True)
    
    # Botón de Descarga
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados como CSV",
        data=csv,
        file_name='datos_filtrados_comportamiento.csv',
        mime='text/csv',
    )
