import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('default')
sns.set_palette("husl")

# Cargar los datos
df = pd.read_excel('Data_Limpio.xlsx', sheet_name='Sheet1')

# Función para formatear números grandes
def format_large_number(n):
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

# Análisis para columnas numéricas
numeric_columns = ['clics', 'paginas_vistas', 'tiempo_en_pagina_seg', 'tasa_rebote', 'valor_compra_usd']

print("=" * 80)
print("ANÁLISIS ESTADÍSTICO COMPLETO - DATOS DE COMPORTAMIENTO DE USUARIOS")
print("=" * 80)

# 1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS
print("\n1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS")
print("-" * 50)

for col in numeric_columns:
    print(f"\n{col.upper()}:")
    print(f"  Media: {format_large_number(df[col].mean())}")
    print(f"  Mediana: {format_large_number(df[col].median())}")
    mode_result = df[col].mode()
    if not mode_result.empty:
        print(f"  Moda: {format_large_number(mode_result.iloc[0])}")
    else:
        print("  Moda: No disponible")
    print(f"  Desviación Estándar: {format_large_number(df[col].std())}")
    print(f"  Varianza: {format_large_number(df[col].var())}")

# 2. CUARTILES, DECILES Y PERCENTILES
print("\n\n2. CUARTILES, DECILES Y PERCENTILES")
print("-" * 50)

for col in numeric_columns:
    print(f"\n{col.upper()}:")
    # Cuartiles
    q1, q2, q3 = np.percentile(df[col].dropna(), [25, 50, 75])
    print(f"  Q1 (25%): {format_large_number(q1)}")
    print(f"  Q2/Mediana (50%): {format_large_number(q2)}")
    print(f"  Q3 (75%): {format_large_number(q3)}")
    
    # Deciles
    deciles = np.percentile(df[col].dropna(), range(10, 100, 10))
    print("  Deciles:", [format_large_number(d) for d in deciles])
    
    # Percentiles seleccionados
    percentiles = np.percentile(df[col].dropna(), [10, 25, 50, 75, 90, 95, 99])
    print("  Percentiles (10, 25, 50, 75, 90, 95, 99):", [format_large_number(p) for p in percentiles])

# 3. ASIMETRÍA Y CURTOSIS
print("\n\n3. ASIMETRÍA Y CURTOSIS")
print("-" * 50)

for col in numeric_columns:
    skewness = stats.skew(df[col].dropna())
    kurtosis = stats.kurtosis(df[col].dropna())
    
    print(f"\n{col.upper()}:")
    print(f"  Asimetría: {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("    → Distribución aproximadamente simétrica")
    elif skewness > 0:
        print("    → Distribución con sesgo positivo (cola a la derecha)")
    else:
        print("    → Distribución con sesgo negativo (cola a la izquierda)")
    
    print(f"  Curtosis: {kurtosis:.4f}")
    if kurtosis > 0:
        print("    → Distribución leptocúrtica (picos más altos, colas más pesadas)")
    elif kurtosis < 0:
        print("    → Distribución platicúrtica (picos más bajos, colas más ligeras)")
    else:
        print("    → Distribución mesocúrtica (similar a la normal)")

# 4. ANÁLISIS POR CATEGORÍAS
print("\n\n4. ANÁLISIS POR CATEGORÍAS")
print("-" * 50)

categorical_columns = ['producto', 'fuente_trafico', 'dispositivo', 'pais']

for cat_col in categorical_columns:
    print(f"\n{cat_col.upper()}:")
    for num_col in numeric_columns:
        grouped = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std'])
        print(f"  {num_col}:")
        for category in grouped.index:
            print(f"    {category}: Media={format_large_number(grouped.loc[category, 'mean'])}, "
                  f"Mediana={format_large_number(grouped.loc[category, 'median'])}, "
                  f"Desv.Est.={format_large_number(grouped.loc[category, 'std'])}")

# 5. REGRESIÓN MÚLTIPLE
print("\n\n5. REGRESIÓN MÚLTIPLE")
print("-" * 50)

# Preparar datos para regresión (usando valor_compra_usd como variable dependiente)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Seleccionar variables relevantes para la regresión
regression_vars = ['clics', 'paginas_vistas', 'tiempo_en_pagina_seg']
X = df[regression_vars].fillna(df[regression_vars].mean())
y = df['valor_compra_usd'].fillna(df['valor_compra_usd'].mean())

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Modelo de Regresión Múltiple:")
print(f"  R²: {r2:.4f}")
print(f"  MSE: {mse:.2f}")
print("  Coeficientes:")
for i, coef in enumerate(model.coef_):
    print(f"    {regression_vars[i]}: {coef:.6f}")
print(f"  Intercepto: {model.intercept_:.2f}")

# Interpretación
print("\n  Interpretación:")
print("  - R² indica qué porcentaje de la variabilidad en 'valor_compra_usd' es explicado por el modelo")
print("  - Los coeficientes muestran cómo cambia 'valor_compra_usd' por cada unidad de cambio en cada variable")

# 6. GRÁFICOS
print("\n\n6. GENERANDO GRÁFICOS...")
print("-" * 50)

# Configurar la visualización de gráficos
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('Análisis Estadístico de Datos de Comportamiento de Usuarios', fontsize=16, fontweight='bold')

# 6.1. Histogramas de variables numéricas
for i, col in enumerate(numeric_columns[:4]):  # Mostrar solo 4 para no saturar
    ax = axes[i//2, i%2]
    df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribución de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frecuencia')
    # Añadir líneas de media y mediana
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Media: {format_large_number(df[col].mean())}')
    ax.axvline(df[col].median(), color='green', linestyle='--', label=f'Mediana: {format_large_number(df[col].median())}')
    ax.legend()

# 6.2. Boxplots por producto
ax = axes[2, 0]
df.boxplot(column='valor_compra_usd', by='producto', ax=ax)
ax.set_title('Valor de Compra por Producto')
ax.set_ylabel('Valor de Compra (USD)')
ax.set_xlabel('Producto')

# 6.3. Heatmap de correlación
ax = axes[2, 1]
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Matriz de Correlación')

plt.tight_layout()
plt.show()

# 7. GRÁFICOS ADICIONALES
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Análisis Adicional de los Datos', fontsize=16, fontweight='bold')

# 7.1. Gráfico de dispersión con línea de regresión
ax = axes2[0, 0]
sns.regplot(x='clics', y='valor_compra_usd', data=df, ax=ax, scatter_kws={'alpha':0.5})
ax.set_title('Relación entre Clics y Valor de Compra')
ax.set_xlabel('Clics')
ax.set_ylabel('Valor de Compra (USD)')

# 7.2. Conteo por fuente de tráfico
ax = axes2[0, 1]
df['fuente_trafico'].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
ax.set_title('Distribución por Fuente de Tráfico')
ax.set_xlabel('Fuente de Tráfico')
ax.set_ylabel('Cantidad')
ax.tick_params(axis='x', rotation=45)

# 7.3. Valor promedio de compra por país
ax = axes2[1, 0]
df.groupby('pais')['valor_compra_usd'].mean().sort_values(ascending=False).plot(
    kind='bar', ax=ax, color='lightgreen', edgecolor='black')
ax.set_title('Valor Promedio de Compra por País')
ax.set_xlabel('País')
ax.set_ylabel('Valor Promedio de Compra (USD)')
ax.tick_params(axis='x', rotation=45)

# 7.4. Tasa de conversión por dispositivo
ax = axes2[1, 1]
conversion_rate = df.groupby('dispositivo')['conversion'].mean() * 100
conversion_rate.plot(kind='bar', ax=ax, color='orange', edgecolor='black')
ax.set_title('Tasa de Conversión por Dispositivo')
ax.set_xlabel('Dispositivo')
ax.set_ylabel('Tasa de Conversión (%)')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 8. ÁREAS BAJO LA CURVA (usando distribución acumulativa)
print("\n\n8. ÁREAS BAJO LA CURVA (Distribución Acumulativa)")
print("-" * 50)

fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
fig3.suptitle('Distribuciones Acumulativas y Áreas Bajo la Curva', fontsize=16, fontweight='bold')

for i, col in enumerate(numeric_columns[:4]):
    ax = axes3[i//2, i%2]
    # Calcular ECDF
    sorted_data = np.sort(df[col].dropna())
    yvals = np.arange(1, len(sorted_data)+1)/len(sorted_data)
    
    ax.plot(sorted_data, yvals, linewidth=2)
    ax.set_title(f'Distribución Acumulativa de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Probabilidad Acumulada')
    ax.grid(True, alpha=0.3)
    
    # Calcular y mostrar áreas bajo la curva para diferentes percentiles
    for percentile in [25, 50, 75, 90]:
        p_value = np.percentile(df[col].dropna(), percentile)
        area = percentile/100
        ax.axvline(p_value, color='red', linestyle='--', alpha=0.7)
        ax.text(p_value, 0.5, f'{percentile}%', rotation=90, verticalalignment='center')

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)
print("\nRESUMEN EJECUTIVO:")
print("- Se analizaron 5 variables numéricas y 4 categóricas")
print("- Se calcularon todas las medidas estadísticas solicitadas")
print("- Se generaron 3 conjuntos de gráficos para visualización")
print("- El modelo de regresión múltiple explica el comportamiento del valor de compra")
print("- Los gráficos muestran distribuciones, relaciones y patrones en los datos")