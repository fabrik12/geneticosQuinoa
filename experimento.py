# experimento.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importar las funciones refactorizadas
from replica_ag import ejecutar_experimento_ag
from inno_pso import ejecutar_experimento_pso

# --- 1. Configuración del Experimento ---
N_EJECUCIONES = 20 # Número de veces que se ejecutará cada algoritmo

print(f"Iniciando experimento. Se realizarán {N_EJECUCIONES} ejecuciones para cada algoritmo.")

# --- 2. Ejecución y Recolección de Datos ---
resultados_ag = []
for i in range(N_EJECUCIONES):
    print(f"  Ejecutando AG - Corrida {i+1}/{N_EJECUCIONES}...")
    resultados_ag.append(ejecutar_experimento_ag())

resultados_pso = []
for i in range(N_EJECUCIONES):
    print(f"  Ejecutando PSO - Corrida {i+1}/{N_EJECUCIONES}...")
    resultados_pso.append(ejecutar_experimento_pso())

print("\nExperimento finalizado. Analizando resultados...")

# --- 3. Análisis Estadístico y Tabular ---
df_ag = pd.DataFrame(resultados_ag)
df_pso = pd.DataFrame(resultados_pso)

# Añadir una columna para identificar el algoritmo
df_ag['algoritmo'] = 'AG'
df_pso['algoritmo'] = 'PSO'

# Combinar ambos resultados en un solo DataFrame
df_resultados = pd.concat([df_ag, df_pso], ignore_index=True)

# a) Resumen estadístico general
resumen_estadistico = df_resultados.groupby('algoritmo')['utilidad_maxima'].agg(['mean', 'std', 'min', 'max']).reset_index()
print("\n--- Resumen Estadístico de Utilidad Máxima ---")
print(resumen_estadistico)


# b) ¡NUEVO! Reporte del mejor resultado global por algoritmo
print("\n--- Mejor Resultado Global Encontrado (La mejor de las N corridas) ---")

# Encontrar el índice de la mejor corrida para cada algoritmo
idx_mejores_corridas = df_resultados.groupby('algoritmo')['utilidad_maxima'].idxmax()
df_mejores_corridas = df_resultados.loc[idx_mejores_corridas]

# Iterar sobre los mejores resultados para mostrarlos
for _, row in df_mejores_corridas.iterrows():
    print(f"\nAlgoritmo: {row['algoritmo']}")
    print(f"  Mejor Utilidad Máxima: {row['utilidad_maxima']:.2f}")
    print(f"  Valores Óptimos: x1 (tierra) = {row['mejor_x1']:.0f}, x2 (precio) = {row['mejor_x2']:.2f}")



# --- 4. Visualización Gráfica ---

# a) Boxplot para comparar la distribución de resultados
plt.figure(figsize=(10, 6))
sns.boxplot(x='algoritmo', y='utilidad_maxima', data=df_resultados)
plt.title('Comparación de Utilidad Máxima ({} ejecuciones)'.format(N_EJECUCIONES))
plt.ylabel('Utilidad Máxima Encontrada')
plt.xlabel('Algoritmo')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# b) Gráficas de convergencia promedio
def plot_convergencia_promedio(df, algoritmo, color):
    # apilar todos los historiales de costo en una matriz
    historiales = np.vstack(df['historial_costo'].values)
    # Calcular promedio y desviación estándar por cada iteración
    mean_history = np.mean(historiales, axis=0)
    std_history = np.std(historiales, axis=0)
    
    plt.plot(mean_history, label=f'Promedio {algoritmo}', color=color)
    plt.fill_between(
        range(len(mean_history)),
        mean_history - std_history,
        mean_history + std_history,
        alpha=0.2,
        color=color
    )

plt.figure(figsize=(12, 7))
plot_convergencia_promedio(df_ag[df_ag['algoritmo'] == 'AG'], 'AG', 'blue')
plot_convergencia_promedio(df_pso[df_pso['algoritmo'] == 'PSO'], 'PSO', 'orange')

plt.title('Convergencia Promedio (con Desviación Estándar)')
plt.xlabel('Generación / Iteración')
plt.ylabel('Mejor Utilidad (Negativo del Costo)')
plt.legend()
plt.grid(True)
plt.show()