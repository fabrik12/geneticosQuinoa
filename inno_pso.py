# innovador_pso.py

import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# Importa la función de utilidad del modelo
from modelo import calcular_utilidad


DEMANDA = 150

# --- 0. Definición de la Función Objetivo para PSO ---

# --- Función de verificación (la misma que usamos en el AG) ---
def verificar_condiciones(particula):
    """Verifica si una partícula cumple las restricciones del problema."""
    x1, x2 = particula[0], particula[1]
    
    # Asumimos x3 (Z) = 75 para la restricción compleja
    Z = DEMANDA 
    X4 = 0.25 # Constante de la competencia

    condicion_x1 = x1 <= 100
    # La siguiente línea puede generar un error si Z es 0, pero nuestro Z es 75.
    condicion_x2 = x2 <= (34 - (20.4 * x1) / Z + 17 * X4)
    
    return condicion_x1 and condicion_x2


# --- 1. Definición de la Función Objetivo para PSO ---

# ¡IMPORTANTE!
# Las librerías de optimización como PySwarms están diseñadas para MINIMIZAR.
# Se quiere MAXIMIZAR la utilidad.
# La solución es minimizar el NEGATIVO de la utilidad.
# Minimizar -f(x) es lo mismo que maximizar f(x).

# --- Función Objetivo Mejorada ---
def funcion_objetivo_pso(particulas):
    """
    Calcula el costo para el enjambre, aplicando penalizaciones
    y tratando x1 como un entero.
    """
    costos = []
    for particula in particulas:
        # MEJORA 1: Redondear x1 al entero más cercano
        x1_entero = np.round(particula[0])
        x2 = particula[1]
        
        # Partícula con el x1 ya redondeado para la verificación
        particula_corregida = [x1_entero, x2]

        # MEJORA 2: Aplicar penalización por restricción
        if not verificar_condiciones(particula_corregida):
            # Si no es válida, se le asigna un costo altísimo
            costos.append(1e10) # Un número muy grande
        else:
            # Si es válida, se calcula la utilidad con el x1 entero
            utilidad = calcular_utilidad(x1_entero, x2)
            costos.append(-utilidad) # Negativo para maximizar
            
    return np.array(costos)


# --- 2. Configuración y Ejecución del Optimizador PSO ---

def ejecutar_experimento_pso():
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bounds = [1.0, 11.0] 
    max_bounds = [100.0, 35.0]
    bounds = (np.array(min_bounds), np.array(max_bounds))

    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)
    
    # IMPORTANTE: Verbose a False para no imprimir 50 líneas en cada una de las N ejecuciones
    costo_optimo, pos_optima = optimizer.optimize(funcion_objetivo_pso, iters=50, verbose=False)
    
    max_utilidad = -costo_optimo
    
    # Devolver un diccionario con los resultados clave
    return {
        "utilidad_maxima": max_utilidad,
        "mejor_x1": np.round(pos_optima[0]),
        "mejor_x2": pos_optima[1],
        "historial_costo": optimizer.cost_history # Para la gráfica de convergencia
    }


if __name__ == '__main__':
    print("\n--- Ejecutando una sola prueba del Optimizador PSO ---")
    resultado = ejecutar_experimento_pso()
    print("\nResultados del Optimizador PSO:")
    print(f"  Máxima utilidad encontrada: {resultado['utilidad_maxima']:.2f}")
    print(f"  Mejor x1: {resultado['mejor_x1']:.2f}")
    print(f"  Mejor x2: {resultado['mejor_x2']:.2f}")

    # Graficar
    plot_cost_history(cost_history=resultado['historial_costo'])
    plt.show()