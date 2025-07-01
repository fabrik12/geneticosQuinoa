# Plantilla para replicador_ag.py

import random
import numpy as np
from deap import base, creator, tools, algorithms

# Importando la función de utilidad del modelo
from modelo import calcular_utilidad 

# --- 1. Configuración del Problema (Maximización) ---
# Se crea un tipo 'FitnessMax' que hereda de 'base.Fitness' y tiene un peso de 1.0 (para maximizar).
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Se crea un tipo 'Individual' que es una lista y tiene un atributo de fitness del tipo 'FitnessMax'.
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Inicialización de la Caja de Herramientas (Toolbox) ---
toolbox = base.Toolbox()

# Un 'individuo' en nuestro problema tiene 2 genes: x1 (tierra) y x2 (precio).
# Definimos los límites para cada gen.
# x1: [1, 100], x2: [11, ~30] (El precio debe ser > 11 para no generar pérdidas)
LOW_X1, UP_X1 = 1.0, 100.0
LOW_X2, UP_X2 = 11.0, 35.0 # Usamos un rango razonable para el precio

# Atributo generador para cada gen
toolbox.register("attr_x1", random.uniform, LOW_X1, UP_X1)
toolbox.register("attr_x2", random.uniform, LOW_X2, UP_X2)

# Estructura del individuo y la población
# Un individuo se crea llamando a 'tools.initCycle' que llenará un contenedor 'creator.Individual'
# con una secuencia de llamadas a 'attr_x1' y 'attr_x2'.
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_x1, toolbox.attr_x2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 3. Definición de Operadores Genéticos ---

# Función de evaluación (Fitness Function)
def evaluar_individuo(individual):
    x1, x2 = individual[0], individual[1]
    # ¡Importante! La función de utilidad es el objetivo
    return (calcular_utilidad(x1, x2),) # DEAP espera una tupla

toolbox.register("evaluate", evaluar_individuo)

# Operadores de Cruce y Mutación (usando los parámetros del paper)
toolbox.register("mate", tools.cxBlend, alpha=0.5) # Cruce de mezcla
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2) # Mutación Gaussiana

# --- TAREA: Configurar y Comparar Métodos de Selección ---

# a) Selección por Torneo (ST) - como se describe en el paper
toolbox.register("select_tournament", tools.selTournament, tournsize=3)

# b) Selección Sexual (SS) - Esto requiere una implementación más personalizada
# La Selección Sexual en AG usualmente implica crear sub-poblaciones (sexos)
# y restringir el cruce. Para una simulación simple, se podria crear una función
# que imite el concepto, por ejemplo, eligiendo parejas basadas en alguna
# métrica de "compatibilidad" o simplemente dividiendo la población.
# Por ahora, se centra en el Torneo, que es estándar en DEAP.

# --- 4. Ejecución del Algoritmo ---

def ejecutar_ag_con_seleccion(metodo_seleccion):
    """Ejecuta el AG usando un método de selección específico."""
    
    if metodo_seleccion == "torneo":
        toolbox.register("select", toolbox.select_tournament)
    # Aquí se podría añadir la lógica para "sexual"
    # elif metodo_seleccion == "sexual":
    #     toolbox.register("select", mi_seleccion_sexual) 
    else:
        raise ValueError("Método de selección no reconocido.")

    # Parámetros del paper
    POP_SIZE = 100
    CXPB = 0.65  # Probabilidad de cruce
    MUTPB = 0.08 # Probabilidad de mutación
    NGEN = 50    # Número de generaciones
    
    pop = toolbox.population(n=POP_SIZE)
    
    # Herramientas para estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Ejecuta el algoritmo
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                                       stats=stats, verbose=True)
    
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Mejor individuo con selección de {metodo_seleccion}:")
    print(f"  x1 (tierra) = {best_ind[0]:.2f}, x2 (precio) = {best_ind[1]:.2f}")
    print(f"  Máxima utilidad = {best_ind.fitness.values[0]:.2f}")
    
    return best_ind, logbook


if __name__ == '__main__':
    print("--- Ejecutando AG con Selección por Torneo (ST) ---")
    mejor_individuo_st, log_st = ejecutar_ag_con_seleccion("torneo")
    
    # Aquí ejecutaríamos la versión con Selección Sexual para comparar.
    # print("\n--- Ejecutando AG con Selección Sexual (SS) ---")
    # mejor_individuo_ss, log_ss = ejecutar_ag_con_seleccion("sexual")