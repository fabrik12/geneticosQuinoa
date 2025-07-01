# replica_ag.py (Refactorizado)

import random
import numpy as np
from deap import base, creator, tools, algorithms

# Importa la función de utilidad del modelo
from modelo import calcular_utilidad 

# --- 1. Configuración del Problema (Maximización) ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Inicialización de la Caja de Herramientas (Toolbox) ---
toolbox = base.Toolbox()

# Límites para cada gen
LOW_X1_INT, UP_X1_INT = 1, 100
LOW_X2, UP_X2 = 11.0, 35.0

DEMANDA = 150

toolbox.register("attr_x1", random.randint, LOW_X1_INT, UP_X1_INT)
toolbox.register("attr_x2", random.uniform, LOW_X2, UP_X2)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_x1, toolbox.attr_x2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 3. Definición de Operadores Genéticos ---

def verificar_condiciones(individual):
    """Verifica si un individuo cumple las restricciones del paper."""
    x1, x2 = individual[0], individual[1]
    Z = DEMANDA
    X4 = 0.25

    condicion_x1 = x1 <= 100
    condicion_x2 = x2 <= (34 - (20.4 * x1) / Z + 17 * X4)
    
    return condicion_x1 and condicion_x2

def evaluar_individuo_con_penalizacion(individual):
    """Evalúa la aptitud de un individuo. Si no es válido, su aptitud es 0."""
    if not verificar_condiciones(individual):
        return (0,)
    x1, x2 = individual[0], individual[1]
    return (calcular_utilidad(x1, x2),)

toolbox.register("evaluate", evaluar_individuo_con_penalizacion)

def cruce_personalizado(ind1, ind2):
    """Aplica un cruce personalizado para tipos mixtos."""
    if random.random() < 0.5:
        ind1[0], ind2[0] = ind2[0], ind1[0]
    alpha = 0.5
    ind1[1] = (1 - alpha) * ind1[1] + alpha * ind2[1]
    ind2[1] = (1 - alpha) * ind2[1] + alpha * ind1[1]
    return ind1, ind2

def mutacion_personalizada(individual, low_int, up_int, mu, sigma, indpb):
    """Aplica una mutación personalizada para tipos mixtos."""
    if random.random() < indpb:
        individual[0] = random.randint(low_int, up_int)
    if random.random() < indpb:
        individual[1] += random.gauss(mu, sigma)
    return individual,

toolbox.register("mate", cruce_personalizado)
toolbox.register("mutate", mutacion_personalizada, 
                 low_int=LOW_X1_INT, up_int=UP_X1_INT,
                 mu=0, sigma=5, indpb=0.2)

# Se usará Selección por Torneo como base para la Selección Sexual
toolbox.register("select", tools.selTournament, tournsize=3)


# --- NUEVA FUNCIÓN PARA EL EXPERIMENTO ---
def ejecutar_experimento_ag(pop_size=100, n_gen=50, cxpb=0.65, mutpb=0.08, verbose=False):
    """
    Ejecuta el algoritmo genético usando Selección Sexual y devuelve los resultados.
    """
    pop = toolbox.population(n=pop_size)
    
    # Herramientas para estadísticas y logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "max", "avg"
    
    # Evaluar la población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    # Bucle de generaciones
    for g in range(1, n_gen + 1):
        # Selección
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Cruce (Lógica de Selección Sexual)
        mid_point = len(offspring) // 2
        machos = offspring[:mid_point]
        hembras = offspring[mid_point:]

        for i in range(min(len(machos), len(hembras))):
            if random.random() < cxpb:
                toolbox.mate(machos[i], hembras[i])
                del machos[i].fitness.values
                del hembras[i].fitness.values
        
        offspring = machos + hembras
        
        # Mutación
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluación
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Reemplazo
        pop[:] = offspring
        
        # Registro
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
            
    # Resultados finales
    best_ind = tools.selBest(pop, 1)[0]
    
    # Extraer historial de utilidad máxima para la gráfica de convergencia
    max_utility_history = logbook.select("max")
    
    # Devolvemos un diccionario con los resultados clave
    return {
        "utilidad_maxima": best_ind.fitness.values[0],
        "mejor_x1": best_ind[0],
        "mejor_x2": best_ind[1],
        # Se devuelve el negativo para ser consistente con el "costo" de PSO
        "historial_costo": -np.array(max_utility_history) 
    }

# --- Bloque principal para ejecución individual ---
if __name__ == '__main__':
    print("--- Ejecutando una sola prueba del Algoritmo Genético (AG) con Selección Sexual ---")
    
    # llamada a la nueva funcion con verbose=True para ver los detalles
    resultado = ejecutar_experimento_ag(verbose=True)
    
    print("\nResultados del Algoritmo Genético:")
    print(f"  Mejor individuo: x1={resultado['mejor_x1']:.2f}, x2={resultado['mejor_x2']:.2f}")
    print(f"  Máxima utilidad encontrada: {resultado['utilidad_maxima']:.2f}")