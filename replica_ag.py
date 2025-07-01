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
LOW_X1_INT, UP_X1_INT = 1, 100
LOW_X2, UP_X2 = 11.0, 35.0 # Usamos un rango razonable para el precio

# Atributo generador para cada gen
toolbox.register("attr_x1", random.randint, LOW_X1_INT, UP_X1_INT)
toolbox.register("attr_x2", random.uniform, LOW_X2, UP_X2)

# Estructura del individuo y la población
# Un individuo se crea llamando a 'tools.initCycle' que llenará un contenedor 'creator.Individual'
# con una secuencia de llamadas a 'attr_x1' y 'attr_x2'.
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_x1, toolbox.attr_x2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 3. Definición de Operadores Genéticos ---

# --- NUEVA FUNCIÓN DE VERIFICACIÓN ---
def verificar_condiciones(individual):
    """
    Verifica si un individuo cumple las restricciones del paper.
    Basado en: Function condition(X, Y, Z)
    Donde: X=x1, Y=x2, Z=x3 (x3 no es un gen, lo asumimos válido para la prueba)
    """
    x1, x2 = individual[0], individual[1]
    
    # La restricción del paper depende de X3 (demanda), que no es parte de nuestro individuo.
    # Para el propósito de la validación del AG, nos enfocaremos en la restricción
    # más simple que sí podemos verificar: x1 <= 100
    # X4 es una constante
    X4 = 0.25

    # Asumimos un valor promedio para Z (x3) para poder usar la fórmula completa
    # Por ejemplo, la mitad del rango, 75. Esto es una simplificación necesaria.
    Z = 75 
    
    # Aplicamos las condiciones del paper 
    condicion_x1 = x1 <= 100
    condicion_x2 = x2 <= (34 - (20.4 * x1) / Z + 17 * X4)
    
    return condicion_x1 and condicion_x2


# --- FUNCIÓN DE EVALUACIÓN ACTUALIZADA ---
def evaluar_individuo_con_penalizacion(individual):
    """
    Evalúa la aptitud de un individuo.
    Si no es válido, su aptitud es 0.
    """
    if not verificar_condiciones(individual):
        return (0,)  # Penalización: aptitud nula si viola las restricciones
    
    # Si es válido, calcula la utilidad normal
    x1, x2 = individual[0], individual[1]
    return (calcular_utilidad(x1, x2),)

toolbox.register("evaluate", evaluar_individuo_con_penalizacion)


# --- FUNCIÓN DE CRUCE PERSONALIZADO PARA TIPOS MIXTOS ---
def cruce_personalizado(ind1, ind2):
    """
    Aplica un cruce de dos puntos al primer gen (entero) y un
    cruce de mezcla (blend) al segundo gen (flotante).
    """
    # Cruce para x1 (Entero): intercambia el valor con 50% de probabilidad
    if random.random() < 0.5:
        ind1[0], ind2[0] = ind2[0], ind1[0]

    # Cruce para x2 (Flotante): blend
    alpha = 0.5
    x2_1 = ind1[1]
    x2_2 = ind2[1]
    ind1[1] = (1 - alpha) * x2_1 + alpha * x2_2
    ind2[1] = (1 - alpha) * x2_2 + alpha * x2_1

    return ind1, ind2

# Operadores de Cruce y Mutación (usando los parámetros del paper)
toolbox.register("mate", cruce_personalizado) # Cruce de mezcla

# --- FUNCIÓN DE MUTACIÓN PERSONALIZADA PARA TIPOS MIXTOS ---
def mutacion_personalizada(individual, low_int, up_int, mu, sigma, indpb):
    """
    Aplica una mutación de entero al primer gen (x1) y una
    mutación gaussiana al segundo gen (x2).
    """
    # Mutación para x1 (Entero)
    if random.random() < indpb:
        individual[0] = random.randint(low_int, up_int)

    # Mutación para x2 (Flotante)
    if random.random() < indpb:
        # Reutilizamos la lógica de mutación gaussiana para el flotante.
        individual[1] += random.gauss(mu, sigma)
    
    return individual,

# Registramos nuestra nueva función de mutación personalizada.
toolbox.register("mutate", mutacion_personalizada, 
                 low_int=LOW_X1_INT, up_int=UP_X1_INT, # Límites para el entero
                 mu=0, sigma=5, indpb=0.2) # Parámetros para el flotante

# --- TAREA: Configurar y Comparar Métodos de Selección ---

# a) Selección por Torneo (ST) - como se describe en el paper
toolbox.register("select_tournament", tools.selTournament, tournsize=3)

# b) Selección Sexual (SS) - Esto requiere una implementación más personalizada
# La Selección Sexual en AG usualmente implica crear sub-poblaciones (sexos)
# y restringir el cruce. Para una simulación simple, se podria crear una función
# que imite el concepto, por ejemplo, eligiendo parejas basadas en alguna
# métrica de "compatibilidad" o simplemente dividiendo la población.

# --- FUNCIÓN PARA EJECUTAR AG CON SELECCIÓN SEXUAL (SS) ---
def ejecutar_ag_con_ss(pop_size=100, n_gen=50, cxpb=0.65, mutpb=0.08):
    """
    Ejecuta el algoritmo genético usando una lógica de Selección Sexual personalizada.
    """
    print("\n--- Ejecutando AG con Selección Sexual (SS) ---")

    # 1. Inicialización
    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Evaluar la población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Bucle de generaciones
    for g in range(n_gen):
        # 2. Selección de la siguiente generación
        # Seleccionamos los mejores individuos para ser los padres de la siguiente generación.
        offspring = toolbox.select_tournament(pop, len(pop))
        # Clonamos los seleccionados para no modificar los originales
        offspring = list(map(toolbox.clone, offspring))

        # 3. Cruce con lógica de Selección Sexual
        # Dividimos la descendencia en dos mitades
        mid_point = len(offspring) // 2
        machos = offspring[:mid_point]
        hembras = offspring[mid_point:]

        # Aplicamos el cruce entre un "macho" y una "hembra"
        for i in range(mid_point):
            if random.random() < cxpb:
                toolbox.mate(machos[i], hembras[i])
                # Liberamos la aptitud de los hijos modificados
                del machos[i].fitness.values
                del hembras[i].fitness.values
        
        # Unimos las dos mitades de nuevo en la descendencia
        offspring = machos + hembras

        # 4. Mutación
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 5. Evaluación de nuevos individuos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 6. Reemplazo de la población
        pop[:] = offspring

        # Registro de estadísticas
        record = stats.compile(pop)
        print(f"Gen {g+1}: Max Utilidad = {record['max']:.2f}, Avg Utilidad = {record['avg']:.2f}")

    # 7. Resultados finales
    best_ind = tools.selBest(pop, 1)[0]
    print("\nMejor individuo con selección sexual:")
    print(f"  x1 (tierra) = {best_ind[0]:.2f}, x2 (precio) = {best_ind[1]:.2f}")
    print(f"  Máxima utilidad = {best_ind.fitness.values[0]:.2f}")

    return best_ind


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
    
    # Ejecución con Selección Sexual
    ejecutar_ag_con_ss()