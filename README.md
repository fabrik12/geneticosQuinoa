# Optimización de la Producción de Quinua usando AG y PSO

Este proyecto replica y mejora el modelo de optimización para la producción de quinua presentado en el artículo "Optimization Model using Nonlinear Programming and Artificial Intelligence Techniques for Quinoa Production in the Puno Region". El objetivo es comparar el rendimiento de un Algoritmo Genético (AG), basado en la propuesta original, con un Optimizador por Enjambre de Partículas (PSO) para determinar la técnica más eficaz y fiable.

Este trabajo fue desarrollado como parte del proyecto final de la asignatura de **Algoritmos Genéticos**.

## Tecnologías Utilizadas

- **Python 3.x**
- **DEAP:** Para la implementación del Algoritmo Genético.
- **PySwarms:** Para la implementación del Optimizador por Enjambre de Partículas.
- **Pandas:** Para la manipulación y análisis de datos.
- **NumPy:** Para cálculos numéricos.
- **Matplotlib & Seaborn:** Para la visualización de resultados (gráficos de convergencia y boxplots).

## Estructura del Repositorio

```
.
├── dataset_sintetico_quinua_refinado.csv   # Dataset generado para los experimentos.
├── experimento.py                          # Script principal para ejecutar la comparativa N veces.
├── generador_datos.py                      # Script para generar el dataset sintético.
├── inno_pso.py                             # Implementación del optimizador PSO.
├── modelo.py                               # Define la función de utilidad a optimizar.
├── replica_ag.py                           # Implementación del Algoritmo Genético.
├── README.md                               # Este archivo.
└── ... (otros archivos como el informe, diapositivas, etc.)
```

## Instalación

Sigue estos pasos para configurar el entorno de desarrollo local.

1.  **Clona el repositorio:**

    ```sh
    git clone <URL-del-repositorio>
    cd <nombre-del-repositorio>
    ```

2.  **Crea un entorno virtual (recomendado):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    El proyecto requiere varias librerías. Puedes instalarlas todas usando el siguiente comando:

    ```sh
    pip install deap pyswarms pandas numpy matplotlib seaborn
    ```

## Uso

Puedes ejecutar los algoritmos de forma individual para una prueba rápida o ejecutar el experimento completo que realiza la comparativa.

- **Para ejecutar una sola prueba del Algoritmo Genético:**

  ```sh
  python replica_ag.py
  ```

- **Para ejecutar una sola prueba del Optimizador por Enjambre de Partículas:**

  ```sh
  python inno_pso.py
  ```

- **Para ejecutar el experimento comparativo completo (N ejecuciones):**
  Este es el script principal que genera la tabla de resumen y los gráficos comparativos.

  ```sh
  python experimento.py
  ```

## Autores

Este proyecto fue desarrollado por:

- Fabricio Balarezo Delgado
- Marco Ponce De Leon
- Juan Manuel Cari Quispe
