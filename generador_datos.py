# generador_datos.py
import numpy as np
import pandas as pd

def generar_datos_sinteticos_refinados(num_muestras=200):
    """
    Genera un conjunto de datos sintéticos refinado, aplicando límites
    inferiores y superiores a la variable x2.
    """
    
    # Constante de la competencia mencionada en el artículo
    X4_COMPETENCIA = 0.25
    
    lista_muestras = []
    
    while len(lista_muestras) < num_muestras:
        # 1. Generar candidatos para x1 y x3
        x1_tierra = np.random.uniform(1, 100)
        x3_demanda = np.random.uniform(1, 150)
        
        # 2. Calcular el límite inferior para x2 (de 2*x1 - x2/x3 <= 0)
        limite_inferior_x2 = 11;
        
        # 3. Calcular el límite superior para x2 (de la fórmula del mercado)
        # Evitamos la división por cero, aunque x3 > 1
        if x3_demanda == 0:
            continue
            
        termino_de_mercado = 34 - (20.4 * x1_tierra) / (x3_demanda);
        limite_superior_x2 = termino_de_mercado + 17 * X4_COMPETENCIA;
        
        # 4. Validar que el rango sea coherente (superior > inferior)
        if limite_superior_x2 > limite_inferior_x2:
            # 5. Generar un valor de x2 dentro del rango válido
            x2_precio = np.random.uniform(limite_inferior_x2, limite_superior_x2)
            
            # Añadir la muestra válida a la lista
            lista_muestras.append({
                'tierra_cultivada_x1': x1_tierra,
                'precio_venta_x2': x2_precio,
                'demanda_x3': x3_demanda
            })
            
    # Convertir la lista de diccionarios a un DataFrame
    df = pd.DataFrame(lista_muestras)
    return df

# Ejemplo de uso:
if __name__ == '__main__':
    datos_refinados = generar_datos_sinteticos_refinados(200)
    print("Se generaron los siguientes datos sintéticos refinados:")
    print(datos_refinados.head())
    
    print(f"\nSe generaron {len(datos_refinados)} muestras válidas.")

    # Guardar los datos
    datos_refinados.to_csv('dataset_sintetico_quinua_refinado.csv', index=False)
    print("\nDatos guardados en 'dataset_sintetico_quinua_refinado.csv'")