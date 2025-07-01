# Contenido para modelo.py

def calcular_utilidad(x1, x2):
  """
  Calcula la utilidad basada en la función objetivo del artículo.
  Max Z = 1.2*x1*x2 - 13.2*x1
  
  Args:
    x1 (float): La cantidad de tierra cultivada.
    x2 (float): El precio de venta.
    
  Returns:
    float: La utilidad calculada.
  """
  return 1.2 * x1 * x2 - 13.2 * x1