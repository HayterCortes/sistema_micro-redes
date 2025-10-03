# --- utils/submuestreo.py ---

import numpy as np

def submuestreo_max(datos, tamano_ventana):
    """
    Submuestrea una o varias series temporales (por columnas) tomando el 
    valor máximo de cada ventana. Es una traducción de la función de MATLAB.

    Args:
        datos (np.ndarray): Array de NumPy (o lista) con los datos.
        tamano_ventana (int): El número de muestras a agrupar en una ventana.

    Returns:
        np.ndarray: El array de datos submuestreado.
    """
    # Asegurar que los datos sean un array de NumPy
    datos = np.asarray(datos)
    if datos.ndim == 1:
        datos = datos.reshape(-1, 1) # Convertir vector 1D a matriz de una columna

    num_series = datos.shape[1]
    num_muestras = datos.shape[0]

    num_ventanas = num_muestras // tamano_ventana
    if num_ventanas == 0:
        return np.array([]).reshape(0, num_series) # Retornar array vacío si no cabe ni una ventana

    # Truncar los datos para que sean un múltiplo exacto del tamaño de la ventana
    longitud_ajustada = num_ventanas * tamano_ventana
    datos_ajustados = datos[:longitud_ajustada, :]
    
    # El paso clave es el orden 'F' (estilo Fortran/MATLAB) para que el reshape
    # agrupe los datos por columnas, igual que en MATLAB.
    # La forma resultante es (tamano_ventana, num_ventanas, num_series)
    matriz_datos = datos_ajustados.reshape(tamano_ventana, num_ventanas, num_series, order='F')
    
    # Calcular el máximo a lo largo del eje 0 (a través de las filas de cada ventana)
    datos_submuestreados = np.max(matriz_datos, axis=0)
    
    return datos_submuestreados