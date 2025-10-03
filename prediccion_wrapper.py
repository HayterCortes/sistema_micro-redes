# --- prediccion_wrapper.py ---
import matlab.engine
import numpy as np
import os

def llamar_generador_predicciones(modelos, hist_data, N):
    """
    Wrapper para llamar a la función de MATLAB 'generar_predicciones_AR'.

    Args:
        modelos (dict): El struct de modelos cargado desde el .mat.
        hist_data (dict): Un diccionario de Python con el historial de datos 
                          {'P_dem': np.array, 'P_gen': np.array, 'Q_dem': np.array}.
        N (int): El horizonte de predicción.

    Returns:
        tuple: Tres arrays de NumPy (p_dem_pred, p_gen_pred, q_dem_pred)
               o (None, None, None) si ocurre un error.
    """
    eng = None
    try:
        # Iniciar el motor de MATLAB
        eng = matlab.engine.start_matlab()

        # Añadir rutas necesarias al path de MATLAB
        project_path = os.getcwd()
        eng.addpath(eng.genpath(os.path.join(project_path, 'models')), nargout=0)
        eng.addpath(eng.genpath(os.path.join(project_path, 'utils')), nargout=0)
        
        # --- Conversión de Datos de Python/NumPy a Tipos de MATLAB ---
        # El historial se pasa como un struct de MATLAB
        hist_data_m = {
            'P_dem': matlab.double(hist_data['P_dem'].tolist()),
            'P_gen': matlab.double(hist_data['P_gen'].tolist()),
            'Q_dem': matlab.double(hist_data['Q_dem'].tolist())
        }
        
        # Llamar a la función de MATLAB, asegurando nargout=3
        p_dem_pred_m, p_gen_pred_m, q_dem_pred_m = eng.generar_predicciones_AR(
            modelos,
            hist_data_m,
            float(N), # Asegurar que N es un tipo numérico simple
            nargout=3
        )

        # Convertir las matrices de salida de MATLAB a arrays de NumPy
        p_dem_pred = np.array(p_dem_pred_m)
        p_gen_pred = np.array(p_gen_pred_m)
        q_dem_pred = np.array(q_dem_pred_m)

        return p_dem_pred, p_gen_pred, q_dem_pred

    except Exception as e:
        print(f"Ocurrió un error en el wrapper de predicción de MATLAB: {e}")
        return None, None, None
    finally:
        # Asegurarse de que el motor de MATLAB siempre se cierre
        if eng:
            eng.quit()