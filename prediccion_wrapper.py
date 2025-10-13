# --- prediccion_wrapper.py ---
import matlab.engine
import numpy as np
import os

def llamar_generador_predicciones(hist_data, N):
    """
    Wrapper para llamar a la función de MATLAB 'generar_predicciones_AR'.
    no necesita recibir los modelos, ya que la función de MATLAB los carga.
    """
    eng = None
    try:
        eng = matlab.engine.start_matlab()
        project_path = os.getcwd()
        eng.addpath(eng.genpath(project_path), nargout=0)
        
        hist_data_m = {
            'P_dem': matlab.double(hist_data['P_dem'].tolist()),
            'P_gen': matlab.double(hist_data['P_gen'].tolist()),
            'Q_dem': matlab.double(hist_data['Q_dem'].tolist())
        }
        
        # Se llama a la función sin el argumento 'modelos'
        p_dem_pred_m, p_gen_pred_m, q_dem_pred_m = eng.generar_predicciones_AR(
            hist_data_m,
            float(N),
            nargout=3
        )

        p_dem_pred = np.array(p_dem_pred_m)
        p_gen_pred = np.array(p_gen_pred_m)
        q_dem_pred = np.array(q_dem_pred_m)

        return p_dem_pred, p_gen_pred, q_dem_pred

    except Exception as e:
        print(f"Ocurrió un error en el wrapper de predicción de MATLAB: {e}")
        return None, None, None
    finally:
        if eng:
            eng.quit()