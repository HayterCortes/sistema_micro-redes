# --- controlador_wrapper.py ---
import matlab.engine
import numpy as np
import os

def llamar_controlador_mpc(mg, soC_0, v_tank_0, v_aq_0, 
                           p_dem_pred, p_gen_pred, q_dem_pred, 
                           q_p_hist_0, p_mgref_hist_0,
                           k_mpc_actual, Q_p_hist_mpc):
    """
    Wrapper para llamar a la función de MATLAB 'controlador_mpc', ahora incluyendo
    todos los historiales necesarios para una ejecución fiel.
    """
    eng = None
    try:
        # Iniciar el motor de MATLAB
        eng = matlab.engine.start_matlab()

        # Añadir rutas necesarias al path de MATLAB
        project_path = os.getcwd()
        eng.addpath(eng.genpath(project_path), nargout=0)
        
        # --- Conversión de Datos de Python/NumPy a Tipos de MATLAB ---
        soC_0_m = matlab.double(soC_0.tolist())
        v_tank_0_m = matlab.double(v_tank_0.tolist())
        p_dem_pred_m = matlab.double(p_dem_pred.tolist())
        p_gen_pred_m = matlab.double(p_gen_pred.tolist())
        q_dem_pred_m = matlab.double(q_dem_pred.tolist())
        q_p_hist_0_m = matlab.double(q_p_hist_0.tolist())
        p_mgref_hist_0_m = matlab.double(p_mgref_hist_0.tolist())
        q_p_hist_mpc_m = matlab.double(Q_p_hist_mpc.tolist())

        # Llamar a la función de MATLAB con la firma completa y correcta
        u_opt_struct = eng.controlador_mpc(
            mg, soC_0_m, v_tank_0_m, v_aq_0,
            p_dem_pred_m, p_gen_pred_m, q_dem_pred_m,
            q_p_hist_0_m, 
            p_mgref_hist_0_m,
            float(k_mpc_actual), 
            q_p_hist_mpc_m,
            nargout=1
        )

        # Convertir la estructura de MATLAB de salida a un diccionario de Python
        if not u_opt_struct:
            return None
            
        u_opt_dict = {key: np.array(u_opt_struct[key]).flatten() for key in u_opt_struct}
        return u_opt_dict

    except Exception as e:
        print(f"Ocurrió un error en el wrapper de MATLAB: {e}")
        return None
    finally:
        # Asegurarse de que el motor de MATLAB siempre se cierre
        if eng:
            eng.quit()