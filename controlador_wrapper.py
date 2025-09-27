import matlab.engine
import numpy as np
import os

def llamar_controlador_mpc(mg, soC_0, v_tank_0, v_aq_0, p_dem_pred, p_gen_pred, q_dem_pred, q_p_hist_0, k_mpc_actual, Q_p_hist_mpc):
    """
    Inicia el motor de MATLAB, llama al controlador MPC con la dinámica de pozo
    y devuelve la decisión en un formato amigable para Python.
    """
    eng = None
    try:
        # Iniciar el motor de MATLAB
        # print("Iniciando motor de MATLAB...") # Descomentar para depuración
        eng = matlab.engine.start_matlab()

        # Añadir las rutas necesarias al path de MATLAB
        project_path = os.getcwd()
        eng.addpath(eng.genpath(os.path.join(project_path, 'data')), nargout=0)
        eng.addpath(eng.genpath(os.path.join(project_path, 'models')), nargout=0)
        eng.addpath(eng.genpath(os.path.join(project_path, 'utils')), nargout=0)
        
        # --- Conversión de Datos de Python/NumPy a Tipos de MATLAB ---
        # Los escalares y structs/dicts suelen pasar bien, pero los arrays necesitan conversión.
        soC_0_m = matlab.double(soC_0.tolist())
        v_tank_0_m = matlab.double(v_tank_0.tolist())
        p_dem_pred_m = matlab.double(p_dem_pred.tolist())
        p_gen_pred_m = matlab.double(p_gen_pred.tolist())
        q_dem_pred_m = matlab.double(q_dem_pred.tolist())
        q_p_hist_0_m = matlab.double(q_p_hist_0.tolist())
        q_p_hist_mpc_m = matlab.double(Q_p_hist_mpc.tolist())

        # Llamar a la función de MATLAB
        u_opt_struct = eng.controlador_mpc(
            mg, soC_0_m, v_tank_0_m, v_aq_0,
            p_dem_pred_m, p_gen_pred_m, q_dem_pred_m,
            q_p_hist_0_m, float(k_mpc_actual), q_p_hist_mpc_m, # Asegurarse que k_mpc es un tipo numérico simple
            nargout=1
        )

        # Convertir la estructura de MATLAB de salida a un diccionario de Python
        if not u_opt_struct:
            return None
            
        # MATLAB a veces devuelve escalares dentro de una lista, .flatten() lo arregla.
        u_opt_dict = {key: np.array(u_opt_struct[key]).flatten() for key in u_opt_struct}
        return u_opt_dict

    except Exception as e:
        print(f"Ocurrió un error en el wrapper de MATLAB: {e}")
        return None
    finally:
        # Asegurarse de que el motor de MATLAB siempre se cierre
        if eng:
            eng.quit()