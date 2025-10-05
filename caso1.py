# --- run_lime_case1_glime.py ---

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import time

# Importar los wrappers de MATLAB
from controlador_wrapper import llamar_controlador_mpc
from prediccion_wrapper import llamar_generador_predicciones

# --- Variables Globales ---
X_original = None
mg = None
P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real = None, None, None
q_p_hist_0_real, p_mgref_hist_0_real, k_instance_mpc, Q_p_hist_mpc_real = None, None, None, None
scaler = None

def funcion_prediccion_mpc_glime(muestras_no_escaladas):
    """
    Función 'black-box' modificada para GLIME.
    Recibe datos NO escalados y los escala antes de la simulación.
    """
    decisiones = []
    muestras_escaladas = scaler.transform(muestras_no_escaladas)
    
    steps_4h, steps_12h = 8, 24

    for i, x_perturbed in enumerate(muestras_no_escaladas):
        print(f"    Evaluando muestra GLIME {i+1}/{len(muestras_no_escaladas)}...")
        
        SoC_p, V_tank_p, V_aq_p = x_perturbed[0:3]/100, x_perturbed[3:6], x_perturbed[6]
        P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p = P_dem_pred_real.copy(), P_gen_pred_real.copy(), Q_dem_pred_real.copy()

        if X_original[7] > 1e-6: P_dem_pred_p[:steps_4h, :] *= (x_perturbed[7] / X_original[7])
        if X_original[8] > 1e-6: P_dem_pred_p[:steps_12h, :] *= (x_perturbed[8] / X_original[8])
        if X_original[9] > 1e-6: P_gen_pred_p[:steps_4h, :] *= (x_perturbed[9] / X_original[9])
        if X_original[10] > 1e-6: P_gen_pred_p[:steps_12h, :] *= (x_perturbed[10] / X_original[10])
        if X_original[11] > 1e-6: Q_dem_pred_p[:steps_4h, :] *= (x_perturbed[11] / X_original[11])
        if X_original[12] > 1e-6: Q_dem_pred_p[:steps_12h, :] *= (x_perturbed[12] / X_original[12])
        
        u_opt = llamar_controlador_mpc(SoC_p, V_tank_p, V_aq_p, P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p, q_p_hist_0_real, p_mgref_hist_0_real, k_instance_mpc, Q_p_hist_mpc_real)

        if u_opt and 'Q_p' in u_opt and u_opt['Q_p'].size > 0: decisiones.append(u_opt['Q_p'][1]) 
        else: decisiones.append(0)
    
    return np.array(decisiones).reshape(-1, 1)

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == '__main__':
    TARGET_HOUR, SEARCH_WINDOW_HOURS, TARGET_MG_INDEX, TARGET_MG_NAME = 35, 5, 1, "60 Viviendas"
    
    # Usaremos los hiperparámetros optimizados, pero ahora `perturbation_strength`
    # controlará la desviación estándar del muestreo Gaussiano.
    best_params = {'num_samples': 150, 'perturbation_strength': 0.25} 
    NUM_FEATURES_TO_EXPLAIN = 6

    print(f'--- PASO 1: Reconstruyendo estado para el bombeo de "{TARGET_MG_NAME}" en t≈{TARGET_HOUR}h ---')
    # ... (La lógica de reconstrucción de estado es idéntica a la versión anterior)
    results = loadmat('results_mpc/resultados_mpc_3mg_7dias.mat')
    Q_p_log, P_grid_log, SoC_log, V_tank_log, V_aq_log = results['Q_p'], results['P_grid'], results['SoC'], results['V_tank'], results['V_aq']
    mg = results['mg']
    Ts_mpc, Ts_sim, paso_mpc_en_sim, max_lags, N = float(mg['Ts_mpc'][0,0][0,0]), float(mg['Ts_sim'][0,0][0,0]), int(float(mg['Ts_mpc'][0,0][0,0]) / float(mg['Ts_sim'][0,0][0,0])), int(mg['max_lags_mpc'][0,0][0,0]), int(mg['N'][0,0][0,0])
    target_step_sim = int(TARGET_HOUR * 3600 / Ts_sim)
    window_sim_steps = int(SEARCH_WINDOW_HOURS * 3600 / Ts_sim)
    search_start, search_end = max(0, target_step_sim - window_sim_steps), min(len(Q_p_log), target_step_sim + window_sim_steps)
    pumping_window = Q_p_log[search_start:search_end, TARGET_MG_INDEX]
    if np.max(pumping_window) < 0.01: raise ValueError(f"No se encontró un evento de bombeo significativo.")
    k_instance_sim = search_start + np.argmax(pumping_window)
    k_instance_mpc = int(np.floor((k_instance_sim - 1) / paso_mpc_en_sim) + 1)
    idx_sim_mpc_start = (k_instance_mpc - 1) * paso_mpc_en_sim
    full_data = loadmat('utils/full_profiles_for_sim.mat')
    from utils.submuestreo import submuestreo_max
    datos_sim_sub_P_dem, datos_sim_sub_P_gen, datos_sim_sub_Q_dem = submuestreo_max(full_data['P_dem_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim), submuestreo_max(full_data['P_gen_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim), submuestreo_max(full_data['Q_dem_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim)
    hist_completo_P_dem, hist_completo_P_gen, hist_completo_Q_dem = np.vstack([full_data['hist_arranque']['P_dem'][0,0], datos_sim_sub_P_dem]), np.vstack([full_data['hist_arranque']['P_gen'][0,0], datos_sim_sub_P_gen]), np.vstack([full_data['hist_arranque']['Q_dem'][0,0], datos_sim_sub_Q_dem])
    hist_data_real_para_AR = {'P_dem': hist_completo_P_dem[-max_lags:],'P_gen': hist_completo_P_gen[-max_lags:],'Q_dem': hist_completo_Q_dem[-max_lags:]}
    P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real = llamar_generador_predicciones(hist_data_real_para_AR, N)
    Q_p_decisiones_mpc = Q_p_log[0::paso_mpc_en_sim, :]
    Q_p_hist_mpc_real = Q_p_decisiones_mpc[:k_instance_mpc, :]
    if k_instance_mpc == 1: q_p_hist_0_real, p_mgref_hist_0_real = np.zeros(Q_p_log.shape[1]), np.zeros(P_grid_log.shape[1])
    else:
        idx_sim_mpc_anterior = (k_instance_mpc - 2) * paso_mpc_en_sim
        q_p_hist_0_real, p_mgref_hist_0_real = Q_p_log[idx_sim_mpc_anterior, :], P_grid_log[idx_sim_mpc_anterior, :]
    SoC_real, V_tank_real, V_aq_real = SoC_log[idx_sim_mpc_start, :], V_tank_log[idx_sim_mpc_start, :], V_aq_log[idx_sim_mpc_start, 0]
    steps_4h, steps_12h = 8, 24
    X_original = np.array([ SoC_real[0]*100, SoC_real[1]*100, SoC_real[2]*100, V_tank_real[0], V_tank_real[1], V_tank_real[2], V_aq_real, np.mean(np.sum(P_dem_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_dem_pred_real[:steps_12h, :], axis=1)), np.mean(np.sum(P_gen_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_gen_pred_real[:steps_12h, :], axis=1)), np.sum(Q_dem_pred_real[:steps_4h, :])*Ts_mpc, np.sum(Q_dem_pred_real[:steps_12h, :])*Ts_mpc ])
    Qp_real = Q_p_log[idx_sim_mpc_start, TARGET_MG_INDEX]
    hora_evento = idx_sim_mpc_start * Ts_sim / 3600
    print(f'Instancia a explicar encontrada en t={hora_evento:.2f}h (Paso MPC {k_instance_mpc}).')
    print(f'Decisión real: Bombeo de {TARGET_MG_NAME} = {Qp_real:.4f} [L/s]')
    print('--- Reconstrucción completada. Iniciando análisis GLIME... ---\n')
    
    # --- EJECUCIÓN DEL ANÁLISIS GLIME ---
    start_time = time.time()
    feature_names = ['SoC MG1 (%)', 'SoC MG2 (%)', 'SoC MG3 (%)', 'V_tank MG1 (m3)', 'V_tank MG2 (m3)', 'V_tank MG3 (m3)', 'V_aq (m3)', 'P_dem_avg_4h (kW)', 'P_dem_avg_12h (kW)', 'P_gen_avg_4h (kW)', 'P_gen_avg_12h (kW)', 'Q_dem_sum_4h (L)', 'Q_dem_sum_12h (L)']
    
    # --- CAMBIO: IMPLEMENTACIÓN GLIME ---
    # 1. Generar muestras de una distribución Gaussiana centrada en la instancia original.
    print(f"Generando {best_params['num_samples']} muestras con el enfoque GLIME-Gauss...")
    stds = scaler.fit(X_original.reshape(1, -1)).scale_ * best_params['perturbation_strength']
    # Evitar desviación estándar de cero para características constantes
    stds[stds == 0] = 1e-6
    covariance_matrix = np.diag(stds**2)
    generated_samples = np.random.multivariate_normal(
        mean=X_original,
        cov=covariance_matrix,
        size=best_params['num_samples']
    )
    
    # 2. Obtener las predicciones del MPC para estas muestras
    y_labels = funcion_prediccion_mpc_glime(generated_samples)
    
    # 3. Escalar los datos y ajustar un modelo LASSO (sin ponderación de kernel)
    X_train_std = scaler.transform(generated_samples)
    # Buscamos el mejor alfa para LASSO que nos de K características
    from sklearn.linear_model import lars_path
    alphas, _, coefs = lars_path(X_train_std, y_labels.flatten(), method='lasso', verbose=False)
    
    # Encontrar el alfa que nos da K (o un poco más) características no nulas
    for i, c in enumerate(coefs.T):
        if np.count_nonzero(c) >= NUM_FEATURES_TO_EXPLAIN:
            alpha_optim = alphas[i]
            break
    else:
        alpha_optim = np.min(alphas)

    lasso_model = Lasso(alpha=alpha_optim)
    lasso_model.fit(X_train_std, y_labels)
    
    end_time = time.time()
    print(f"\nAnálisis GLIME completado en {end_time - start_time:.2f} segundos.")
    
    # --- Interpretación de Resultados ---
    print(f'\n--- EXPLICACIÓN GLIME (TOP {NUM_FEATURES_TO_EXPLAIN} FACTORES) ---')
    print(f'La decisión de bombear {Qp_real:.4f} [L/s] para "{TARGET_MG_NAME}" se debió a los siguientes factores:\n')
    print('-' * 80)
    print(f'{"Característica":<35} | {"Coeficiente (w)":<25} | {"Influencia en Bombeo"}')
    print('-' * 80)
    
    # Ordenar los coeficientes por su magnitud para mostrar los más importantes
    coef_pairs = sorted(zip(feature_names, lasso_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, weight in coef_pairs[:NUM_FEATURES_TO_EXPLAIN]:
        if abs(weight) > 1e-9: # Ignorar los que son numéricamente cero
            influencia = "➡️ AUMENTA" if weight > 0 else "⬅️ REDUCE"
            print(f"{feature:<35} | {weight:<25.4e} | {influencia}")
    print('-' * 80)