import numpy as np
from scipy.io import loadmat
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
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

def funcion_prediccion_mpc(muestras_perturbadas_std):
    muestras_perturbadas = scaler.inverse_transform(muestras_perturbadas_std)
    decisiones = []
    steps_4h, steps_12h = 8, 24
    for i, x_perturbed in enumerate(muestras_perturbadas):
        print(f"    Evaluando muestra hipotética {i+1}/{len(muestras_perturbadas)}...")
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

if __name__ == '__main__':
    TARGET_HOUR, SEARCH_WINDOW_HOURS, TARGET_MG_INDEX, TARGET_MG_NAME = 35, 5, 1, "60 Viviendas"
    best_params = {'num_samples': 79, 'perturbation_strength': 0.2198, 'kernel_width': 0.3164}
    print(f'--- PASO 1: Reconstruyendo estado para el bombeo de "{TARGET_MG_NAME}" en t≈{TARGET_HOUR}h ---')
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
    datos_sim_sub_P_dem = submuestreo_max(full_data['P_dem_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim)
    datos_sim_sub_P_gen = submuestreo_max(full_data['P_gen_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim)
    datos_sim_sub_Q_dem = submuestreo_max(full_data['Q_dem_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim)
    hist_completo_P_dem = np.vstack([full_data['hist_arranque']['P_dem'][0,0], datos_sim_sub_P_dem])
    hist_completo_P_gen = np.vstack([full_data['hist_arranque']['P_gen'][0,0], datos_sim_sub_P_gen])
    hist_completo_Q_dem = np.vstack([full_data['hist_arranque']['Q_dem'][0,0], datos_sim_sub_Q_dem])
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
    X_original = np.array([
        SoC_real[0]*100, SoC_real[1]*100, SoC_real[2]*100, V_tank_real[0], V_tank_real[1], V_tank_real[2], V_aq_real,
        np.mean(np.sum(P_dem_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_dem_pred_real[:steps_12h, :], axis=1)),
        np.mean(np.sum(P_gen_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_gen_pred_real[:steps_12h, :], axis=1)),
        np.sum(Q_dem_pred_real[:steps_4h, :])*Ts_mpc, np.sum(Q_dem_pred_real[:steps_12h, :])*Ts_mpc
    ])
    Qp_real = Q_p_log[idx_sim_mpc_start, TARGET_MG_INDEX]
    hora_evento = idx_sim_mpc_start * Ts_sim / 3600
    print(f'Instancia a explicar encontrada en t={hora_evento:.2f}h (Paso MPC {k_instance_mpc}).')
    print(f'Decisión real: Bombeo de {TARGET_MG_NAME} = {Qp_real:.4f} [L/s]')
    print('--- Reconstrucción completada. Iniciando análisis final... ---\n')
    print(f"--- Usando los mejores hiperparámetros encontrados (R²≈0.9996) ---")
    for key, value in best_params.items(): print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    print("\n--- EJECUTANDO ANÁLISIS FINAL DE LIME. Esto tardará unos minutos... ---")
    start_time = time.time()
    NUM_FEATURES_TO_EXPLAIN = 6
    scaler = StandardScaler()
    ruido = 1 + best_params['perturbation_strength'] * (2 * np.random.rand(best_params['num_samples'], len(X_original)) - 1)
    X_train_lime = X_original * ruido
    X_train_lime_std = scaler.fit_transform(X_train_lime)
    X_original_std = scaler.transform(X_original.reshape(1, -1))
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_lime_std,
        feature_names=['SoC MG1 (%)', 'SoC MG2 (%)', 'SoC MG3 (%)', 'V_tank MG1 (m3)', 'V_tank MG2 (m3)', 'V_tank MG3 (m3)', 'V_aq (m3)', 'P_dem_avg_4h (kW)', 'P_dem_avg_12h (kW)', 'P_gen_avg_4h (kW)', 'P_gen_avg_12h (kW)', 'Q_dem_sum_4h (L)', 'Q_dem_sum_12h (L)'],
        class_names=[f'Bombeo {TARGET_MG_NAME} (L/s)'],
        mode='regression', kernel_width=best_params['kernel_width'], discretize_continuous=False, verbose=False
    )
    explanation = explainer.explain_instance(data_row=X_original_std.flatten(), predict_fn=funcion_prediccion_mpc, num_features=NUM_FEATURES_TO_EXPLAIN, num_samples=best_params['num_samples'])
    end_time = time.time()
    print(f"\nAnálisis completado en {end_time - start_time:.2f} segundos.")
    print(f'\n--- EXPLICACIÓN DE LA DECISIÓN (TOP {NUM_FEATURES_TO_EXPLAIN} FACTORES) ---')
    print(f'La decisión de bombear {Qp_real:.4f} [L/s] para "{TARGET_MG_NAME}" se debió principalmente a los siguientes factores:\n')
    print('-' * 80)
    print(f'{"Característica":<35} | {"Coeficiente (w)":<25} | {"Influencia en Bombeo"}')
    print('-' * 80)
    explanation_list = explanation.as_list()
    for feature, weight in explanation_list:
        influencia = "AUMENTA" if weight > 0 else "REDUCE"
        print(f"{feature:<35} | {weight:<25.4e} | {influencia}")
    print('-' * 80)