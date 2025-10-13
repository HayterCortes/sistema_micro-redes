import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, lars_path
import time
import itertools
import matplotlib.pyplot as plt
import multiprocessing 
# Importar los wrappers de MATLAB
from controlador_wrapper import llamar_controlador_mpc
from prediccion_wrapper import llamar_generador_predicciones

# --- Variables Globales y Funciones Auxiliares ---

X_original, mg, P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real = None, None, None, None, None
q_p_hist_0_real, p_mgref_hist_0_real, k_instance_mpc, Q_p_hist_mpc_real = None, None, None, None
scaler = None
feature_names = None

def funcion_prediccion_mpc_glime(muestras_no_escaladas, verbose=False):
    decisiones = []
    steps_4h, steps_12h = 8, 24
    for i, x_perturbed in enumerate(muestras_no_escaladas):
        if verbose: print(f"    Evaluando muestra {i+1}/{len(muestras_no_escaladas)}...")
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

def run_single_glime_explanation(num_samples, perturbation_strength, num_features_to_explain, verbose=False):
    stds = scaler.scale_ * perturbation_strength
    stds[stds == 0] = 1e-6
    covariance_matrix = np.diag(stds**2)
    generated_samples = np.random.multivariate_normal(mean=X_original, cov=covariance_matrix, size=num_samples)
    y_labels = funcion_prediccion_mpc_glime(generated_samples, verbose=verbose)
    X_train_std = scaler.transform(generated_samples)
    if np.all(y_labels == y_labels[0]): return [], []
    alphas, _, coefs = lars_path(X_train_std, y_labels.flatten(), method='lasso', verbose=False)
    for i, c in enumerate(coefs.T):
        if np.count_nonzero(c) >= num_features_to_explain: alpha_optim = alphas[i]; break
    else: alpha_optim = np.min(alphas) if alphas.size > 0 else 0.01
    lasso_model = Lasso(alpha=alpha_optim)
    lasso_model.fit(X_train_std, y_labels)
    return sorted(zip(feature_names, lasso_model.coef_), key=lambda x: abs(x[1]), reverse=True)

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2)); union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def find_elbow(x_data, y_data):
    points = np.array([x_data, y_data]).T; first_point = points[0]
    line_vec = points[-1] - first_point
    if np.linalg.norm(line_vec) == 0: return x_data[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = points - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(x_data), 1)), axis=1)
    vec_from_first_parallel = scalar_product.reshape(len(x_data), 1) * np.tile(line_vec_norm, (len(x_data), 1))
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    elbow_idx = np.argmax(dist_to_line)
    return x_data[elbow_idx]

def evaluate_single_sample(x_perturbed):
    """
    Función que evalúa UNA ÚNICA muestra. Esto es lo que ejecutará cada
    proceso trabajador en la piscina de paralelismo.
    """

    SoC_p, V_tank_p, V_aq_p = x_perturbed[0:3]/100, x_perturbed[3:6], x_perturbed[6]
    P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p = P_dem_pred_real.copy(), P_gen_pred_real.copy(), Q_dem_pred_real.copy()
    if X_original[7] > 1e-6: P_dem_pred_p[:steps_4h, :] *= (x_perturbed[7] / X_original[7])
    if X_original[8] > 1e-6: P_dem_pred_p[:steps_12h, :] *= (x_perturbed[8] / X_original[8])
    if X_original[9] > 1e-6: P_gen_pred_p[:steps_4h, :] *= (x_perturbed[9] / X_original[9])
    if X_original[10] > 1e-6: P_gen_pred_p[:steps_12h, :] *= (x_perturbed[10] / X_original[10])
    if X_original[11] > 1e-6: Q_dem_pred_p[:steps_4h, :] *= (x_perturbed[11] / X_original[11])
    if X_original[12] > 1e-6: Q_dem_pred_p[:steps_12h, :] *= (x_perturbed[12] / X_original[12])
    u_opt = llamar_controlador_mpc(SoC_p, V_tank_p, V_aq_p, P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p, q_p_hist_0_real, p_mgref_hist_0_real, k_instance_mpc, Q_p_hist_mpc_real)
    if u_opt and 'Q_p' in u_opt and u_opt['Q_p'].size > 0: return u_opt['Q_p'][1] 
    else: return 0.0

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)

    # --- Parámetros de Configuración ---
    TARGET_HOUR, SEARCH_WINDOW_HOURS, TARGET_MG_INDEX, TARGET_MG_NAME = 35, 5, 1, "60 Viviendas"
    PERTURBATION_STRENGTH = 0.25
    NUM_FEATURES_TO_EXPLAIN = 6
    
    # --- PASO 1: Reconstrucción del estado ---
    print(f'--- PASO 1: Reconstruyendo estado para el bombeo de "{TARGET_MG_NAME}" en t≈{TARGET_HOUR}h ---')
    results = loadmat('results_mpc/resultados_mpc_3mg_7dias.mat'); Q_p_log, P_grid_log, SoC_log, V_tank_log, V_aq_log = results['Q_p'], results['P_grid'], results['SoC'], results['V_tank'], results['V_aq']; mg = results['mg']
    Ts_mpc, Ts_sim, paso_mpc_en_sim, max_lags, N = float(mg['Ts_mpc'][0,0][0,0]), float(mg['Ts_sim'][0,0][0,0]), int(float(mg['Ts_mpc'][0,0][0,0]) / float(mg['Ts_sim'][0,0][0,0])), int(mg['max_lags_mpc'][0,0][0,0]), int(mg['N'][0,0][0,0])
    target_step_sim = int(TARGET_HOUR * 3600 / Ts_sim); window_sim_steps = int(SEARCH_WINDOW_HOURS * 3600 / Ts_sim)
    search_start, search_end = max(0, target_step_sim - window_sim_steps), min(len(Q_p_log), target_step_sim + window_sim_steps)
    pumping_window = Q_p_log[search_start:search_end, TARGET_MG_INDEX]
    if np.max(pumping_window) < 0.01: raise ValueError(f"No se encontró un evento de bombeo significativo.")
    k_instance_sim = search_start + np.argmax(pumping_window); k_instance_mpc = int(np.floor((k_instance_sim - 1) / paso_mpc_en_sim) + 1); idx_sim_mpc_start = (k_instance_mpc - 1) * paso_mpc_en_sim
    full_data = loadmat('utils/full_profiles_for_sim.mat'); from utils.submuestreo import submuestreo_max
    datos_sim_sub_P_dem, datos_sim_sub_P_gen, datos_sim_sub_Q_dem = submuestreo_max(full_data['P_dem_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim), submuestreo_max(full_data['P_gen_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim), submuestreo_max(full_data['Q_dem_sim'][:idx_sim_mpc_start+1, :], paso_mpc_en_sim)
    hist_completo_P_dem, hist_completo_P_gen, hist_completo_Q_dem = np.vstack([full_data['hist_arranque']['P_dem'][0,0], datos_sim_sub_P_dem]), np.vstack([full_data['hist_arranque']['P_gen'][0,0], datos_sim_sub_P_gen]), np.vstack([full_data['hist_arranque']['Q_dem'][0,0], datos_sim_sub_Q_dem])
    hist_data_real_para_AR = {'P_dem': hist_completo_P_dem[-max_lags:],'P_gen': hist_completo_P_gen[-max_lags:],'Q_dem': hist_completo_Q_dem[-max_lags:]}
    P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real = llamar_generador_predicciones(hist_data_real_para_AR, N)
    Q_p_decisiones_mpc = Q_p_log[0::paso_mpc_en_sim, :]; Q_p_hist_mpc_real = Q_p_decisiones_mpc[:k_instance_mpc, :]
    if k_instance_mpc == 1: q_p_hist_0_real, p_mgref_hist_0_real = np.zeros(Q_p_log.shape[1]), np.zeros(P_grid_log.shape[1])
    else: idx_sim_mpc_anterior = (k_instance_mpc - 2) * paso_mpc_en_sim; q_p_hist_0_real, p_mgref_hist_0_real = Q_p_log[idx_sim_mpc_anterior, :], P_grid_log[idx_sim_mpc_anterior, :]
    SoC_real, V_tank_real, V_aq_real = SoC_log[idx_sim_mpc_start, :], V_tank_log[idx_sim_mpc_start, :], V_aq_log[idx_sim_mpc_start, 0]
    steps_4h, steps_12h = 8, 24
    X_original = np.array([ SoC_real[0]*100, SoC_real[1]*100, SoC_real[2]*100, V_tank_real[0], V_tank_real[1], V_tank_real[2], V_aq_real, np.mean(np.sum(P_dem_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_dem_pred_real[:steps_12h, :], axis=1)), np.mean(np.sum(P_gen_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_gen_pred_real[:steps_12h, :], axis=1)), np.sum(Q_dem_pred_real[:steps_4h, :])*Ts_mpc, np.sum(Q_dem_pred_real[:steps_12h, :])*Ts_mpc ])
    feature_names = ['SoC MG1 (%)', 'SoC MG2 (%)', 'SoC MG3 (%)', 'V_tank MG1 (m3)', 'V_tank MG2 (m3)', 'V_tank MG3 (m3)', 'V_aq (m3)', 'P_dem_avg_4h (kW)', 'P_dem_avg_12h (kW)', 'P_gen_avg_4h (kW)', 'P_gen_avg_12h (kW)', 'Q_dem_sum_4h (L)', 'Q_dem_sum_12h (L)']
    scaler_fit_samples = X_original * (1 + PERTURBATION_STRENGTH * (2 * np.random.rand(1000, len(X_original)) - 1)); scaler = StandardScaler().fit(scaler_fit_samples)
    print('--- Reconstrucción completada. ---\n')
    
    # --- FASE DE CALIBRACIÓN ---
    print("--- INICIANDO FASE DE CALIBRACIÓN (RÁPIDA) PARA ENCONTRAR N ÓPTIMO ---")
    sample_sizes_to_test = [10]; num_repetitions = 10; final_stability_scores = []
    for n_samples in sample_sizes_to_test:
        print(f"\n--- Evaluando estabilidad para n_samples = {n_samples} ---")
        feature_sets = []
        for i in range(num_repetitions):
            print(f"  Repetición {i+1}/{num_repetitions}...")
            coef_pairs = run_single_glime_explanation(n_samples, PERTURBATION_STRENGTH, NUM_FEATURES_TO_EXPLAIN)
            top_features = {feature for feature, weight in coef_pairs[:NUM_FEATURES_TO_EXPLAIN]}
            if top_features: feature_sets.append(top_features)
        if len(feature_sets) < 2: avg_jaccard_score = 0.0
        else: pairwise_jaccards = [jaccard_similarity(s1, s2) for s1, s2 in itertools.combinations(feature_sets, 2)]; avg_jaccard_score = np.mean(pairwise_jaccards) if pairwise_jaccards else 0.0
        final_stability_scores.append(avg_jaccard_score)
        print(f"Estabilidad (Jaccard Promedio) para {n_samples} muestras: {avg_jaccard_score:.4f}")
    n_optimo = find_elbow(sample_sizes_to_test, final_stability_scores)
    print(f"\n--- CALIBRACIÓN COMPLETADA ---"); print(f"El número óptimo de muestras (n_optimo) se ha determinado en: {n_optimo}")
    plt.figure(figsize=(10, 6)); plt.plot(sample_sizes_to_test, final_stability_scores, marker='o', label='Estabilidad (Jaccard Promedio)'); plt.axvline(x=n_optimo, color='r', linestyle='--', label=f'N Óptimo (Codo) = {n_optimo}'); plt.xlabel("Número de Muestras"); plt.ylabel("Estabilidad (Jaccard Promedio)"); plt.title("Análisis de Convergencia GLIME"); plt.legend(); plt.grid(True); plt.savefig("convergencia_estabilidad_glime.png")
    print("Gráfico de convergencia guardado como 'convergencia_estabilidad_glime.png'.")

    # --- FASE DE EXPLICACIÓN FINAL ---
    print(f"\n\n--- INICIANDO EXPLICACIÓN FINAL PARALELIZADA CON N ÓPTIMO = {n_optimo} ---")
    start_time = time.time()
    stds = scaler.scale_ * PERTURBATION_STRENGTH; stds[stds == 0] = 1e-6
    covariance_matrix = np.diag(stds**2)
    generated_samples = np.random.multivariate_normal(mean=X_original, cov=covariance_matrix, size=n_optimo)
    
    # --- PARALELIZACIÓN CON MULTIPROCESSING.POOL ---
    num_procesos = 3
    print(f"Evaluando {n_optimo} muestras en paralelo usando {num_procesos} núcleos...")
    with multiprocessing.Pool(processes=num_procesos) as pool:
        results = pool.map(evaluate_single_sample, generated_samples, chunksize=max(1, n_optimo // (num_procesos * 2)))
    
    y_labels = np.array(results).reshape(-1, 1)
    print("Evaluación paralela completada.")
    
    X_train_std = scaler.transform(generated_samples)
    alphas, _, coefs = lars_path(X_train_std, y_labels.flatten(), method='lasso', verbose=False)
    for i, c in enumerate(coefs.T):
        if np.count_nonzero(c) >= NUM_FEATURES_TO_EXPLAIN: alpha_optim = alphas[i]; break
    else: alpha_optim = np.min(alphas) if alphas.size > 0 else 0.01
    lasso_model = Lasso(alpha=alpha_optim)
    lasso_model.fit(X_train_std, y_labels)
    final_coef_pairs = sorted(zip(feature_names, lasso_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    end_time = time.time()
    print(f"Análisis final completado en {end_time - start_time:.2f} segundos.")
    
    Qp_real = Q_p_log[idx_sim_mpc_start, TARGET_MG_INDEX]
    print(f'\n--- EXPLICACIÓN GLIME (TOP {NUM_FEATURES_TO_EXPLAIN} FACTORES) ---')
    print(f'La decisión de bombear {Qp_real:.4f} [L/s] para "{TARGET_MG_NAME}" se debió a los siguientes factores:\n')
    print('-' * 80)
    print(f'{"Característica":<35} | {"Coeficiente (w)":<25} | {"Influencia en Bombeo"}')
    print('-' * 80)
    for feature, weight in final_coef_pairs[:NUM_FEATURES_TO_EXPLAIN]:
        if abs(weight) > 1e-9: influencia = "➡️ AUMENTA" if weight > 0 else "⬅️ REDUCE"; print(f"{feature:<35} | {weight:<25.4e} | {influencia}")
    print('-' * 80)