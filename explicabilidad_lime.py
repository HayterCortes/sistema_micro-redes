import numpy as np
from scipy.io import loadmat
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import rbf_kernel
import time
import optuna

# Importar los wrappers de MATLAB
from controlador_wrapper import llamar_controlador_mpc
from prediccion_wrapper import llamar_generador_predicciones

# --- Variables Globales para la Función Objective ---
X_original = None
mg = None
P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real = None, None, None
q_p_hist_0_real, p_mgref_hist_0_real, k_instance_mpc, Q_p_hist_mpc_real = None, None, None, None
scaler = None # Scaler también se hará global para que la función de predicción lo use

def funcion_prediccion_mpc(muestras_perturbadas_std):
    """
    Función 'black-box' que LIME interroga. Ejecuta el MPC para cada muestra.
    """
    muestras_perturbadas = scaler.inverse_transform(muestras_perturbadas_std)
    decisiones = []
    
    steps_4h = 8
    steps_12h = 24

    for i, x_perturbed in enumerate(muestras_perturbadas):
        SoC_p = x_perturbed[0:3] / 100
        V_tank_p = x_perturbed[3:6]
        V_aq_p = x_perturbed[6]
        
        P_dem_pred_p = P_dem_pred_real.copy()
        P_gen_pred_p = P_gen_pred_real.copy()
        Q_dem_pred_p = Q_dem_pred_real.copy()

        if X_original[7] > 1e-6: P_dem_pred_p[:steps_4h, :] *= (x_perturbed[7] / X_original[7])
        if X_original[8] > 1e-6: P_dem_pred_p[:steps_12h, :] *= (x_perturbed[8] / X_original[8])
        if X_original[9] > 1e-6: P_gen_pred_p[:steps_4h, :] *= (x_perturbed[9] / X_original[9])
        if X_original[10] > 1e-6: P_gen_pred_p[:steps_12h, :] *= (x_perturbed[10] / X_original[10])
        if X_original[11] > 1e-6: Q_dem_pred_p[:steps_4h, :] *= (x_perturbed[11] / X_original[11])
        if X_original[12] > 1e-6: Q_dem_pred_p[:steps_12h, :] *= (x_perturbed[12] / X_original[12])
        
        u_opt = llamar_controlador_mpc(
            SoC_p, V_tank_p, V_aq_p,
            P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p,
            q_p_hist_0_real, p_mgref_hist_0_real, k_instance_mpc, Q_p_hist_mpc_real
        )

        if u_opt and 'Q_p' in u_opt and u_opt['Q_p'].size > 0:
            # --- CAMBIO: CASO DE ESTUDIO 1 ---
            # Explicar la decisión de la Microgrid 2 (60 Viviendas)
            decisiones.append(u_opt['Q_p'][1]) 
        else:
            decisiones.append(0)
    
    return np.array(decisiones).reshape(-1, 1)


def objective(trial):
    """
    Función objetivo que Optuna intentará maximizar.
    """
    global scaler
    
    # 1. Sugerir los hiperparámetros
    num_samples = trial.suggest_int('num_samples', 30, 80)
    perturbation_strength = trial.suggest_float('perturbation_strength', 0.15, 0.6)
    kernel_width = trial.suggest_float('kernel_width', 0.2, 0.9)

    # 2. Ejecutar el proceso de LIME
    ruido = 1 + perturbation_strength * (2 * np.random.rand(num_samples, len(X_original)) - 1)
    X_train_lime = X_original * ruido
    
    scaler = StandardScaler()
    X_train_lime_std = scaler.fit_transform(X_train_lime)
    X_original_std = scaler.transform(X_original.reshape(1, -1))
    
    y_labels = funcion_prediccion_mpc(X_train_lime_std)

    # 3. Ajustar modelo local y calcular fidelidad (R²)
    distances = np.linalg.norm(X_train_lime_std - X_original_std, axis=1)
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    lasso_model = Lasso(alpha=0.01)
    
    if np.all(y_labels == y_labels[0]): return 0.0

    lasso_model.fit(X_train_lime_std, y_labels, sample_weight=weights)
    y_pred_lime = lasso_model.predict(X_train_lime_std)
    
    fidelity_r2 = r2_score(y_labels, y_pred_lime, sample_weight=weights)
    
    return fidelity_r2


# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # --- CAMBIO: CASO DE ESTUDIO 1 ---
    # Definir los parámetros para la búsqueda del evento
    TARGET_HOUR = 35
    SEARCH_WINDOW_HOURS = 5
    TARGET_MG_INDEX = 1 # 0: 30 Viv, 1: 60 Viv, 2: Escuela
    TARGET_MG_NAME = "60 Viviendas"

    # --- PASOS 1, 2, 3: Carga, Selección y Reconstrucción ---
    print(f'--- PASO 1-3: Reconstruyendo estado para el bombeo de "{TARGET_MG_NAME}" en t≈{TARGET_HOUR}h ---')
    results = loadmat('results_mpc/resultados_mpc_3mg_7dias.mat')
    Q_p_log, P_grid_log, SoC_log, V_tank_log, V_aq_log = \
        results['Q_p'], results['P_grid'], results['SoC'], results['V_tank'], results['V_aq']
    mg = results['mg']
    #modelos_predictivos = loadmat('models/modelos_prediccion_AR.mat')['modelos']
    
    Ts_mpc = float(mg['Ts_mpc'][0,0][0,0])
    Ts_sim = float(mg['Ts_sim'][0,0][0,0])
    paso_mpc_en_sim = int(Ts_mpc / Ts_sim)
    max_lags = int(mg['max_lags_mpc'][0,0][0,0])
    N = int(mg['N'][0,0][0,0]
    )
    # --- CAMBIO: CASO DE ESTUDIO 1 ---
    # Lógica para encontrar el pico de bombeo en la ventana de tiempo deseada
    target_step_sim = int(TARGET_HOUR * 3600 / Ts_sim)
    window_sim_steps = int(SEARCH_WINDOW_HOURS * 3600 / Ts_sim)
    search_start = max(0, target_step_sim - window_sim_steps)
    search_end = min(len(Q_p_log), target_step_sim + window_sim_steps)

    pumping_window = Q_p_log[search_start:search_end, TARGET_MG_INDEX]
    if np.max(pumping_window) < 0.01:
        raise ValueError(f"No se encontró un evento de bombeo significativo para {TARGET_MG_NAME} cerca de la hora {TARGET_HOUR}.")
    
    local_peak_idx = np.argmax(pumping_window)
    k_instance_sim = search_start + local_peak_idx
    
    k_instance_mpc = int(np.floor((k_instance_sim - 1) / paso_mpc_en_sim) + 1)
    idx_sim_mpc_start = (k_instance_mpc - 1) * paso_mpc_en_sim
    
    # Reconstrucción (el resto de la lógica es la misma)
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
    if k_instance_mpc == 1:
        q_p_hist_0_real, p_mgref_hist_0_real = np.zeros(Q_p_log.shape[1]), np.zeros(P_grid_log.shape[1])
    else:
        idx_sim_mpc_anterior = (k_instance_mpc - 2) * paso_mpc_en_sim
        q_p_hist_0_real = Q_p_log[idx_sim_mpc_anterior, :]
        p_mgref_hist_0_real = P_grid_log[idx_sim_mpc_anterior, :]
        
    SoC_real = SoC_log[idx_sim_mpc_start, :]
    V_tank_real = V_tank_log[idx_sim_mpc_start, :]
    V_aq_real = V_aq_log[idx_sim_mpc_start, 0]
    steps_4h, steps_12h = 8, 24
    X_original = np.array([
        SoC_real[0]*100, SoC_real[1]*100, SoC_real[2]*100,
        V_tank_real[0], V_tank_real[1], V_tank_real[2], V_aq_real,
        np.mean(np.sum(P_dem_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_dem_pred_real[:steps_12h, :], axis=1)),
        np.mean(np.sum(P_gen_pred_real[:steps_4h, :], axis=1)), np.mean(np.sum(P_gen_pred_real[:steps_12h, :], axis=1)),
        np.sum(Q_dem_pred_real[:steps_4h, :])*Ts_mpc, np.sum(Q_dem_pred_real[:steps_12h, :])*Ts_mpc
    ])
    
    Qp_real = Q_p_log[idx_sim_mpc_start, TARGET_MG_INDEX]
    hora_evento = idx_sim_mpc_start * Ts_sim / 3600
    print(f'Instancia a explicar encontrada en t={hora_evento:.2f}h (Paso MPC {k_instance_mpc}).')
    print(f'Decisión real: Bombeo de {TARGET_MG_NAME} = {Qp_real:.4f} [L/s]')
    print('--- Reconstrucción completada. Iniciando optimización de hiperparámetros... ---\n')
    
    # --- EJECUCIÓN DEL ESTUDIO DE OPTIMIZACIÓN ---
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, n_jobs=2)
    
    # --- RESULTADOS DE LA OPTIMIZACIÓN ---
    print("\n\n--- OPTIMIZACIÓN DE HIPERPARÁMETROS COMPLETADA ---")
    print(f"Mejor valor de Fidelidad (R²): {study.best_value:.4f}")
    print("Mejores Hiperparámetros encontrados:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    
    # --- EJECUCIÓN FINAL DE LIME CON LOS MEJORES PARÁMETROS ---
    print("\n\n--- EJECUTANDO ANÁLISIS FINAL DE LIME CON PARÁMETROS ÓPTIMOS ---")
    NUM_FEATURES_TO_EXPLAIN = 6
    scaler = StandardScaler()
    ruido = 1 + best_params['perturbation_strength'] * (2 * np.random.rand(best_params['num_samples'], len(X_original)) - 1)
    X_train_lime = X_original * ruido
    X_train_lime_std = scaler.fit_transform(X_train_lime)
    X_original_std = scaler.transform(X_original.reshape(1, -1))
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_lime_std,
        feature_names=[
            'SoC MG1 (%)', 'SoC MG2 (%)', 'SoC MG3 (%)',
            'V_tank MG1 (m3)', 'V_tank MG2 (m3)', 'V_tank MG3 (m3)', 'V_aq (m3)',
            'P_dem_avg_4h (kW)', 'P_dem_avg_12h (kW)',
            'P_gen_avg_4h (kW)', 'P_gen_avg_12h (kW)',
            'Q_dem_sum_4h (L)', 'Q_dem_sum_12h (L)'
        ],
        class_names=[f'Bombeo {TARGET_MG_NAME} (L/s)'],
        mode='regression',
        discretize_continuous=False,
        kernel_width=best_params['kernel_width'],
        verbose=False
    )
    
    explanation = explainer.explain_instance(
        data_row=X_original_std.flatten(),
        predict_fn=funcion_prediccion_mpc,
        num_features=NUM_FEATURES_TO_EXPLAIN
    )
    
    print(f'\n--- EXPLICACIÓN DE LA DECISIÓN (TOP {NUM_FEATURES_TO_EXPLAIN} FACTORES) ---')
    print(f'La decisión de bombear {Qp_real:.4f} [L/s] para "{TARGET_MG_NAME}" se debió principalmente a los siguientes factores:\n')
    print('-' * 75)
    print(f'{"Característica":<25} | {"Coeficiente (w)":<20} | {"Influencia en Bombeo"}')
    print('-' * 75)
    explanation_list = explanation.as_list()
    for feature, weight in explanation_list:
        influencia = "➡️ AUMENTA" if weight > 0 else "⬅️ REDUCE"
        print(f"{feature:<25} | {weight:<20.4f} | {influencia}")
    print('-' * 75)