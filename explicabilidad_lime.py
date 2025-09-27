import numpy as np
from scipy.io import loadmat
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
from controlador_wrapper import llamar_controlador_mpc
import time

def run_lime_explanation():
    """
    Script principal para ejecutar el análisis de explicabilidad con LIME,
    interactuando con la simulación de MATLAB.
    """
    # --- CONFIGURACIÓN ---
    num_samples = 50  # Número de muestras para la aproximación LIME
    perturbation_strength = 0.5  # Perturbación (+/- 25% uniforme)
    
    print('Iniciando análisis LIME para el sistema de gestión de microrredes...\n')

    # --- PASO 0: CARGAR RESULTADOS E IDENTIFICAR INSTANCIA ---
    print('--- PASO 0: Identificando la instancia a explicar ---')
    results = loadmat('results_mpc/resultados_mpc_3mg_7dias.mat')
    Q_p_log = results['Q_p']
    SoC_log = results['SoC']
    V_tank_log = results['V_tank']
    V_aq_log = results['V_aq']
    mg = results['mg']
    
    # Lógica de búsqueda (traducción de MATLAB)
    Ts_mpc = mg['Ts_mpc'][0,0][0,0]
    Ts_sim = mg['Ts_sim'][0,0][0,0]
    paso_mpc_en_sim = Ts_mpc / Ts_sim
    
    k_instance_sim_list = np.where(Q_p_log[:, 2] > 0.01)[0]
    if len(k_instance_sim_list) == 0:
        raise ValueError("No se encontró ningún evento de bombeo significativo.")
    k_instance_sim = k_instance_sim_list[0]
    k_instance_mpc = np.floor((k_instance_sim - 1) / paso_mpc_en_sim) * paso_mpc_en_sim + 1
    
    # El índice en Python es k-1
    idx = int(k_instance_mpc) - 1
    Qp_real = Q_p_log[idx, 2]
    hora_evento = idx * Ts_sim / 3600
    
    print(f'Instancia encontrada en la hora {hora_evento:.2f} (paso de simulación k={idx+1}).')
    print(f'Decisión a explicar: Bombeo de la Escuela Q_p = {Qp_real:.4f} [L/s]\n')

    # --- PASO 1: EXTRAER CARACTERÍSTICAS DE LA INSTANCIA ORIGINAL ---
    print('--- PASO 1: Aislando datos y extrayendo características ---')
    # Cargar datos históricos para generar predicciones (traducción de MATLAB)
    # (Este bloque es una simplificación, asume que los perfiles ya están en un .mat)
    profiles = loadmat('utils/full_profiles.mat') # Asumimos que creaste este .mat por conveniencia
    P_dem_ts = profiles['P_dem_ts']
    P_gen_ts = profiles['P_gen_ts']
    Q_dem_ts = profiles['Q_dem_ts']
    
    # Generar predicciones para la instancia (esto se podría hacer llamando a generar_predicciones_AR.m)
    # Por simplicidad aquí, cargamos las predicciones si ya estuvieran guardadas
    # [P_dem_pred, P_gen_pred, Q_dem_pred] = ...

    # Suponemos que ya tenemos las predicciones. Extraemos las características.
    SoC_real = SoC_log[idx, :]
    V_tank_real = V_tank_log[idx, :]
    V_aq_real = V_aq_log[idx, 0]
    
    # Placeholder para las predicciones, en una implementación real se deben generar
    N = int(mg['N'][0,0][0,0])
    num_mg = SoC_real.shape[0]
    P_dem_pred = np.random.rand(N, num_mg) * 50
    P_gen_pred = np.random.rand(N, num_mg) * 20
    Q_dem_pred = np.random.rand(N, num_mg) * 1.5
    
    feature_names = [
        'SoC MG1 (%)', 'SoC MG2 (%)', 'SoC MG3 (%)',
        'V_tank MG1 (m3)', 'V_tank MG2 (m3)', 'V_tank MG3 (m3)',
        'V_aq (m3)',
        'P_dem_avg_24h (kW)', 'P_gen_avg_24h (kW)', 'Q_dem_sum_24h (L)'
    ]
    
    X_original = np.array([
        SoC_real[0] * 100, SoC_real[1] * 100, SoC_real[2] * 100,
        V_tank_real[0], V_tank_real[1], V_tank_real[2],
        V_aq_real,
        np.mean(np.sum(P_dem_pred, axis=1)),
        np.mean(np.sum(P_gen_pred, axis=1)),
        np.sum(Q_dem_pred) * Ts_mpc
    ])
    print("Vector de características original extraído.\n")

    # --- PASO 2 y 3: DEFINIR FUNCIÓN PREDICTIVA Y GENERAR DATOS ---
    print('--- PASO 2 y 3: Preparando el entorno LIME ---')
    
    def funcion_prediccion_mpc(muestras_perturbadas_std):
        # La función predictiva debe trabajar con los datos en la escala original.
        # Por lo tanto, invertimos la estandarización.
        muestras_perturbadas = scaler.inverse_transform(muestras_perturbadas_std)
        
        decisiones = []
        
        for i, x_perturbed in enumerate(muestras_perturbadas):
            print(f"  Etiquetando muestra {i+1}/{len(muestras_perturbadas)}...")
            # Reconstruir los inputs del MPC desde el vector de características perturbado
            SoC_p = x_perturbed[0:3] / 100
            V_tank_p = x_perturbed[3:6]
            V_aq_p = x_perturbed[6]
            
            # Escalar las predicciones base según la perturbación
            P_dem_pred_p = P_dem_pred * (x_perturbed[7] / X_original[7])
            P_gen_pred_p = P_gen_pred * (x_perturbed[8] / X_original[8])
            Q_dem_pred_p = Q_dem_pred * (x_perturbed[9] / X_original[9])
            
            # Placeholder para los parámetros de descenso de pozo (k_mpc y Q_p_hist)
            k_mpc_actual = 100 # Ejemplo
            Q_p_hist_mpc = np.zeros((k_mpc_actual, num_mg))
            q_p_hist_0 = np.zeros(num_mg)

            # Llamada al wrapper que ejecuta el código de MATLAB
            u_opt = llamar_controlador_mpc(
                mg, SoC_p, V_tank_p, V_aq_p,
                P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p,
                q_p_hist_0, k_mpc_actual, Q_p_hist_mpc
            )

            if u_opt and 'Q_p' in u_opt and u_opt['Q_p'].size > 0:
                decisiones.append(u_opt['Q_p'][2]) # Bombeo de la Escuela (MG3)
            else:
                decisiones.append(0) # Valor por defecto si el MPC falla
        
        return np.array(decisiones).reshape(-1, 1)

    # Generar un vecindario para entrenar el scaler y el explicador
    ruido = 1 + perturbation_strength * (2 * np.random.rand(num_samples, len(X_original)) - 1)
    X_train = X_original * ruido
    
    # --- PASO 4: ESTANDARIZAR Y ENTRENAR EXPLICADOR LIME ---
    print('\n--- PASO 4: Entrenando el modelo explicativo ---')
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=scaler.transform(X_train),
        feature_names=feature_names,
        class_names=['Bombeo Q_p (L/s)'],
        mode='regression',
        discretize_continuous=False,
        verbose=False
    )
    
    # Estandarizar la instancia original que queremos explicar
    X_original_std = scaler.transform(X_original.reshape(1, -1))
    
    print("Generando explicación para la instancia de interés...")
    start_time = time.time()
    
    explanation = explainer.explain_instance(
        data_row=X_original_std.flatten(),
        predict_fn=funcion_prediccion_mpc,
        num_features=len(feature_names)
    )
    
    end_time = time.time()
    print(f"Explicación generada en {end_time - start_time:.2f} segundos.\n")
    
    # --- PASO 5: INTERPRETAR LOS RESULTADOS ---
    print('--- PASO 5: EXPLICACIÓN DE LA DECISIÓN DE BOMBEO ---')
    print(f'La decisión de bombear {Qp_real:.4f} [L/s] se debió a la siguiente combinación de factores:\n')
    print('----------------------------------------------------------------------------------')
    print(f'{"Característica":<25} | {"Coeficiente (w)":<20} | {"Influencia":<20}')
    print('----------------------------------------------------------------------------------')
    
    explanation_list = sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, weight in explanation_list:
        influencia = "Neutra"
        if weight > 1e-3:
            influencia = "AUMENTA el bombeo"
        elif weight < -1e-3:
            influencia = "REDUCE el bombeo"
        print(f"{feature:<25} | {weight:<20.4f} | {influencia:<20}")
    print('----------------------------------------------------------------------------------')


if __name__ == '__main__':
    run_lime_explanation()