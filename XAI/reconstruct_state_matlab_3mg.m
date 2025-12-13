%% --- Archivo: reconstruct_state_matlab_3mg.m ---
%
% Reconstruye el estado COMPLETO del sistema de 3 Micro-redes.
% Crea un vector de características aplanado (X_original) de dimensión 16:
% [MG1_Feats (5), MG2_Feats (5), MG3_Feats (5), V_aq (1)]
%
% ** ACTUALIZACIÓN: Guarda los valores reales de Q_t para las 3 MGs.
%--------------------------------------------------------------------------
function [estado, params] = reconstruct_state_matlab_3mg(K_GLOBAL_TARGET)

    % --- 1. Cargar Datos y Resultados ---
    try
        results = load('results_mpc/resultados_mpc_3mg_7dias.mat');
        mg = results.mg; 
    catch
        error('No se encontró "results_mpc/resultados_mpc_3mg_7dias.mat". Ejecute main_mpc.m primero.');
    end
    
    try
        profiles = load('utils/full_profiles_for_sim.mat');
        hist_arranque = profiles.hist_arranque; 
    catch
        error('Error cargando utils/full_profiles_for_sim.mat');
    end
    
    try
        load('models/modelos_prediccion_AR.mat');
    catch
        error('Error cargando modelos AR.');
    end

    % --- 2. Parámetros de Tiempo ---
    Ts_sim = mg(1).Ts_sim;          
    Ts_mpc = mg(1).Ts_mpc;          
    paso_mpc_en_sim = Ts_mpc / Ts_sim; 
    N_horizon = mg(1).N;            
    num_mg = 3;

    if mod(K_GLOBAL_TARGET - 1, paso_mpc_en_sim) ~= 0
        error('K_GLOBAL_TARGET (%d) debe coincidir con un paso de decisión del MPC.', K_GLOBAL_TARGET);
    end

    % Índices
    k_global_idx = K_GLOBAL_TARGET;
    k_mpc_idx = (k_global_idx - 1) / paso_mpc_en_sim + 1;

    % --- 3. Reconstruir ESTADO (Features para LIME) ---
    X_original = [];
    feature_names = {};
    
    % Recuperar pronósticos
    datos_sim_sub.P_dem = submuestreo_max(profiles.P_dem_sim(1:k_global_idx, :), paso_mpc_en_sim);
    datos_sim_sub.P_gen = submuestreo_max(profiles.P_gen_sim(1:k_global_idx, :), paso_mpc_en_sim);
    datos_sim_sub.Q_dem = submuestreo_max(profiles.Q_dem_sim(1:k_global_idx, :), paso_mpc_en_sim);

    hist_completo.P_dem = [hist_arranque.P_dem; datos_sim_sub.P_dem];
    hist_completo.P_gen = [hist_arranque.P_gen; datos_sim_sub.P_gen];
    hist_completo.Q_dem = [hist_arranque.Q_dem; datos_sim_sub.Q_dem];

    hist_data.P_dem = hist_completo.P_dem(end - mg(1).max_lags_mpc + 1:end, :);
    hist_data.P_gen = hist_completo.P_gen(end - mg(1).max_lags_mpc + 1:end, :);
    hist_data.Q_dem = hist_completo.Q_dem(end - mg(1).max_lags_mpc + 1:end, :);

    [P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real] = generar_predicciones_AR(hist_data, N_horizon);

    % --- Construcción del Vector Plano por Agente ---
    for i = 1:num_mg
        % Estados físicos actuales
        soc_val = results.SoC(k_global_idx, i);
        vtank_val = results.V_tank(k_global_idx, i);
        
        % Features de pronóstico (k=1)
        p_dem_k1 = P_dem_pred_real(1, i);
        p_gen_k1 = P_gen_pred_real(1, i);
        q_dem_k1 = Q_dem_pred_real(1, i);
        
        % Agregar al vector X
        X_original = [X_original, soc_val, vtank_val, p_dem_k1, p_gen_k1, q_dem_k1];
        
        % Nombres para gráficos
        prefix = sprintf('MG%d_', i);
        feature_names{end+1} = [prefix 'SoC'];
        feature_names{end+1} = [prefix 'V_tank'];
        feature_names{end+1} = [prefix 'P_dem'];
        feature_names{end+1} = [prefix 'P_gen'];
        feature_names{end+1} = [prefix 'Q_dem'];
    end

    % --- Recurso Compartido (Acuífero) ---
    v_aq_val = results.V_aq(k_global_idx);
    X_original = [X_original, v_aq_val];
    feature_names{end+1} = 'V_aq (Shared)';

    % --- 4. Reconstruir CONSTANTES (Historiales para MPC) ---
    q_p_hist_0_real = zeros(1, num_mg);
    p_mgref_hist_0_real = zeros(1, num_mg);
    
    if k_mpc_idx > 1
         k_prev_action_idx = (k_mpc_idx - 2) * paso_mpc_en_sim + 1;
         q_p_hist_0_real = results.Q_p(k_prev_action_idx, :);
         p_mgref_hist_0_real = results.P_grid(k_prev_action_idx, :); 
    end

    H_hist_theis = 48;
    Q_p_hist_mpc_real = zeros(H_hist_theis, num_mg);
    
    if k_mpc_idx > 1
        q_p_sim_full = results.Q_p;
        q_p_mpc_extracted = q_p_sim_full(1:paso_mpc_en_sim:end, :); 
        idx_end = k_mpc_idx - 1;
        idx_start = max(1, idx_end - H_hist_theis + 1);
        data_chunk = q_p_mpc_extracted(idx_start:idx_end, :);
        len_chunk = size(data_chunk, 1);
        Q_p_hist_mpc_real(end-len_chunk+1:end, :) = data_chunk;
    end

    % --- 5. Empaquetar Salida ---
    estado.X_original = X_original;
    estado.feature_names = feature_names;
    
    estado.constants.p_dem_pred_full = P_dem_pred_real;
    estado.constants.p_gen_pred_full = P_gen_pred_real;
    estado.constants.q_dem_pred_full = Q_dem_pred_real;
    estado.constants.q_p_hist_0 = q_p_hist_0_real;
    estado.constants.p_mgref_hist_0 = p_mgref_hist_0_real;
    estado.constants.k_mpc_actual = k_mpc_idx;
    estado.constants.Q_p_hist_mpc = Q_p_hist_mpc_real;
    
    % ** ACTUALIZACIÓN: Guardar vector real de intercambio para las 3 MGs **
    estado.Y_target_real_vector = results.Q_t(k_global_idx, :); 

    params.mg = mg;
end