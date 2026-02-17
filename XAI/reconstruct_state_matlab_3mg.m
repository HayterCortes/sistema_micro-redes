%% --- Archivo: reconstruct_state_matlab_3mg.m ---
%
% Reconstruye el estado COMPLETO del sistema de 3 Micro-redes.
% ** VERSIÓN ROBUSTA PARA SUB-CARPETAS PROFUNDAS (XAI/pumping) **
%
% Entradas:
%   K_GLOBAL_TARGET: Instante de tiempo a reconstruir (índice de simulación).
%   TIPO_MODELO: 'AR' o 'TS' (String).
%--------------------------------------------------------------------------
function [estado, params] = reconstruct_state_matlab_3mg(K_GLOBAL_TARGET, TIPO_MODELO)

    if nargin < 2
        TIPO_MODELO = 'AR'; % Valor por defecto si no se especifica
        warning('No se especificó TIPO_MODELO. Usando AR por defecto.');
    end

    % --- 1. CARGA ROBUSTA DE RESULTADOS ---
    fname_specific = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    fname_generic  = 'resultados_mpc_3mg_7dias.mat';
    
    % Lista de rutas posibles (Orden: Profundo -> Cercano -> Local)
    possible_paths = {
        fullfile('..', '..', 'results_mpc'), ... % Subir 2 niveles (desde XAI/pumping)
        fullfile('..', 'results_mpc'),       ... % Subir 1 nivel (desde XAI)
        'results_mpc', ...                       % Carpeta hija
        '.'                                      % Carpeta actual
    };
    
    file_results = '';
    
    % 1.A Buscar archivo específico
    for i = 1:length(possible_paths)
        p = fullfile(possible_paths{i}, fname_specific);
        if isfile(p), file_results = p; break; end
    end
    
    % 1.B Si falla, buscar genérico
    if isempty(file_results)
        for i = 1:length(possible_paths)
            p = fullfile(possible_paths{i}, fname_generic);
            if isfile(p), file_results = p; break; end
        end
    end

    if isempty(file_results)
        error('reconstruct_state:FileNotFound', ...
            'No se encontraron archivos de resultados en ninguna ruta (../../, ../, etc).');
    end

    try
        results = load(file_results);
        mg = results.mg;
    catch ME
        error('Error cargando archivo de resultados (%s): %s', file_results, ME.message);
    end
    
    % --- 2. CARGA DE PERFILES (UTILS) ---
    path_utils = '';
    % Misma lógica de profundidad para utils
    possible_utils = {
        fullfile('..', '..', 'utils'), ...
        fullfile('..', 'utils'), ...
        'utils', '.'
    };
    
    for i = 1:length(possible_utils)
        p = fullfile(possible_utils{i}, 'full_profiles_for_sim.mat');
        if isfile(p), path_utils = p; break; end
    end
    
    if isempty(path_utils)
        error('No se encontró full_profiles_for_sim.mat (revise carpeta utils).');
    end

    try
        profiles = load(path_utils);
        hist_arranque = profiles.hist_arranque; 
    catch
        error('Error cargando perfiles de simulación.');
    end
    
    % --- 3. CARGA DE MODELOS PREDICTIVOS (AR/TS) ---
    path_models_dir = '';
    possible_models = {
        fullfile('..', '..', 'models'), ...
        fullfile('..', 'models'), ...
        'models', '.'
    };
    
    for i = 1:length(possible_models)
        if exist(possible_models{i}, 'dir'), path_models_dir = possible_models{i}; break; end
    end

    if isempty(path_models_dir)
        error('No se encontró la carpeta "models".');
    end

    if strcmp(TIPO_MODELO, 'TS')
        f_model = fullfile(path_models_dir, 'modelos_prediccion_TS.mat');
        if ~isfile(f_model)
            error('Falta el archivo de modelos TS: %s', f_model);
        end
        % CORRECCIÓN: Eliminamos la carga explícita de variables que no existen
        % load(f_model, 'nets_P_dem', 'nets_P_gen', 'nets_Q_dem'); <--- ELIMINADO
    else
        f_model = fullfile(path_models_dir, 'modelos_prediccion_AR.mat');
        if ~isfile(f_model)
            error('Falta el archivo de modelos AR: %s', f_model);
        end
        load(f_model); 
    end

    % --- 4. Parámetros de Tiempo ---
    Ts_sim = mg(1).Ts_sim;          
    Ts_mpc = mg(1).Ts_mpc;          
    paso_mpc_en_sim = Ts_mpc / Ts_sim; 
    N_horizon = mg(1).N;            
    num_mg = 3;

    % Validación de Índice K
    if abs(mod(K_GLOBAL_TARGET - 1, paso_mpc_en_sim)) > 1e-5
        % warning('K_GLOBAL_TARGET (%d) no coincide exactamente con un paso MPC. Redondeando.', K_GLOBAL_TARGET);
        K_GLOBAL_TARGET = round((K_GLOBAL_TARGET-1)/paso_mpc_en_sim)*paso_mpc_en_sim + 1;
    end

    % Índices
    k_global_idx = K_GLOBAL_TARGET;
    k_mpc_idx = round((k_global_idx - 1) / paso_mpc_en_sim) + 1;

    % --- 5. Reconstruir ESTADO (Features para LIME) ---
    X_original = [];
    feature_names = {};
    
    % Recuperar pronósticos históricos
    datos_sim_sub.P_dem = submuestreo_max(profiles.P_dem_sim(1:k_global_idx, :), paso_mpc_en_sim);
    datos_sim_sub.P_gen = submuestreo_max(profiles.P_gen_sim(1:k_global_idx, :), paso_mpc_en_sim);
    datos_sim_sub.Q_dem = submuestreo_max(profiles.Q_dem_sim(1:k_global_idx, :), paso_mpc_en_sim);

    hist_completo.P_dem = [hist_arranque.P_dem; datos_sim_sub.P_dem];
    hist_completo.P_gen = [hist_arranque.P_gen; datos_sim_sub.P_gen];
    hist_completo.Q_dem = [hist_arranque.Q_dem; datos_sim_sub.Q_dem];

    hist_data.P_dem = hist_completo.P_dem(end - mg(1).max_lags_mpc + 1:end, :);
    hist_data.P_gen = hist_completo.P_gen(end - mg(1).max_lags_mpc + 1:end, :);
    hist_data.Q_dem = hist_completo.Q_dem(end - mg(1).max_lags_mpc + 1:end, :);

    % --- GENERACIÓN DE PREDICCIONES (SWAP) ---
    % Asumimos que las funciones generar_predicciones están en el path
    if strcmp(TIPO_MODELO, 'TS')
        [P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real] = generar_predicciones_TS(hist_data, N_horizon);
    else
        [P_dem_pred_real, P_gen_pred_real, Q_dem_pred_real] = generar_predicciones_AR(hist_data, N_horizon);
    end

    % --- Construcción del Vector Plano por Agente ---
    for i = 1:num_mg
        % Estados físicos actuales
        soc_val = results.SoC(k_global_idx, i);
        vtank_val = results.V_tank(k_global_idx, i);
        
        % Features de pronóstico (k=1)
        p_dem_k1 = P_dem_pred_real(1, i);
        p_gen_k1 = P_gen_pred_real(1, i);
        q_dem_k1 = Q_dem_pred_real(1, i);
        
        X_original = [X_original, soc_val, vtank_val, p_dem_k1, p_gen_k1, q_dem_k1];
        
        prefix = sprintf('MG%d_', i);
        feature_names{end+1} = [prefix 'SoC'];
        feature_names{end+1} = [prefix 'V_tank'];
        feature_names{end+1} = [prefix 'P_dem'];
        feature_names{end+1} = [prefix 'P_gen'];
        feature_names{end+1} = [prefix 'Q_dem'];
    end

    % --- Recurso Compartido (Acuífero) ---
    if isfield(results, 'V_aq')
        v_aq_val = results.V_aq(k_global_idx);
    elseif isfield(results, 'EAW')
        v_aq_val = results.EAW(k_mpc_idx);
    else
        v_aq_val = 0; warning('V_aq no encontrado en resultados.');
    end
    
    X_original = [X_original, v_aq_val];
    feature_names{end+1} = 'V_aq_EAW';

    % --- 6. Reconstruir CONSTANTES (Historiales para MPC) ---
    q_p_hist_0_real = zeros(1, num_mg);
    p_mgref_hist_0_real = zeros(1, num_mg);
    
    if k_mpc_idx > 1
         k_prev_sim_idx = (k_mpc_idx - 2) * paso_mpc_en_sim + 1;
         if k_prev_sim_idx < 1, k_prev_sim_idx = 1; end
         q_p_hist_0_real = results.Q_p(k_prev_sim_idx, :);
         p_mgref_hist_0_real = results.P_grid(k_prev_sim_idx, :); 
    end

    H_hist_theis = 48;
    Q_p_hist_mpc_real = zeros(H_hist_theis, num_mg);
    
    if k_mpc_idx > 1
        q_p_sim_full = results.Q_p;
        q_p_mpc_extracted = q_p_sim_full(1:paso_mpc_en_sim:end, :); 
        idx_end_mpc = k_mpc_idx - 1;
        idx_start_mpc = max(1, idx_end_mpc - H_hist_theis + 1);
        data_chunk = q_p_mpc_extracted(idx_start_mpc:idx_end_mpc, :);
        len_chunk = size(data_chunk, 1);
        Q_p_hist_mpc_real(end-len_chunk+1:end, :) = data_chunk;
    end

    % --- 7. Empaquetar Salida ---
    estado.X_original = X_original;
    estado.feature_names = feature_names;
    
    estado.constants.p_dem_pred_full = P_dem_pred_real;
    estado.constants.p_gen_pred_full = P_gen_pred_real;
    estado.constants.q_dem_pred_full = Q_dem_pred_real;
    estado.constants.q_p_hist_0 = q_p_hist_0_real;
    estado.constants.p_mgref_hist_0 = p_mgref_hist_0_real;
    estado.constants.k_mpc_actual = k_mpc_idx;
    estado.constants.Q_p_hist_mpc = Q_p_hist_mpc_real;
    
    if isfield(results, 'Q_t')
        estado.Y_target_real_vector = results.Q_t(k_global_idx, :); 
    else
        estado.Y_target_real_vector = [0 0 0];
    end

    params.mg = mg;
end