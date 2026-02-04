%% --- Archivo: lime_temporal_pumping_main_MEAN.m ---
%
% ANÁLISIS TEMPORAL LIME PARA BOMBEO (Q_p) - VERSIÓN MEAN + AR/TS + R2 QUALITY
% Ejecuta LIME a intervalos regulares usando características PROMEDIO.
%
% CORRECCIÓN: Llamada directa a función (sin evalc) para recuperar stats correctamente.
%--------------------------------------------------------------------------
close all; clear; clc;

% --- 1. CONFIGURACIÓN ---
TIPO_MODELO = 'TS';      % <--- CAMBIA ESTO A 'AR' O 'TS'
TARGETS = [1, 2, 3];     % Las 3 MGs
INTERVALO_HORAS = 12;    % Resolución temporal
NUM_RUNS_PER_POINT = 1;  % Estabilidad (1 para rápido, 10+ para paper)

fprintf('--- LIME TEMPORAL BOMBEO (MEAN + R2) - Modelo: %s ---\n', TIPO_MODELO);

% 2. Cargar Datos (Ruta Ultra-Robusta)
try
    fname_specific = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    fname_generic  = 'resultados_mpc_3mg_7dias.mat';
    
    % LISTA DE RUTAS POSIBLES (Prioridad descendente)
    possible_paths = {
        fullfile('..', '..', 'results_mpc'), ... % 1. Subir 2 niveles (desde XAI/pumping)
        fullfile('..', 'results_mpc'),       ... % 2. Subir 1 nivel (desde XAI)
        'results_mpc', ...                       % 3. Subcarpeta local
        '.'                                      % 4. Carpeta actual
    };
    
    fname = '';
    
    % Búsqueda del archivo específico
    for i = 1:length(possible_paths)
        p = fullfile(possible_paths{i}, fname_specific);
        if isfile(p), fname = p; break; end
    end
    
    % Fallback al genérico
    if isempty(fname)
        for i = 1:length(possible_paths)
            p = fullfile(possible_paths{i}, fname_generic);
            if isfile(p)
                fname = p;
                fprintf('(!) Aviso: Usando archivo genérico: %s\n', fname);
                break;
            end
        end
    end
    
    if isempty(fname)
        error('No se encuentra el archivo de resultados en ninguna ruta (../../, ../, ./).');
    end
    
    results = load(fname);
    mg = results.mg;
    fprintf('Datos cargados correctamente desde:\n -> %s\n', fname);
    
catch ME
    error('Error crítico cargando datos: %s', ME.message);
end

% 3. Definir Vector de Tiempos
Ts_sim = mg(1).Ts_sim;              
Ts_mpc = mg(1).Ts_mpc;              
paso_mpc = Ts_mpc / Ts_sim;  
Total_Steps = length(results.SoC);
steps_interval = (INTERVALO_HORAS * 3600) / Ts_sim;
k_list_raw = 1 : steps_interval : Total_Steps;

k_list = [];
for k = k_list_raw
    k_mpc_idx = round((k - 1) / paso_mpc);
    k_adjusted = k_mpc_idx * paso_mpc + 1;
    if k_adjusted < Total_Steps
        k_list = [k_list, k_adjusted];
    end
end
fprintf('Puntos a analizar: %d por Agente.\n', length(k_list));

% --- 4. BUCLE POR AGENTE ---
for t_idx = TARGETS
    fprintf('\n==================================================\n');
    fprintf('   EVOLUCIÓN TEMPORAL BOMBEO: AGENTE MG %d\n', t_idx);
    fprintf('==================================================\n');
    
    temporal_results = struct();
    temporal_results.k_list = k_list;
    temporal_results.time_days = (k_list - 1) * Ts_sim / 86400;
    temporal_results.weights_history = [];
    temporal_results.target_real_history = [];
    temporal_results.quality_history = []; % Nuevo campo R2
    temporal_results.feature_names = {}; 
    
    % --- BUCLE TIEMPO ---
    for idx = 1:length(k_list)
        k_target = k_list(idx);
        day_curr = temporal_results.time_days(idx);
        
        fprintf('MG%d -> K=%d (Día %.2f)... ', t_idx, k_target, day_curr);
        
        % A. Reconstruir Estado
        [estado, params] = reconstruct_state_matlab_3mg(k_target, TIPO_MODELO);
        
        % B. Feature Engineering (MEAN)
        P_dem = estado.constants.p_dem_pred_full; 
        P_gen = estado.constants.p_gen_pred_full; 
        Q_dem = estado.constants.q_dem_pred_full; 
        
        m_P_dem = mean(P_dem, 1); m_P_gen = mean(P_gen, 1); m_Q_dem = mean(Q_dem, 1);
        
        base_idx = [3, 8, 13];
        for m = 1:3
            bi = base_idx(m);
            estado.X_original(bi)   = m_P_dem(m); 
            estado.X_original(bi+1) = m_P_gen(m); 
            estado.X_original(bi+2) = m_Q_dem(m);
            
            if idx == 1
                for p=0:2
                    curr = estado.feature_names{bi+p};
                    if ~startsWith(curr, 'Mean_')
                        estado.feature_names{bi+p} = ['Mean_' curr];
                    end
                end
            end
        end
        
        if idx == 1, temporal_results.feature_names = estado.feature_names; end
        
        % C. Compilar (Solo 1 vez o dinámico si es necesario)
        % Nota: Para simplificar, recompilamos en cada paso para asegurar estado fresco
        controller_obj = get_compiled_mpc_controller_3mg(params.mg);
        
        % D. Valor Real Q_p
        val_real = results.Q_p(k_target, t_idx);
        temporal_results.target_real_history(idx) = val_real;
        
        % E. Ejecutar LIME (LLAMADA DIRECTA SIN EVALC)
        % <--- CORRECCIÓN CRÍTICA AQUÍ ---
        [lime_stats, all_explanations] = calculate_lime_pumping_3mg_with_quality(estado, controller_obj, params, NUM_RUNS_PER_POINT, t_idx);
        
        % Almacenar R2
        temporal_results.quality_history(idx) = lime_stats.R2_mean;
        
        % F. Promediar Pesos
        w_mat = zeros(length(temporal_results.feature_names), NUM_RUNS_PER_POINT);
        for r = 1:NUM_RUNS_PER_POINT
            d = all_explanations{r};
            map_w = containers.Map(d(:,1), [d{:,2}]);
            for f = 1:length(temporal_results.feature_names)
                name = temporal_results.feature_names{f};
                if isKey(map_w, name), w_mat(f,r) = map_w(name); end
            end
        end
        temporal_results.weights_history(:, idx) = mean(w_mat, 2);
        
        fprintf('[OK] Q_p=%.3f, R2=%.4f\n', val_real, lime_stats.R2_mean);
    end
    
    % Guardado
    filename_out = sprintf('lime_temporal_PUMP_%s_MG%d_7days_MEAN.mat', TIPO_MODELO, t_idx);
    save(filename_out, 'temporal_results');
    fprintf('--> Datos guardados en %s\n', filename_out);
end
fprintf('\n--- ANÁLISIS TEMPORAL DE BOMBEO FINALIZADO ---\n');

%% --- Helper: Compilador (Requerido) ---
function Controller = get_compiled_mpc_controller_3mg(mg_array)
    yalmip('clear');
    N = mg_array(1).N; Ts = mg_array(1).Ts_mpc; num_mg = 3;
    
    soC_0_var    = sdpvar(1, 3, 'full');
    v_tank_0_var = sdpvar(1, 3, 'full');
    v_aq_0_var   = sdpvar(1, 1, 'full');
    p_dem_pred_var = sdpvar(N, 3, 'full');
    p_gen_pred_var = sdpvar(N, 3, 'full');
    q_dem_pred_var = sdpvar(N, 3, 'full');
    q_p_hist_0_var = sdpvar(1, 3, 'full');
    p_mgref_hist_0_var = sdpvar(1, 3, 'full');
    k_mpc_var = sdpvar(1, 1, 'full');
    Q_p_hist_mpc_var = sdpvar(48, 3, 'full');
    
    P_mgref = sdpvar(N, num_mg, 'full'); Q_p = sdpvar(N, num_mg, 'full');
    Q_buy = sdpvar(N, num_mg, 'full'); Q_t = sdpvar(N, num_mg, 'full');
    P_B = sdpvar(N, num_mg, 'full'); P_p = sdpvar(N, num_mg, 'full');
    SoC = sdpvar(N+1, num_mg, 'full'); V_tank = sdpvar(N+1, num_mg, 'full');
    EAW = sdpvar(N+1, 1, 'full'); s_pozo = sdpvar(N, num_mg, 'full');
    
    slack_EAW = sdpvar(1, 1, 'full'); 
    P_shed = sdpvar(N, num_mg, 'full'); Q_shed = sdpvar(N, num_mg, 'full');
    slack_s_pozo = sdpvar(N, num_mg, 'full');
    
    C_p = 110; C_q = 644; lambda_P = 1e-1; lambda_Q = 1e-1; costo_shed = 1e9;
    
    constraints = []; objective = 0;
    
    constraints = [constraints, SoC(1,:) == soC_0_var, V_tank(1,:) == v_tank_0_var, EAW(1) == v_aq_0_var];
    constraints = [constraints, slack_EAW >= 0, P_shed >= 0, Q_shed >= 0, slack_s_pozo >= 0];
    
    delta_Q_p_futuro = sdpvar(N, num_mg, 'full');
    constraints = [constraints, delta_Q_p_futuro(1,:) == Q_p(1,:) - q_p_hist_0_var];
    constraints = [constraints, delta_Q_p_futuro(2:end,:) == Q_p(2:end,:) - Q_p(1:end-1,:)];
    for k = 1:N
        for i = 1:num_mg
            descenso = 0;
            S = mg_array(1).S_aq; T_val = mg_array(1).T_aq; r = mg_array(1).r_p; ts = mg_array(1).Ts_mpc;
            for l = 1:k
                u = (S * r^2) / (4 * T_val * (k - l + 1) * ts);
                descenso = descenso + (delta_Q_p_futuro(l,i)/1000) * expint(u);
            end
            constraints = [constraints, s_pozo(k,i) == (1/(4*pi*T_val)) * descenso]; 
            constraints = [constraints, s_pozo(k,i) <= mg_array(1).s_max + slack_s_pozo(k,i)];
        end
        
        for i = 1:num_mg
             constraints = [constraints, P_mgref(k,i) + P_shed(k,i) == p_dem_pred_var(k,i) - p_gen_pred_var(k,i) + P_p(k,i) - P_B(k,i)];
             dem_agua = q_dem_pred_var(k,i) - Q_shed(k,i);
             constraints = [constraints, V_tank(k+1,i) == V_tank(k,i) + (Q_p(k,i) + Q_buy(k,i) - Q_t(k,i) - dem_agua)*Ts];
             constraints = [constraints, P_p(k,i) == mg_array(i).Mp * 9800 * Q_p(k,i) * (mg_array(i).h_ptub + mg_array(i).h_Tank_max)/1e6];
             constraints = [constraints, mg_array(i).E_batt_max * SoC(k+1,i) == mg_array(i).E_batt_max * SoC(k,i) - P_B(k,i)*(Ts/3600)];
             constraints = [constraints, mg_array(i).SoC_min <= SoC(k+1,i) <= mg_array(i).SoC_max];
             p_chg = mg_array(i).alpha_C * mg_array(i).P_batt_max * (1 - SoC(k,i));
             p_dis = mg_array(i).alpha_D * mg_array(i).P_batt_max * SoC(k,i);
             constraints = [constraints, -p_chg <= P_B(k,i) <= p_dis];
             constraints = [constraints, 0 <= V_tank(k+1,i) <= mg_array(i).V_max];
             constraints = [constraints, 0 <= Q_p(k,i) <= mg_array(i).Mp * mg_array(i).Q_pump_max_unit];
             constraints = [constraints, 0 <= Q_buy(k,i)];
             constraints = [constraints, mg_array(i).Q_t_min <= Q_t(k,i) <= mg_array(i).Q_t_max];
        end
        
        constraints = [constraints, sum(Q_t(k,:)) == 0];
        constraints = [constraints, 0 <= sum(P_mgref(k,:)) <= mg_array(1).P_grid_max];
        constraints = [constraints, 0 <= sum(Q_buy(k,:)) <= mg_array(1).Q_DNO_max];
        
        p_dno = sum(P_mgref(k,:)); q_dno = sum(Q_buy(k,:));
        c_e = C_p * p_dno * (Ts/3600); c_a = C_q * q_dno * Ts/1000;
        
        if k==1
            dP = P_mgref(k,:) - p_mgref_hist_0_var; dQ = Q_p(k,:) - q_p_hist_0_var;
        else
            dP = P_mgref(k,:) - P_mgref(k-1,:); dQ = Q_p(k,:) - Q_p(k-1,:);
        end
        c_smooth = lambda_P*sum(dP.^2) + lambda_Q*sum(dQ.^2);
        c_pen = costo_shed * (sum(P_shed(k,:)) + sum(Q_shed(k,:)));
        objective = objective + c_e + c_a + c_smooth + c_pen;
    end
    
    recarga = (mg_array(1).Rp * Ts/60) * ones(N,1);
    bombeo = sum(Q_p, 2) * Ts;
    constraints = [constraints, EAW(2:end) == EAW(1:end-1) + recarga - bombeo];
    constraints = [constraints, EAW(end) >= mg_array(1).V_aq_0 - slack_EAW];
    
    objective = objective + costo_shed*slack_EAW + costo_shed*sum(sum(slack_s_pozo));
    ops = sdpsettings('solver','gurobi','verbose',0);
    Inputs = {soC_0_var, v_tank_0_var, v_aq_0_var, p_dem_pred_var, p_gen_pred_var, q_dem_pred_var, q_p_hist_0_var, p_mgref_hist_0_var, k_mpc_var, Q_p_hist_mpc_var};
    Outputs = {P_mgref, Q_p, Q_buy, Q_t, s_pozo};
    Controller = optimizer(constraints, objective, ops, Inputs, Outputs);
end