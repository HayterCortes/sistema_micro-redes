%% --- Archivo: RUN_LIME_TEMPORAL_MASTER.m ---
%
% EL SCRIPT DEFINITIVO: EVOLUCIÓN TEMPORAL LIME MASIVA
%
% Ejecuta y guarda la evolución temporal a lo largo de los 7 días para:
% - Modelos: AR, TS
% - Casos: 1 (Mean), 2 (Standard Moments), 3 (AE Moments)
% - Perturbaciones: GAUSSIAN, PARETO
% - Targets: Intercambio (Q_t) y Bombeo (Q_p) en todos los instantes evaluados.
%
% Genera matrices tridimensionales (RAW Weights) y series de tiempo de R2/RBO.
%--------------------------------------------------------------------------
close all; clear; clc;

% --- MAGIA DE RUTAS (Path) ---
project_root = fullfile(pwd, '..'); 
addpath(genpath(project_root));
fprintf('--- RUTAS AÑADIDAS: MATLAB ahora ve todas las subcarpetas del proyecto ---\n');

% --- CONFIGURACIÓN GLOBAL TEMPORAL ---
NUM_RUNS_PER_POINT = 2;  % Corridas por instante de tiempo (estadística RBO)
RBO_P = 0.9;             % Persistencia RBO
INTERVALO_HORAS = 12;    % Cada cuántas horas se evalúa LIME a lo largo de la semana

MODELS = {'AR', 'TS'};
PERTURBATIONS = {'GAUSSIAN', 'PARETO'};
CASES = [1, 2, 3]; 
TARGETS = [1, 2, 3];

% Nombres de Casos y Carpetas
base_out = 'temporal_results';
dirs_ex = {fullfile(base_out, '16_features_exchange'), fullfile(base_out, '34_features_exchange'), fullfile(base_out, '34_features_AE_exchange')};
dirs_pp = {fullfile(base_out, 'pumping', '16_features_pump'), fullfile(base_out, 'pumping', '34_features_pump'), fullfile(base_out, 'pumping', '34_features_AE_pump')};
case_tags = {'MEAN', 'STANDARD', 'AE_MOMENTS'};

% Crear estructura de carpetas
fprintf('--- CREANDO ESTRUCTURA DE CARPETAS TEMPORALES ---\n');
if ~exist(base_out, 'dir'), mkdir(base_out); end
for d = 1:3
    if ~exist(dirs_ex{d}, 'dir'), mkdir(dirs_ex{d}); end
    if ~exist(dirs_pp{d}, 'dir'), mkdir(dirs_pp{d}); end
end

fprintf('=========================================================\n');
fprintf(' INICIANDO EJECUCIÓN MAESTRA TEMPORAL LIME\n');
fprintf('=========================================================\n\n');

%% --- BUCLE 1: MODELOS ---
for m_idx = 1:length(MODELS)
    TIPO_MODELO = MODELS{m_idx};
    fprintf('\n>>>>>>>>>> CARGANDO MODELO: %s <<<<<<<<<<\n', TIPO_MODELO);
    
    % 1. CARGA DE DATOS DEL MODELO
    fname_base = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    possible_paths = {fullfile('..', 'results_mpc'), 'results_mpc', '.'};
    fname = ''; for i=1:length(possible_paths), p=fullfile(possible_paths{i}, fname_base); if isfile(p), fname=p; break; end; end
    if isempty(fname), error('Datos no encontrados: %s', fname_base); end
    results = load(fname);
    mg = results.mg; Q_t = results.Q_t; Q_p = results.Q_p; 
    
    % 2. DEFINICIÓN DEL VECTOR DE TIEMPO (k_list)
    Ts_sim = mg(1).Ts_sim; Ts_mpc = mg(1).Ts_mpc; paso_mpc = Ts_mpc / Ts_sim;  
    Total_Steps = length(results.SoC);
    steps_interval = (INTERVALO_HORAS * 3600) / Ts_sim;
    k_list_raw = 1 : steps_interval : Total_Steps;
    k_list = [];
    for k = k_list_raw
        k_adj = round((k-1)/paso_mpc)*paso_mpc + 1; 
        if k_adj < Total_Steps, k_list = [k_list, k_adj]; end
    end
    
    % 3. COMPILAR CONTROLADOR YALMIP (Se compila 1 vez por modelo)
    fprintf('Compilando YALMIP MPC para modelo %s...\n', TIPO_MODELO);
    [~, params_init] = reconstruct_state_matlab_3mg(k_list(1), TIPO_MODELO);
    controller_obj = get_compiled_mpc_controller_3mg(params_init.mg);
    
    % 4. PRE-ASIGNACIÓN DE ESTRUCTURAS DE ALMACENAMIENTO (Optimizador de Memoria)
    ex_res = cell(2, 3, 3); % {Perturbation, Case, Target}
    pp_res = cell(2, 3, 3);
    for p = 1:2
        for c = 1:3
            N_feats = 16; if c > 1, N_feats = 34; end % Tamaño dinámico de matriz
            for t = 1:3
                ex_res{p,c,t}.k_list = k_list; ex_res{p,c,t}.time_days = (k_list - 1) * Ts_sim / 86400;
                ex_res{p,c,t}.perturbation = PERTURBATIONS{p}; ex_res{p,c,t}.feature_names = {};
                ex_res{p,c,t}.weights_history = nan(N_feats, length(k_list));
                ex_res{p,c,t}.weights_raw_history = nan(N_feats, NUM_RUNS_PER_POINT, length(k_list));
                ex_res{p,c,t}.target_real_history = nan(1, length(k_list));
                ex_res{p,c,t}.quality_history = nan(1, length(k_list));
                ex_res{p,c,t}.rbo_history = nan(1, length(k_list));
                ex_res{p,c,t}.rbo_std_history = nan(1, length(k_list));
                pp_res{p,c,t} = ex_res{p,c,t}; % Misma estructura para bombeo
            end
        end
    end
    
    %% --- BUCLE 2: TIEMPO (Evolución Temporal) ---
    for idx = 1:length(k_list)
        k_target = k_list(idx);
        fprintf('\n[K=%d | Día %.2f]\n', k_target, (k_target-1)*Ts_sim/86400);
        
        % Reconstrucción Pesada (UNA VEZ por instante temporal)
        [estado_raw, params_raw] = reconstruct_state_matlab_3mg(k_target, TIPO_MODELO);
        
        try, P_dem = estado_raw.constants.p_dem_pred_full; P_gen = estado_raw.constants.p_gen_pred_full; Q_dem = estado_raw.constants.q_dem_pred_full;
        catch, P_dem = params_raw.P_dem_pred; P_gen = params_raw.P_gen_pred; Q_dem = params_raw.Q_dem_pred; end
        m_P_dem = mean(P_dem, 1); m_P_gen = mean(P_gen, 1); m_Q_dem = mean(Q_dem, 1);
        max_P_gen = max(P_gen, [], 1); max_P_dem = max(P_dem, [], 1); max_Q_dem = max(Q_dem, [], 1);
        std_P_gen = std(P_gen, 0, 1); std_P_dem = std(P_dem, 0, 1); std_Q_dem = std(Q_dem, 0, 1);
        
        %% --- BUCLES ANIDADOS: TARGET -> PERTURBATION -> CASE ---
        for t_idx = TARGETS
            val_real_qt = results.Q_t(k_target, t_idx);
            val_real_qp = results.Q_p(k_target, t_idx);
            
            for p_idx = 1:length(PERTURBATIONS)
                PERT_TYPE = PERTURBATIONS{p_idx};
                
                for c_idx = CASES
                    % 1. Construcción del Feature Engineering
                    estado = estado_raw; 
                    estado.X_original([3,8,13]) = m_P_dem; estado.X_original([4,9,14]) = m_P_gen; estado.X_original([5,10,15]) = m_Q_dem;
                    
                    if c_idx == 1 % MEAN (16 variables)
                        prefixes = {'P_dem', 'P_gen', 'Q_dem'}; idx_replace = [3, 4, 5, 8, 9, 10, 13, 14, 15]; count = 1;
                        for m = 1:3, for p_f = 1:3, old = estado.feature_names{idx_replace(count)};
                            if ~startsWith(old, 'Mean_'), estado.feature_names{idx_replace(count)} = strrep(old, prefixes{p_f}, ['Mean_' prefixes{p_f}]); end
                            count = count + 1; end; end
                    else % MOMENTS (34 variables)
                        estado.X_original = [estado.X_original, max_P_gen, max_P_dem, max_Q_dem, std_P_gen, std_P_dem, std_Q_dem];
                        for i=1:length(estado.feature_names), if ~startsWith(estado.feature_names{i},'Mean_') && i~=1 && i~=2 && i~=6 && i~=7 && i~=11 && i~=12 && i~=16
                            estado.feature_names{i}=['Mean_' estado.feature_names{i}]; end; end
                        new_names = {};
                        for i=1:3, new_names{end+1}=sprintf('Max_P_gen_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Max_P_dem_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Max_Q_dem_MG%d',i); end
                        for i=1:3, new_names{end+1}=sprintf('Std_P_gen_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Std_P_dem_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Std_Q_dem_MG%d',i); end
                        estado.feature_names = [estado.feature_names, new_names];
                    end
                    
                    % Guardar nombres solo en la primera iteración temporal
                    if idx == 1
                        ex_res{p_idx,c_idx,t_idx}.feature_names = estado.feature_names;
                        pp_res{p_idx,c_idx,t_idx}.feature_names = estado.feature_names;
                    end
                    
                    fprintf('  > MG%d | %10s | %10s -> ', t_idx, case_tags{c_idx}, PERT_TYPE);
                    
                    % --- EJECUTAR INTERCAMBIO (Q_t) ---
                    ex_res{p_idx,c_idx,t_idx}.target_real_history(idx) = val_real_qt;
                    try
                        if c_idx == 1 % Mean
                            [lime_ex, expl_ex] = calculate_lime_stability_3mg_with_quality(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx, PERT_TYPE);
                        elseif c_idx == 2 % Standard Moments
                            if strcmp(PERT_TYPE,'PARETO'), [lime_ex, expl_ex] = calculate_lime_exchange_3mg_STANDARD_MOMENTS_PARETO(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx);
                            else, [lime_ex, expl_ex] = calculate_lime_exchange_3mg_STANDARD_MOMENTS(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx); end
                        elseif c_idx == 3 % AE Moments
                            if strcmp(PERT_TYPE,'PARETO'), [lime_ex, expl_ex] = calculate_lime_exchange_3mg_with_AE_MOMENTS_PARETO(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx);
                            else, [lime_ex, expl_ex] = calculate_lime_exchange_3mg_with_AE_MOMENTS(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx); end
                        end
                        rbo_ex = compute_rbo_stats_temporal(expl_ex, NUM_RUNS_PER_POINT, RBO_P, estado.feature_names);
                        
                        ex_res{p_idx,c_idx,t_idx}.quality_history(idx) = lime_ex.R2_mean;
                        ex_res{p_idx,c_idx,t_idx}.rbo_history(idx) = rbo_ex.mean;
                        ex_res{p_idx,c_idx,t_idx}.rbo_std_history(idx) = rbo_ex.std;
                        ex_res{p_idx,c_idx,t_idx}.weights_history(:, idx) = mean(rbo_ex.w_mat, 2);
                        ex_res{p_idx,c_idx,t_idx}.weights_raw_history(:, :, idx) = rbo_ex.w_mat;
                        fprintf('Q_t[R2:%.2f] ', lime_ex.R2_mean);
                    catch ME
                        fprintf('Q_t[ERR] ');
                    end
                    
                    % --- EJECUTAR BOMBEO (Q_p) ---
                    pp_res{p_idx,c_idx,t_idx}.target_real_history(idx) = val_real_qp;
                    try
                        if c_idx == 1 % Mean
                            [lime_pp, expl_pp] = calculate_lime_pumping_3mg_with_quality(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx, PERT_TYPE);
                        elseif c_idx == 2 % Standard Moments
                            if strcmp(PERT_TYPE,'PARETO'), [lime_pp, expl_pp] = calculate_lime_pumping_3mg_STANDARD_MOMENTS_PARETO(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx);
                            else, [lime_pp, expl_pp] = calculate_lime_pumping_3mg_STANDARD_MOMENTS(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx); end
                        elseif c_idx == 3 % AE Moments
                            if strcmp(PERT_TYPE,'PARETO'), [lime_pp, expl_pp] = calculate_lime_pumping_3mg_with_AE_MOMENTS_PARETO(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx);
                            else, [lime_pp, expl_pp] = calculate_lime_pumping_3mg_with_AE_MOMENTS(estado, controller_obj, params_raw, NUM_RUNS_PER_POINT, t_idx); end
                        end
                        rbo_pp = compute_rbo_stats_temporal(expl_pp, NUM_RUNS_PER_POINT, RBO_P, estado.feature_names);
                        
                        pp_res{p_idx,c_idx,t_idx}.quality_history(idx) = lime_pp.R2_mean;
                        pp_res{p_idx,c_idx,t_idx}.rbo_history(idx) = rbo_pp.mean;
                        pp_res{p_idx,c_idx,t_idx}.rbo_std_history(idx) = rbo_pp.std;
                        pp_res{p_idx,c_idx,t_idx}.weights_history(:, idx) = mean(rbo_pp.w_mat, 2);
                        pp_res{p_idx,c_idx,t_idx}.weights_raw_history(:, :, idx) = rbo_pp.w_mat;
                        fprintf('Q_p[R2:%.2f]\n', lime_pp.R2_mean);
                    catch ME
                        fprintf('Q_p[ERR]\n');
                    end
                    
                end % Fin Cases
            end % Fin Perturbations
        end % Fin Targets
    end % Fin Time Loop
    
    %% --- BLOQUE DE GUARDADO (Al finalizar todo el tiempo para este modelo) ---
    fprintf('\nGuardando archivos de Resultados Temporales para %s...\n', TIPO_MODELO);
    for p_idx = 1:length(PERTURBATIONS)
        for c_idx = CASES
            for t_idx = TARGETS
                % Guardar Exchange
                temporal_results = ex_res{p_idx, c_idx, t_idx};
                fname_ex = fullfile(dirs_ex{c_idx}, sprintf('lime_temporal_EXCHANGE_%s_MG%d_7days_%s_RAW_RBO_%s.mat', TIPO_MODELO, t_idx, case_tags{c_idx}, PERTURBATIONS{p_idx}));
                save(fname_ex, 'temporal_results');
                
                % Guardar Pumping
                temporal_results = pp_res{p_idx, c_idx, t_idx};
                fname_pp = fullfile(dirs_pp{c_idx}, sprintf('lime_temporal_PUMPING_%s_MG%d_7days_%s_RAW_RBO_%s.mat', TIPO_MODELO, t_idx, case_tags{c_idx}, PERTURBATIONS{p_idx}));
                save(fname_pp, 'temporal_results');
            end
        end
    end
end % Fin Modelos

fprintf('\n=== EJECUCIÓN MAESTRA TEMPORAL CONCLUIDA EXITOSAMENTE ===\n');

%% --- FUNCIONES AUXILIARES GLOBALES ---

function rbo_stats = compute_rbo_stats_temporal(all_explanations, NUM_RUNS, RBO_P, feature_names)
    % Calcula métricas de estabilidad y devuelve la matriz RAW de pesos
    w_mat = zeros(length(feature_names), NUM_RUNS);
    run_rankings = cell(1, NUM_RUNS);
    
    for r = 1:NUM_RUNS
        d = all_explanations{r}; 
        map_w = containers.Map(d(:,1), [d{:,2}]);
        for f = 1:length(feature_names)
            name = feature_names{f};
            if isKey(map_w, name), w_mat(f,r) = map_w(name); end
        end
        weights = cell2mat(d(:,2));
        [~, sort_idx] = sort(abs(weights), 'descend');
        run_rankings{r} = d(sort_idx, 1);
    end
    
    rbo_values = [];
    for i = 1:NUM_RUNS
        for j = i+1:NUM_RUNS
            score = calculate_rbo_score(run_rankings{i}, run_rankings{j}, RBO_P);
            rbo_values = [rbo_values, score];
        end
    end
    if isempty(rbo_values), rbo_mean = 1; rbo_std = 0; else, rbo_mean = mean(rbo_values); rbo_std = std(rbo_values); end
    
    rbo_stats.mean = rbo_mean;
    rbo_stats.std = rbo_std;
    rbo_stats.w_mat = w_mat;
end

function rbo = calculate_rbo_score(list1, list2, p)
    if nargin < 3, p = 0.9; end; k = min(length(list1), length(list2)); sum_series = 0;
    for d = 1:k, set1 = list1(1:d); set2 = list2(1:d); intersection_size = length(intersect(set1, set2)); A_d = intersection_size / d; sum_series = sum_series + (p^(d-1)) * A_d; end
    rbo = (1 - p) * sum_series;
end

function Controller = get_compiled_mpc_controller_3mg(mg_array)
    yalmip('clear'); N = mg_array(1).N; Ts = mg_array(1).Ts_mpc; num_mg = 3;
    soC_0_var = sdpvar(1, 3, 'full'); v_tank_0_var = sdpvar(1, 3, 'full'); v_aq_0_var = sdpvar(1, 1, 'full');
    p_dem_pred_var = sdpvar(N, 3, 'full'); p_gen_pred_var = sdpvar(N, 3, 'full'); q_dem_pred_var = sdpvar(N, 3, 'full');
    q_p_hist_0_var = sdpvar(1, 3, 'full'); p_mgref_hist_0_var = sdpvar(1, 3, 'full'); k_mpc_var = sdpvar(1, 1, 'full');
    Q_p_hist_mpc_var = sdpvar(48, 3, 'full');
    P_mgref = sdpvar(N, num_mg, 'full'); Q_p = sdpvar(N, num_mg, 'full'); Q_buy = sdpvar(N, num_mg, 'full'); Q_t = sdpvar(N, num_mg, 'full');
    P_B = sdpvar(N, num_mg, 'full'); P_p = sdpvar(N, num_mg, 'full'); SoC = sdpvar(N+1, num_mg, 'full'); V_tank = sdpvar(N+1, num_mg, 'full');
    EAW = sdpvar(N+1, 1, 'full'); s_pozo = sdpvar(N, num_mg, 'full');
    slack_EAW = sdpvar(1, 1, 'full'); P_shed = sdpvar(N, num_mg, 'full'); Q_shed = sdpvar(N, num_mg, 'full'); slack_s_pozo = sdpvar(N, num_mg, 'full');
    C_p = 110; C_q = 644; lambda_P = 1e-1; lambda_Q = 1e-1; costo_shed = 1e9;
    constraints = []; objective = 0;
    constraints = [constraints, SoC(1,:) == soC_0_var, V_tank(1,:) == v_tank_0_var, EAW(1) == v_aq_0_var];
    constraints = [constraints, slack_EAW >= 0, P_shed >= 0, Q_shed >= 0, slack_s_pozo >= 0];
    delta_Q_p_futuro = sdpvar(N, num_mg, 'full');
    constraints = [constraints, delta_Q_p_futuro(1,:) == Q_p(1,:) - q_p_hist_0_var];
    constraints = [constraints, delta_Q_p_futuro(2:end,:) == Q_p(2:end,:) - Q_p(1:end-1,:)];
    for k = 1:N
        for i = 1:num_mg
            descenso = 0; S = mg_array(1).S_aq; T_val = mg_array(1).T_aq; r = mg_array(1).r_p; ts = mg_array(1).Ts_mpc;
            for l = 1:k, u = (S * r^2) / (4 * T_val * (k - l + 1) * ts); descenso = descenso + (delta_Q_p_futuro(l,i)/1000) * expint(u); end
            constraints = [constraints, s_pozo(k,i) == (1/(4*pi*T_val)) * descenso]; constraints = [constraints, s_pozo(k,i) <= mg_array(1).s_max + slack_s_pozo(k,i)];
        end
        for i = 1:num_mg
             constraints = [constraints, P_mgref(k,i) + P_shed(k,i) == p_dem_pred_var(k,i) - p_gen_pred_var(k,i) + P_p(k,i) - P_B(k,i)];
             dem_agua = q_dem_pred_var(k,i) - Q_shed(k,i);
             constraints = [constraints, V_tank(k+1,i) == V_tank(k,i) + (Q_p(k,i) + Q_buy(k,i) - Q_t(k,i) - dem_agua)*Ts];
             constraints = [constraints, P_p(k,i) == mg_array(i).Mp * 9800 * Q_p(k,i) * (mg_array(i).h_ptub + mg_array(i).h_Tank_max)/1e6];
             constraints = [constraints, mg_array(i).E_batt_max * SoC(k+1,i) == mg_array(i).E_batt_max * SoC(k,i) - P_B(k,i)*(Ts/3600)];
             constraints = [constraints, mg_array(i).SoC_min <= SoC(k+1,i) <= mg_array(i).SoC_max];
             p_chg = mg_array(i).alpha_C * mg_array(i).P_batt_max * (1 - SoC(k,i)); p_dis = mg_array(i).alpha_D * mg_array(i).P_batt_max * SoC(k,i);
             constraints = [constraints, -p_chg <= P_B(k,i) <= p_dis];
             constraints = [constraints, 0 <= V_tank(k+1,i) <= mg_array(i).V_max];
             constraints = [constraints, 0 <= Q_p(k,i) <= mg_array(i).Mp * mg_array(i).Q_pump_max_unit];
             constraints = [constraints, 0 <= Q_buy(k,i)];
             constraints = [constraints, mg_array(i).Q_t_min <= Q_t(k,i) <= mg_array(i).Q_t_max];
        end
        constraints = [constraints, sum(Q_t(k,:)) == 0]; constraints = [constraints, 0 <= sum(P_mgref(k,:)) <= mg_array(1).P_grid_max]; constraints = [constraints, 0 <= sum(Q_buy(k,:)) <= mg_array(1).Q_DNO_max];
        p_dno = sum(P_mgref(k,:)); q_dno = sum(Q_buy(k,:)); c_e = C_p * p_dno * (Ts/3600); c_a = C_q * q_dno * Ts/1000;
        if k==1, dP = P_mgref(k,:) - p_mgref_hist_0_var; dQ = Q_p(k,:) - q_p_hist_0_var; else, dP = P_mgref(k,:) - P_mgref(k-1,:); dQ = Q_p(k,:) - Q_p(k-1,:); end
        c_smooth = lambda_P*sum(dP.^2) + lambda_Q*sum(dQ.^2); c_pen = costo_shed * (sum(P_shed(k,:)) + sum(Q_shed(k,:)));
        objective = objective + c_e + c_a + c_smooth + c_pen;
    end
    recarga = (mg_array(1).Rp * Ts/60) * ones(N,1); bombeo = sum(Q_p, 2) * Ts;
    constraints = [constraints, EAW(2:end) == EAW(1:end-1) + recarga - bombeo]; constraints = [constraints, EAW(end) >= mg_array(1).V_aq_0 - slack_EAW];
    objective = objective + costo_shed*slack_EAW + costo_shed*sum(sum(slack_s_pozo)); ops = sdpsettings('solver','gurobi','verbose',0);
    Inputs = {soC_0_var, v_tank_0_var, v_aq_0_var, p_dem_pred_var, p_gen_pred_var, q_dem_pred_var, q_p_hist_0_var, p_mgref_hist_0_var, k_mpc_var, Q_p_hist_mpc_var};
    Outputs = {P_mgref, Q_p, Q_buy, Q_t, s_pozo}; Controller = optimizer(constraints, objective, ops, Inputs, Outputs);
end