%% --- Archivo: RUN_LIME_PATCH_PUMPING_TS.m ---
%
% SCRIPT PARCHE: EJECUCIÓN LIME SOLO PARA BOMBEO (Q_p) EN MODELO TS
%
% Ejecuta exclusivamente lo faltante debido al bug de coincidencia de K:
% - Modelo: TS
% - Casos: 1, 2, 3
% - Perturbaciones: GAUSSIAN, PARETO
% - Target: SOLO Bombeo (Q_p) en escenario Altruismo
%--------------------------------------------------------------------------
close all; clear; clc;

% --- MAGIA DE RUTAS (Path) ---
project_root = fullfile(pwd, '..'); 
addpath(genpath(project_root));
fprintf('--- RUTAS AÑADIDAS: MATLAB ahora ve todas las subcarpetas del proyecto ---\n');

% --- CONFIGURACIÓN GLOBAL (SOLO TS) ---
NUM_RUNS = 10;           
RBO_P = 0.9;             

MODELS = {'TS'}; % <--- RESTRINGIDO SOLO A TS
PERTURBATIONS = {'GAUSSIAN', 'PARETO'};
CASES = [1, 2, 3]; 

% Nombres de Casos y Carpetas
dirs_pp = {fullfile('pumping','16_features_pump'), fullfile('pumping','34_features_pump'), fullfile('pumping','34_features_AE_pump')};
suf_pp = {'MEAN_RBO', 'STANDARD_RBO', 'MOMENTS_AE'};
case_names = {'Case 1 (MEAN)', 'Case 2 (STANDARD)', 'Case 3 (AE MOMENTS)'};

fprintf('=========================================================\n');
fprintf(' INICIANDO EJECUCIÓN PARCHE (Solo Bombeo TS)\n');
fprintf('=========================================================\n\n');

%% --- BUCLE 1: MODELOS (Solo TS) ---
for m_idx = 1:length(MODELS)
    TIPO_MODELO = MODELS{m_idx};
    fprintf('\n>>>>>>>>>> CARGANDO MODELO: %s <<<<<<<<<<\n', TIPO_MODELO);
    
    % 1. CARGA DE DATOS DEL MODELO
    fname_base = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    possible_paths = {fullfile('..', 'results_mpc'), 'results_mpc', '.'};
    fname = ''; for i=1:length(possible_paths), p=fullfile(possible_paths{i}, fname_base); if isfile(p), fname=p; break; end; end
    if isempty(fname), error('Datos no encontrados: %s', fname_base); end
    results = load(fname);
    
    fname_prof = ''; for i=1:length(possible_paths), p=fullfile(possible_paths{i}, '../utils/full_profiles_for_sim.mat'); if isfile(p), fname_prof=p; break; end; end
    if isempty(fname_prof), fname_prof = 'utils/full_profiles_for_sim.mat'; end
    profiles = load(fname_prof);
    
    mg = results.mg; Q_t = results.Q_t; Q_p = results.Q_p; 
    Ts_ratio = mg(1).Ts_mpc / mg(1).Ts_sim; indices_mpc = 1:Ts_ratio:size(Q_t, 1);
    if indices_mpc(end) > size(Q_t, 1), indices_mpc(end) = []; end
    
    % 2. DETECCIÓN DE ESCENARIOS (Lógica Ratio Consolidada y Corregida)
    intercambio_total = sum(abs(Q_t(indices_mpc, :)), 2); [~, idx_peak] = max(intercambio_total);
    k_peak_global = indices_mpc(idx_peak); 
    
    k_alt_vec = zeros(1, 3);
    valid_altruism = false(1, 3); % BANDERA DE ALTRUISMO VÁLIDO
    for i=1:3
        q_t_sub = Q_t(indices_mpc, i); q_p_sub = Q_p(indices_mpc, i);
        mask = (q_t_sub > 0.01) & (q_p_sub > 0.1); scores = zeros(size(q_t_sub));
        if any(mask)
            ratio = q_t_sub(mask) ./ (q_p_sub(mask) + 1e-6); ratio(ratio > 1) = 1; 
            scores(mask) = q_t_sub(mask) .* ratio; [~, idx_sc] = max(scores); k_alt_vec(i) = indices_mpc(idx_sc);
            valid_altruism(i) = true; % MARCAMOS COMO ALTRUISMO VÁLIDO
        else
            k_alt_vec(i) = k_peak_global; 
        end
    end
    
    % Solo nos importa el Altruismo para este parche
    scenarios(1).name='Altruismo'; scenarios(1).k_list=k_alt_vec;
    
    % 3. COMPILAR CONTROLADOR YALMIP 
    fprintf('Compilando YALMIP MPC para modelo %s...\n', TIPO_MODELO);
    [~, params_init] = reconstruct_state_matlab_3mg(k_peak_global, TIPO_MODELO);
    controller_obj = get_compiled_mpc_controller_3mg(params_init.mg);
    
    %% --- BUCLE 2: ESCENARIOS Y AGENTES ---
    for s_idx = 1:length(scenarios)
        scn = scenarios(s_idx);
        
        for t_idx = 1:3
            k_target = scn.k_list(t_idx);
            
            % Si no hubo altruismo válido, saltamos a la siguiente MG
            if ~valid_altruism(t_idx)
                fprintf('\n  [!] MG%d sin evento altruista válido. Saltando.\n', t_idx);
                continue; 
            end
            
            fprintf('\n  [+] Evaluando BOMBEO en MG%d, Escenario: %s (K=%d)\n', t_idx, scn.name, k_target);
            
            % Reconstrucción Pesada
            [estado_raw, params_raw] = reconstruct_state_matlab_3mg(k_target, TIPO_MODELO);
            
            try, P_dem = estado_raw.constants.p_dem_pred_full; P_gen = estado_raw.constants.p_gen_pred_full; Q_dem = estado_raw.constants.q_dem_pred_full;
            catch, P_dem = params_raw.P_dem_pred; P_gen = params_raw.P_gen_pred; Q_dem = params_raw.Q_dem_pred; end
            
            m_P_dem = mean(P_dem, 1); m_P_gen = mean(P_gen, 1); m_Q_dem = mean(Q_dem, 1);
            max_P_gen = max(P_gen, [], 1); max_P_dem = max(P_dem, [], 1); max_Q_dem = max(Q_dem, [], 1);
            std_P_gen = std(P_gen, 0, 1); std_P_dem = std(P_dem, 0, 1); std_Q_dem = std(Q_dem, 0, 1);
            
            %% --- BUCLE 3: PERTURBACIÓN Y CASOS ---
            for p_idx = 1:length(PERTURBATIONS)
                PERTURBATION_TYPE = PERTURBATIONS{p_idx};
                
                for c_idx = CASES
                    fprintf('      -> %s | %s | ', case_names{c_idx}, PERTURBATION_TYPE);
                    
                    % 1. Configurar Features según el Caso
                    estado = estado_raw; 
                    
                    estado.X_original([3,8,13]) = m_P_dem;
                    estado.X_original([4,9,14]) = m_P_gen;
                    estado.X_original([5,10,15]) = m_Q_dem;
                    
                    if c_idx == 1 % CASO 1: MEAN (16 variables)
                        prefixes = {'P_dem', 'P_gen', 'Q_dem'};
                        idx_replace = [3, 4, 5, 8, 9, 10, 13, 14, 15]; count = 1;
                        for m = 1:3, for p = 1:3, old = estado.feature_names{idx_replace(count)};
                            if ~startsWith(old, 'Mean_'), estado.feature_names{idx_replace(count)} = strrep(old, prefixes{p}, ['Mean_' prefixes{p}]); end
                            count = count + 1; end; end
                    else % CASO 2 y 3: MOMENTS (34 variables)
                        estado.X_original = [estado.X_original, max_P_gen, max_P_dem, max_Q_dem, std_P_gen, std_P_dem, std_Q_dem];
                        for i=1:length(estado.feature_names), if ~startsWith(estado.feature_names{i},'Mean_') && i~=1 && i~=2 && i~=6 && i~=7 && i~=11 && i~=12 && i~=16
                            estado.feature_names{i}=['Mean_' estado.feature_names{i}]; end; end
                        new_names = {};
                        for i=1:3, new_names{end+1}=sprintf('Max_P_gen_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Max_P_dem_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Max_Q_dem_MG%d',i); end
                        for i=1:3, new_names{end+1}=sprintf('Std_P_gen_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Std_P_dem_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Std_Q_dem_MG%d',i); end
                        estado.feature_names = [estado.feature_names, new_names];
                    end
                    
                    % 3. EJECUCIÓN EXCLUSIVA DE BOMBEO (Q_p)
                    try
                        if c_idx == 1 % Mean
                            [lime_pp, expl_pp] = calculate_lime_pumping_3mg_with_quality(estado, controller_obj, params_raw, NUM_RUNS, t_idx, PERTURBATION_TYPE);
                        elseif c_idx == 2 % Standard Moments
                            if strcmp(PERTURBATION_TYPE,'PARETO'), [lime_pp, expl_pp] = calculate_lime_pumping_3mg_STANDARD_MOMENTS_PARETO(estado, controller_obj, params_raw, NUM_RUNS, t_idx);
                            else, [lime_pp, expl_pp] = calculate_lime_pumping_3mg_STANDARD_MOMENTS(estado, controller_obj, params_raw, NUM_RUNS, t_idx); end
                        elseif c_idx == 3 % AE Moments
                            if strcmp(PERTURBATION_TYPE,'PARETO'), [lime_pp, expl_pp] = calculate_lime_pumping_3mg_with_AE_MOMENTS_PARETO(estado, controller_obj, params_raw, NUM_RUNS, t_idx);
                            else, [lime_pp, expl_pp] = calculate_lime_pumping_3mg_with_AE_MOMENTS(estado, controller_obj, params_raw, NUM_RUNS, t_idx); end
                        end
                        rbo_pp = compute_rbo_stats(expl_pp, NUM_RUNS, RBO_P);
                        
                        f_pp = fullfile(dirs_pp{c_idx}, sprintf('lime_pumping_Scenario_%s_%s_MG%d_%s_%s.mat', scn.name, TIPO_MODELO, t_idx, suf_pp{c_idx}, PERTURBATION_TYPE));
                        feature_names = estado.feature_names; lime_stats = lime_pp; rbo_stats = rbo_pp; all_explanations = expl_pp;
                        
                        % --- FIX APLICADO: AHORA SÍ DECLARAMOS LAS VARIABLES ---
                        K_TARGET = k_target; 
                        target_mg_idx = t_idx;
                        
                        save(f_pp, 'all_explanations', 'feature_names', 'lime_stats', 'rbo_stats', 'K_TARGET', 'target_mg_idx', 'scn', 'estado');
                        fprintf('Q_p[R2:%.2f|RBO:%.2f]\n', lime_pp.R2_mean, rbo_pp.mean);
                    catch ME
                        fprintf('Q_p[ERROR: %s]\n', ME.message);
                    end
                    
                end % Fin Casos
            end % Fin Perturbaciones
        end % Fin Targets
    end % Fin Escenarios
end % Fin Modelos

fprintf('\n=== EJECUCIÓN PARCHE DE BOMBEO CONCLUIDA EXITOSAMENTE ===\n');

%% --- FUNCIONES AUXILIARES GLOBALES ---

function rbo_stats = compute_rbo_stats(all_explanations, NUM_RUNS, RBO_P)
    run_rankings = cell(1, NUM_RUNS);
    for r = 1:NUM_RUNS
        expl = all_explanations{r}; weights = cell2mat(expl(:,2));
        [~, sort_idx] = sort(abs(weights), 'descend'); run_rankings{r} = expl(sort_idx, 1);
    end
    rbo_values = [];
    for i = 1:NUM_RUNS
        for j = i+1:NUM_RUNS
            score = calculate_rbo_score(run_rankings{i}, run_rankings{j}, RBO_P);
            rbo_values = [rbo_values, score];
        end
    end
    if isempty(rbo_values), rbo_stats.mean = 1; rbo_stats.std = 0; else, rbo_stats.mean = mean(rbo_values); rbo_stats.std = std(rbo_values); end
    rbo_stats.all_pairwise = rbo_values; rbo_stats.p_param = RBO_P;
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