%% --- Archivo: lime_analysis_main_SCENARIOS_pumping_MOMENTS_RBO.m ---
%
% SCRIPT MAESTRO DE ANÁLISIS DE ESCENARIOS BOMBEO (Q_p) - AE MOMENTS (34 vars)
%
% Objetivo: Explicar bombeo en escenarios de Altruismo usando Autoencoder.
% Features: 34 Variables (Estados + Momentos).
% Métricas: R2 (Fidelidad), RBO (Estabilidad).
%
%--------------------------------------------------------------------------
close all; clear; clc;

% --- 1. CONFIGURACIÓN ---
TIPO_MODELO = 'TS';      % 'AR' o 'TS'
NUM_RUNS = 10;           % Cantidad de corridas para estadística RBO
RBO_P = 0.9;             % Persistencia RBO

% *** SELECTOR DE PERTURBACIÓN ***
PERTURBATION_TYPE = 'GAUSSIAN'; % Opciones: 'GAUSSIAN' o 'PARETO'

fprintf('--- LIME PUMPING SCENARIOS (AE MOMENTS) - Modelo: %s - Perturb: %s ---\n', TIPO_MODELO, PERTURBATION_TYPE);

%% 2. CARGA DE DATOS
try
    fname_base = sprintf('resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);
    possible_paths = {fullfile('C:', 'Users', 'hayte', 'Documents', 'MATLAB', 'sistema_micro-redes', 'results_mpc'), fullfile('..', '..', 'results_mpc'), fullfile('..', 'results_mpc'), 'results_mpc', '.'};
    fname = ''; for i=1:length(possible_paths), p=fullfile(possible_paths{i}, fname_base); if isfile(p), fname=p; break; end; end
    if isempty(fname), error('Datos no encontrados.'); end
    results = load(fname);
    
    possible_utils = {fullfile('C:', 'Users', 'hayte', 'Documents', 'MATLAB', 'sistema_micro-redes', 'utils'), fullfile('..', 'utils'), 'utils'};
    fname_prof = ''; for i=1:length(possible_utils), p=fullfile(possible_utils{i}, 'full_profiles_for_sim.mat'); if isfile(p), fname_prof=p; break; end; end
    if isempty(fname_prof), error('Perfiles no encontrados.'); end
    profiles = load(fname_prof);
    mg = results.mg; fprintf('Datos cargados: %s\n', fname);
catch ME, error(ME.message); end

Q_t = results.Q_t; Q_p = results.Q_p; 
Ts_ratio = mg(1).Ts_mpc / mg(1).Ts_sim; indices_mpc = 1:Ts_ratio:size(Q_t, 1);
if indices_mpc(end) > size(Q_t, 1), indices_mpc(end) = []; end

%% 3. DETECCIÓN DE ESCENARIOS (SOLO ALTRUISMO)
k_peak_global = indices_mpc(1); 
k_alt_vec = zeros(1, 3);
fprintf('\n>>> DETECTANDO ALTRUISMO (Para explicar Q_p) <<<\n');

for i = 1:3
    q_t_sub = Q_t(indices_mpc, i); 
    q_p_sub = Q_p(indices_mpc, i); 
    
    % Filtro: Exportación y Bombeo simultáneos
    mask = (q_t_sub > 0.01) & (q_p_sub > 0.1);
    scores = zeros(size(q_t_sub));
    
    if any(mask)
        % Ratio Eficiencia: Cuánto del bombeo se exporta
        ratio = q_t_sub(mask) ./ (q_p_sub(mask) + 1e-6);
        ratio(ratio > 1) = 1; 
        scores(mask) = q_t_sub(mask) .* ratio;
        
        [~, idx_sc] = max(scores);
        k_alt_vec(i) = indices_mpc(idx_sc);
        fprintf('  MG%d: K=%d (Score=%.3f)\n', i, k_alt_vec(i), max(scores));
    else
        k_alt_vec(i) = k_peak_global;
        fprintf('  MG%d: Sin comportamiento altruista.\n', i);
    end
end

scenarios(1).name = 'Altruismo'; 
scenarios(1).k_list = k_alt_vec;

%% 4. BUCLE MAESTRO
fprintf('\nCompilando Controlador MPC 3-MG...\n');
[~, params_init] = reconstruct_state_matlab_3mg(k_peak_global, TIPO_MODELO);
controller_obj = get_compiled_mpc_controller_3mg(params_init.mg);

for s_idx = 1:length(scenarios)
    scn = scenarios(s_idx);
    fprintf('\n==========================================================\n');
    fprintf(' PROCESANDO ESCENARIO (AE MOMENTS %s): %s\n', PERTURBATION_TYPE, scn.name);
    fprintf('==========================================================\n');
    
    for t_idx = 1:3
        k_target = scn.k_list(t_idx);
        
        if k_target == k_peak_global
             fprintf('  > MG%d: Saltando (No hay evento).\n', t_idx);
             continue;
        end
        
        % A. Reconstruir
        [estado, params] = reconstruct_state_matlab_3mg(k_target, TIPO_MODELO);
        
        % B. Feature Engineering (34 VARS)
        try, P_dem = estado.constants.p_dem_pred_full; P_gen = estado.constants.p_gen_pred_full; Q_dem = estado.constants.q_dem_pred_full;
        catch, P_dem = params.P_dem_pred; P_gen = params.P_gen_pred; Q_dem = params.Q_dem_pred; end
        
        % 1. Medias
        m_P_dem = mean(P_dem, 1); m_P_gen = mean(P_gen, 1); m_Q_dem = mean(Q_dem, 1);
        % 2. Max
        max_P_gen = max(P_gen, [], 1); max_P_dem = max(P_dem, [], 1); max_Q_dem = max(Q_dem, [], 1);
        % 3. Std
        std_P_gen = std(P_gen, 0, 1); std_P_dem = std(P_dem, 0, 1); std_Q_dem = std(Q_dem, 0, 1);
        
        x_base = estado.X_original;
        x_base([3,8,13]) = m_P_dem; x_base([4,9,14]) = m_P_gen; x_base([5,10,15]) = m_Q_dem;
        estado.X_original = [x_base, max_P_gen, max_P_dem, max_Q_dem, std_P_gen, std_P_dem, std_Q_dem];
        
        % Nombres
        for i=1:length(estado.feature_names), if ~startsWith(estado.feature_names{i},'Mean_') && i~=1 && i~=2 && i~=6 && i~=7 && i~=11 && i~=12 && i~=16, estado.feature_names{i}=['Mean_' estado.feature_names{i}]; end; end
        new_names = {};
        for i=1:3, new_names{end+1}=sprintf('Max_P_gen_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Max_P_dem_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Max_Q_dem_MG%d',i); end
        for i=1:3, new_names{end+1}=sprintf('Std_P_gen_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Std_P_dem_MG%d',i); end; for i=1:3, new_names{end+1}=sprintf('Std_Q_dem_MG%d',i); end
        estado.feature_names = [estado.feature_names, new_names];
        
        % Valor Real (BOMBEO)
        val_real = results.Q_p(k_target, t_idx);
        fprintf('  > Analizando MG%d en K=%d... (Q_p=%.3f)... ', t_idx, k_target, val_real);
        
        % *** C. EJECUTAR LIME AE BOMBEO ***
        if strcmp(PERTURBATION_TYPE, 'PARETO')
            [lime_stats, all_explanations] = calculate_lime_pumping_3mg_with_AE_MOMENTS_PARETO(...
                estado, controller_obj, params, NUM_RUNS, t_idx);
        else
            [lime_stats, all_explanations] = calculate_lime_pumping_3mg_with_AE_MOMENTS(...
                estado, controller_obj, params, NUM_RUNS, t_idx);
        end
        
        % --- D. CÁLCULO RBO ---
        run_rankings = cell(1, NUM_RUNS);
        for r = 1:NUM_RUNS
            expl = all_explanations{r}; weights = cell2mat(expl(:,2));
            [~, sort_idx] = sort(abs(weights), 'descend'); run_rankings{r} = expl(sort_idx, 1);
        end
        rbo_values = [];
        for i = 1:NUM_RUNS, for j = i+1:NUM_RUNS, score = calculate_rbo_score(run_rankings{i}, run_rankings{j}, RBO_P); rbo_values = [rbo_values, score]; end; end
        if isempty(rbo_values), rbo_mean = 1; rbo_std = 0; else, rbo_mean = mean(rbo_values); rbo_std = std(rbo_values); end
        rbo_stats.mean = rbo_mean; rbo_stats.std = rbo_std; rbo_stats.all_pairwise = rbo_values; rbo_stats.p_param = RBO_P;
        
        fprintf('R2=%.4f | RBO=%.4f [OK]\n', lime_stats.R2_mean, rbo_mean);
        
        filename = sprintf('lime_pumping_Scenario_%s_%s_MG%d_MOMENTS_AE_%s.mat', scn.name, TIPO_MODELO, t_idx, PERTURBATION_TYPE);
        K_TARGET = k_target; target_mg_idx = t_idx; feature_names = estado.feature_names;
        save(filename, 'all_explanations', 'feature_names', 'lime_stats', 'rbo_stats', 'K_TARGET', 'target_mg_idx', 'scn');
    end
end
fprintf('\n=== FIN ANÁLISIS BOMBEO AE MOMENTS ===\n');

%% --- Helpers ---
function rbo = calculate_rbo_score(list1, list2, p)
    if nargin < 3, p = 0.9; end; k = min(length(list1), length(list2)); sum_series = 0;
    for d = 1:k, set1 = list1(1:d); set2 = list2(1:d); intersection_size = length(intersect(set1, set2)); A_d = intersection_size / d; sum_series = sum_series + (p^(d-1)) * A_d; end
    rbo = (1 - p) * sum_series;
end

function Controller = get_compiled_mpc_controller_3mg(mg_array)
    yalmip('clear');
    N = mg_array(1).N; Ts = mg_array(1).Ts_mpc; num_mg = 3;
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