%% --- Archivo: lime_analysis_main_SCENARIOS_mean.m ---
%
% SCRIPT MAESTRO DE ANÁLISIS LIME (Variable: Q_t) - ENFOQUE PROMEDIO (MEAN)
%
% MODIFICACIÓN CLAVE:
% Las variables de entrada a LIME (P_gen, P_dem, Q_dem) se reemplazan por
% el PROMEDIO del horizonte de predicción (N pasos).
% Esto dota a LIME de "contexto temporal" para explicar decisiones estratégicas.
%
% ESCENARIOS:
% 1. Global Peak
% 2. Altruismo
% 3. Direct Satisfaction
%
%--------------------------------------------------------------------------
close all; clear; clc;

% --- CONFIGURACIÓN ---
NUM_RUNS = 1; % Aumentar para resultados finales (ej. 10 o 20)
fprintf('--- LIME INTERCAMBIO (Q_t): ANÁLISIS CON FEATURE ENGINEERING (MEAN) ---\n');

%% 1. CARGA DE DATOS
try
    results = load('results_mpc/resultados_mpc_3mg_7dias.mat');
    profiles = load('utils/full_profiles_for_sim.mat');
    mg = results.mg;
    fprintf('Datos cargados correctamente.\n');
catch
    error('Faltan archivos. Ejecute main_mpc.m');
end

Q_t = results.Q_t; 
Q_p = results.Q_p; 
Q_buy = results.Q_DNO; 
Q_dem_full = profiles.Q_dem_sim; 

% Índices válidos
Ts_ratio = mg(1).Ts_mpc / mg(1).Ts_sim;
indices_mpc = 1:Ts_ratio:size(Q_t, 1);
if indices_mpc(end) > size(Q_t, 1), indices_mpc(end) = []; end

%% 2. DETECCIÓN DE ESCENARIOS (Igual al original)

% --- ESCENARIO A: GLOBAL PEAK (Sincrónico) ---
intercambio_total = sum(abs(Q_t(indices_mpc, :)), 2);
[~, idx_peak] = max(intercambio_total);
k_peak_global = indices_mpc(idx_peak);
k_peak_vec = [k_peak_global, k_peak_global, k_peak_global];
fprintf('\n>>> A. GLOBAL PEAK (Sincrónico) <<<\n');
fprintf('  Sistema: K=%d (Día %.2f)\n', k_peak_global, k_peak_global*mg(1).Ts_sim/86400);

% --- ESCENARIO B: ALTRUISMO (Individual) ---
k_alt_vec = zeros(1, 3);
fprintf('\n>>> B. ALTRUISMO (Individual) <<<\n');
for i=1:3
    q_t_sub = Q_t(indices_mpc, i);
    q_p_sub = Q_p(indices_mpc, i);
    mask = (q_t_sub > 0.01) & (q_p_sub > 0.1);
    scores = zeros(size(q_t_sub));
    if any(mask)
        ratio = q_t_sub(mask) ./ (q_p_sub(mask) + 1e-6);
        ratio(ratio > 1) = 1; 
        scores(mask) = q_t_sub(mask) .* ratio;
        [~, idx_sc] = max(scores);
        k_alt_vec(i) = indices_mpc(idx_sc);
    else
        k_alt_vec(i) = k_peak_global;
    end
end

% --- ESCENARIO C: DIRECT SATISFACTION (Individual) ---
k_direct_vec = zeros(1, 3);
fprintf('\n>>> C. DIRECT SATISFACTION (Pass-through) <<<\n');
for i=1:3
    q_t_sub = Q_t(indices_mpc, i);
    q_p_sub = Q_p(indices_mpc, i);
    q_buy_sub = Q_buy(indices_mpc, i);
    q_dem_sub = Q_dem_full(indices_mpc, i);
    
    mask = (q_t_sub < -0.1) & (q_dem_sub > 0.1);
    scores = zeros(size(q_t_sub));
    if any(mask)
        import_val = abs(q_t_sub(mask));
        dem_val = q_dem_sub(mask);
        pump_val = q_p_sub(mask);
        buy_val = q_buy_sub(mask);
        similarity = 1 - (abs(import_val - dem_val) ./ (import_val + dem_val));
        purity = 1 ./ (1 + pump_val + buy_val);
        scores(mask) = similarity .* purity .* import_val;
        [~, idx_sc] = max(scores);
        k_direct_vec(i) = indices_mpc(idx_sc);
    else
        [~, idx_fb] = min(q_t_sub); 
        k_direct_vec(i) = indices_mpc(idx_fb);
    end
end

scenarios(1).name = 'GlobalPeak';         scenarios(1).k_list = k_peak_vec;
scenarios(2).name = 'Altruismo';          scenarios(2).k_list = k_alt_vec;
scenarios(3).name = 'DirectSatisfaction'; scenarios(3).k_list = k_direct_vec;

%% 3. PRE-COMPILACIÓN
fprintf('\nCompilando Controlador MPC 3-MG...\n');
[~, params_init] = reconstruct_state_matlab_3mg(k_peak_global);
controller_obj = get_compiled_mpc_controller_3mg(params_init.mg);

%% 4. BUCLE MAESTRO
for s_idx = 1:length(scenarios)
    scn = scenarios(s_idx);
    
    fprintf('\n==========================================================\n');
    fprintf(' PROCESANDO ESCENARIO (MEAN): %s\n', scn.name);
    fprintf('==========================================================\n');
    
    for t_idx = 1:3
        k_target_actual = scn.k_list(t_idx);
        fprintf('  > Analizando MG%d en K=%d... ', t_idx, k_target_actual);
        
        % A. Reconstruir Estado Original
        [estado, params] = reconstruct_state_matlab_3mg(k_target_actual);
        
        % --- B. CORRECCIÓN: CÁLCULO DE PROMEDIOS (Desde estado.constants) ---
        % Usamos 'estado.constants' porque ahí seguro están las matrices completas
        % con los nombres que usa el wrapper (p_dem_pred_full, etc.)
        try
            P_dem_pred = estado.constants.p_dem_pred_full; % N x 3
            P_gen_pred = estado.constants.p_gen_pred_full; % N x 3
            Q_dem_pred = estado.constants.q_dem_pred_full; % N x 3
        catch
            % Fallback por si los nombres son distintos en tu versión de reconstruct
            warning('Usando nombres alternativos para predicciones...');
            P_dem_pred = params.P_dem_pred; 
            P_gen_pred = params.P_gen_pred; 
            Q_dem_pred = params.Q_dem_pred;
        end
        
        % Calculamos el promedio sobre el horizonte (Dimensión 1 es el tiempo)
        mean_P_dem = mean(P_dem_pred, 1);
        mean_P_gen = mean(P_gen_pred, 1);
        mean_Q_dem = mean(Q_dem_pred, 1);
        
        % Reemplazamos en el vector X_original para LIME
        % Índices Standard: 
        % MG1: P_dem(3), P_gen(4), Q_dem(5)
        % MG2: P_dem(8), P_gen(9), Q_dem(10)
        % MG3: P_dem(13), P_gen(14), Q_dem(15)
        
        % MG1 Replacement
        estado.X_original(3) = mean_P_dem(1);
        estado.X_original(4) = mean_P_gen(1);
        estado.X_original(5) = mean_Q_dem(1);
        
        % MG2 Replacement
        estado.X_original(8) = mean_P_dem(2);
        estado.X_original(9) = mean_P_gen(2);
        estado.X_original(10) = mean_Q_dem(2);
        
        % MG3 Replacement
        estado.X_original(13) = mean_P_dem(3);
        estado.X_original(14) = mean_P_gen(3);
        estado.X_original(15) = mean_Q_dem(3);
        
        % --- C. ACTUALIZAR NOMBRES DE VARIABLES ---
        names = estado.feature_names;
        idx_replace = [3, 4, 5, 8, 9, 10, 13, 14, 15];
        prefixes    = {'P_dem', 'P_gen', 'Q_dem'};
        
        count = 1;
        for m = 1:3 % MGs
            for p = 1:3 % Vars
                old_name = names{idx_replace(count)};
                % Agregamos "Mean_" al nombre para el gráfico
                new_name = strrep(old_name, prefixes{p}, ['Mean_' prefixes{p}]);
                estado.feature_names{idx_replace(count)} = new_name;
                count = count + 1;
            end
        end
        % -------------------------------------------------------------------
        
        val_real = estado.Y_target_real_vector(t_idx);
        fprintf('(Q_t Real = %.3f L/s)... ', val_real);
        
        % Ejecutar LIME
        [~, all_explanations] = evalc('calculate_lime_stability_3mg(estado, controller_obj, params, NUM_RUNS, t_idx)');
        
        feature_names = estado.feature_names;
        
        % Guardamos con sufijo _MEAN
        filename = sprintf('lime_Scenario_%s_MG%d_MEAN.mat', scn.name, t_idx);
        K_TARGET = k_target_actual; 
        target_mg_idx = t_idx;
        
        save(filename, 'all_explanations', 'feature_names', 'estado', ...
             'K_TARGET', 'target_mg_idx', 'scn');
        fprintf('[OK]\n');
    end
end
fprintf('\n=== ANÁLISIS DE INTERCAMBIO (MEAN) COMPLETADO ===\n');

%% --- Helper: Compilador MPC ---
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