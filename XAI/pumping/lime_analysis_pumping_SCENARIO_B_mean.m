%% --- Archivo: lime_analysis_pumping_ALTRUISMO_robust.m ---
% ANÁLISIS LIME DE BOMBEO (Q_p) - ESCENARIO B (ALTRUISMO)
% REPLICA EL CONTROLADOR COMPLETO (Theis Drawdown + Cost Function)
%--------------------------------------------------------------------------
close all; clear; clc;

% --- CONFIGURACIÓN ---
NUM_RUNS = 1; 
fprintf('--- LIME BOMBEO (Q_p): ALTRUISMO CON CONTROLADOR ROBUSTO ---\n');

%% 1. CARGA DE DATOS
try
    results = load('results_mpc/resultados_mpc_3mg_7dias.mat');
    profiles = load('utils/full_profiles_for_sim.mat');
    mg = results.mg;
    fprintf('Datos cargados correctamente.\n');
catch
    error('Faltan archivos de resultados.');
end

Q_t = results.Q_t; Q_p = results.Q_p; 
Ts_ratio = mg(1).Ts_mpc / mg(1).Ts_sim;
indices_mpc = 1:Ts_ratio:size(Q_p, 1);
if indices_mpc(end) > size(Q_p, 1), indices_mpc(end) = []; end

%% 2. DETECCIÓN ESCENARIO B (ALTRUISMO)
k_alt_vec = zeros(1, 3);
for i=1:3
    q_t_sub = Q_t(indices_mpc, i); q_p_sub = Q_p(indices_mpc, i);
    mask = (q_t_sub > 0.01) & (q_p_sub > 0.1);
    scores = zeros(size(q_t_sub));
    if any(mask)
        ratio = q_t_sub(mask) ./ (q_p_sub(mask) + 1e-6);
        ratio(ratio > 1) = 1; 
        scores(mask) = q_t_sub(mask) .* ratio;
        [~, idx_sc] = max(scores);
        k_alt_vec(i) = indices_mpc(idx_sc);
    else
        [~, idx_sys] = max(sum(abs(Q_t(indices_mpc, :)), 2));
        k_alt_vec(i) = indices_mpc(idx_sys);
    end
end
scn_name = 'Altruismo';

%% 3. COMPILACIÓN DEL CONTROLADOR ROBUSTO (Usa Script 2 Logic)
fprintf('\nCompilando Controlador MPC Robusto (Theis + Full Costs)...\n');
[~, params_init] = reconstruct_state_matlab_3mg(k_alt_vec(1));
controller_obj = get_compiled_mpc_controller_3mg_robust(params_init.mg);

%% 4. BUCLE LIME PARA MG1 (ALTRUISMO Q_p)
t_idx = 1; % Analizamos la MG1 (La exportadora principal)
k_target = k_alt_vec(t_idx);
fprintf('  > Analizando MG%d en K=%d (Día %.2f)... ', t_idx, k_target, k_target*mg(1).Ts_sim/86400);

[estado, params] = reconstruct_state_matlab_3mg(k_target);

% Feature Engineering: MEAN predictions
P_dem_pred = estado.constants.p_dem_pred_full; 
P_gen_pred = estado.constants.p_gen_pred_full; 
Q_dem_pred = estado.constants.q_dem_pred_full; 

m_P_dem = mean(P_dem_pred, 1); m_P_gen = mean(P_gen_pred, 1); m_Q_dem = mean(Q_dem_pred, 1);
base_idx = [3, 8, 13];
for m = 1:3
    bi = base_idx(m);
    estado.X_original(bi) = m_P_dem(m); 
    estado.X_original(bi+1) = m_P_gen(m); 
    estado.X_original(bi+2) = m_Q_dem(m);
    for p = 1:3, estado.feature_names{bi+p-1} = ['Mean_' estado.feature_names{bi+p-1}]; end
end

estado.Y_target_real_vector = results.Q_p(k_target, :); 
[~, all_explanations] = evalc('calculate_lime_pumping_3mg(estado, controller_obj, params, NUM_RUNS, t_idx)');

% Guardar con el nombre que espera el script de gráficos
filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', scn_name, t_idx);
feature_names = estado.feature_names;
save(filename, 'all_explanations', 'feature_names', 'estado', 'k_target');
fprintf('[OK]\n');

%% --- FUNCIÓN COMPILADORA ROBUSTA ---
function Controller = get_compiled_mpc_controller_3mg_robust(mg_array)
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
                descenso = descenso + (delta_Q_p_futuro(l,i)/1000) * expint(u); % MODELO THEIS
            end
            constraints = [constraints, s_pozo(k,i) == (1/(4*pi*T_val)) * descenso]; 
            constraints = [constraints, s_pozo(k,i) <= mg_array(1).s_max + slack_s_pozo(k,i)];
            
            % Dinámica y Balances Completos
            P_p_val = mg_array(i).Mp * 9800 * Q_p(k,i) * (mg_array(i).h_ptub + mg_array(i).h_Tank_max)/1e6;
            constraints = [constraints, P_mgref(k,i) + P_shed(k,i) == p_dem_pred_var(k,i) - p_gen_pred_var(k,i) + P_p_val - P_B(k,i)];
            dem_agua = q_dem_pred_var(k,i) - Q_shed(k,i);
            constraints = [constraints, V_tank(k+1,i) == V_tank(k,i) + (Q_p(k,i) + Q_buy(k,i) - Q_t(k,i) - dem_agua)*Ts];
            constraints = [constraints, mg_array(i).E_batt_max * SoC(k+1,i) == mg_array(i).E_batt_max * SoC(k,i) - P_B(k,i)*(Ts/3600)];
            constraints = [constraints, mg_array(i).SoC_min <= SoC(k+1,i) <= mg_array(i).SoC_max];
            constraints = [constraints, 0 <= V_tank(k+1,i) <= mg_array(i).V_max];
            constraints = [constraints, 0 <= Q_p(k,i) <= mg_array(i).Mp * mg_array(i).Q_pump_max_unit];
        end
        constraints = [constraints, sum(Q_t(k,:)) == 0];
        constraints = [constraints, 0 <= sum(P_mgref(k,:)) <= mg_array(1).P_grid_max];
        
        % Función Objetivo Completa
        p_dno = sum(P_mgref(k,:)); q_dno = sum(Q_buy(k,:));
        if k==1, dP = P_mgref(k,:) - p_mgref_hist_0_var; dQ = Q_p(k,:) - q_p_hist_0_var;
        else, dP = P_mgref(k,:) - P_mgref(k-1,:); dQ = Q_p(k,:) - Q_p(k-1,:); end
        objective = objective + C_p*p_dno*(Ts/3600) + C_q*q_dno*Ts/1000 + lambda_P*sum(dP.^2) + lambda_Q*sum(dQ.^2) + costo_shed*(sum(P_shed(k,:)) + sum(Q_shed(k,:)));
    end
    constraints = [constraints, EAW(2:end) == EAW(1:end-1) + (mg_array(1).Rp * Ts/60) - sum(Q_p, 2) * Ts];
    constraints = [constraints, EAW(end) >= mg_array(1).V_aq_0 - slack_EAW];
    objective = objective + costo_shed*slack_EAW + costo_shed*sum(sum(slack_s_pozo));
    
    ops = sdpsettings('solver','gurobi','verbose',0);
    Inputs = {soC_0_var, v_tank_0_var, v_aq_0_var, p_dem_pred_var, p_gen_pred_var, q_dem_pred_var, q_p_hist_0_var, p_mgref_hist_0_var, k_mpc_var, Q_p_hist_mpc_var};
    Outputs = {P_mgref, Q_p, Q_buy, Q_t, s_pozo};
    Controller = optimizer(constraints, objective, ops, Inputs, Outputs);
end