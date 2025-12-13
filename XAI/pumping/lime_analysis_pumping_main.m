%% --- Archivo: lime_analysis_pumping_main.m ---
%
% SCRIPT MAESTRO: ANÁLISIS DE BOMBEO (Q_p) - VERSIÓN ASÍNCRONA
%
% NOVEDAD TÉCNICA:
% 1. Para 'Energy Efficiency', busca el pico óptimo INDIVIDUAL para cada MG.
%    (Evita analizar ceros si las MGs operan a horas distintas).
% 2. Para 'Aquifer Constraint', mantiene un pico GLOBAL (Estrés compartido).
%
%--------------------------------------------------------------------------
close all; clear; clc;

NUM_RUNS = 1; 

fprintf('--- LIME BOMBEO (Q_p): ANÁLISIS ASÍNCRONO HÍBRIDO ---\n');

%% 1. CARGA DE DATOS
try
    results = load('results_mpc/resultados_mpc_3mg_7dias.mat');
    profiles = load('utils/full_profiles_for_sim.mat');
    mg = results.mg;
    fprintf('Datos cargados correctamente.\n');
catch
    error('Faltan archivos de resultados. Ejecute main_mpc.m primero.');
end

Q_p = results.Q_p; 
h_p = results.h_p; 
Q_dem = profiles.Q_dem_sim; 
P_gen = profiles.P_gen_sim; 

% Índices válidos (Pasos MPC)
Ts_ratio = mg(1).Ts_mpc / mg(1).Ts_sim;
indices_mpc = 1:Ts_ratio:size(Q_p, 1);
if indices_mpc(end) > size(Q_p, 1), indices_mpc(end) = []; end

%% 2. DETECCIÓN DE ESCENARIOS (HÍBRIDA)

% --- ESCENARIO 1: ENERGY EFFICIENCY (Individual) ---
% Buscamos el mejor momento K para CADA micro-red por separado.
k_solar_vec = zeros(1, 3);
fprintf('\n>>> DETECTANDO PICOS DE EFICIENCIA ENERGÉTICA (INDIVIDUALES) <<<\n');

for i=1:3
    q_p_sub = Q_p(indices_mpc, i);
    p_gen_sub = P_gen(indices_mpc, i);
    
    % Score: Coincidencia Bombeo * Sol
    scores = q_p_sub .* p_gen_sub;
    [max_val, idx_max] = max(scores);
    
    k_solar_vec(i) = indices_mpc(idx_max);
    
    fprintf('  MG%d: Mejor momento en K=%d (Día %.2f) -> Score: %.1f\n', ...
        i, k_solar_vec(i), k_solar_vec(i)*mg(1).Ts_sim/86400, max_val);
end

% --- ESCENARIO 2: AQUIFER CONSTRAINT (Global) ---
% Buscamos el momento de mayor estrés para el sistema COMPLETO.
% Usamos el mismo K para todos para ver la reacción simultánea.
fprintf('\n>>> DETECTANDO RESTRICCIÓN DE ACUÍFERO (GLOBAL) <<<\n');

h_p0 = mg(1).h_p0;
% Suma de descensos (Drawdown total)
s_total = sum(h_p(indices_mpc, :) - h_p0, 2);
% Demanda total
q_dem_total = sum(Q_dem(indices_mpc, :), 2);

% Score: Estrés = Drawdown Total * Demanda Total
scores_aq = s_total .* q_dem_total;
[~, idx_aq] = max(scores_aq);
k_aquifer_global = indices_mpc(idx_aq);

% Vectorizamos el K global para que la estructura sea uniforme
k_aquifer_vec = [k_aquifer_global, k_aquifer_global, k_aquifer_global];

fprintf('  Sistema: Máximo Estrés en K=%d (Día %.2f)\n', ...
    k_aquifer_global, k_aquifer_global*mg(1).Ts_sim/86400);


% --- ESTRUCTURA DE ESCENARIOS ---
% Ahora 'k_list' es un vector de 3 elementos [k_mg1, k_mg2, k_mg3]
scenarios(1).name = 'EnergyEfficiency';  scenarios(1).k_list = k_solar_vec;
scenarios(2).name = 'AquiferConstraint'; scenarios(2).k_list = k_aquifer_vec;

%% 3. PRE-COMPILACIÓN (Usamos un estado dummy para compilar)
fprintf('\nCompilando Controlador MPC 3-MG...\n');
[~, params_init] = reconstruct_state_matlab_3mg(k_solar_vec(1));
controller_obj = get_compiled_mpc_controller_3mg(params_init.mg);

%% 4. BUCLE MAESTRO DE ANÁLISIS
for s_idx = 1:length(scenarios)
    scn = scenarios(s_idx);
    
    fprintf('\n==========================================================\n');
    fprintf(' PROCESANDO ESCENARIO: %s\n', scn.name);
    fprintf('==========================================================\n');
    
    % Bucle por Micro-red (Ahora cada una puede tener su propio tiempo)
    for t_idx = 1:3
        
        % 1. Seleccionar el K específico para esta MG en este escenario
        k_target_actual = scn.k_list(t_idx);
        
        fprintf('  > Analizando MG%d en K=%d... ', t_idx, k_target_actual);
        
        % 2. Reconstruir Estado en ese instante específico
        [estado, params] = reconstruct_state_matlab_3mg(k_target_actual);
        
        % 3. Obtener valor real de Q_p para referencia
        val_real = results.Q_p(k_target_actual, t_idx);
        fprintf('(Q_p Real = %.3f L/s)... ', val_real);
        
        if val_real < 0.01 && strcmp(scn.name, 'EnergyEfficiency')
            fprintf('[SKIP] (Inactiva)\n'); 
            % Opcional: saltar si es cero, pero LIME puede explicar ceros también.
            % Lo dejamos correr para ver por qué es cero.
        end
        
        % 4. Ejecutar LIME BOMBEO
        % Silenciamos salida interna
        [~, all_explanations] = evalc('calculate_lime_pumping_3mg(estado, controller_obj, params, NUM_RUNS, t_idx)');
        
        feature_names = estado.feature_names;
        
        % 5. Guardar
        filename = sprintf('lime_PUMP_%s_MG%d.mat', scn.name, t_idx);
        
        % Guardamos variables clave para el plotter
        K_TARGET = k_target_actual; 
        target_mg_idx = t_idx;
        
        save(filename, 'all_explanations', 'feature_names', 'estado', ...
             'K_TARGET', 'target_mg_idx', 'scn');
         
        fprintf('[OK]\n');
    end
end

fprintf('\n=== ANÁLISIS DE BOMBEO ASÍNCRONO COMPLETADO ===\n');

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