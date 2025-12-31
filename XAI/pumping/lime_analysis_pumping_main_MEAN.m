%% --- Archivo: lime_analysis_pumping_main_MEAN.m ---
%
% SCRIPT MAESTRO: ANÁLISIS DE BOMBEO (Q_p) - ENFOQUE PROMEDIO (MEAN)
%
% Este script detecta los instantes críticos para bombeo y aplica LIME 
% utilizando promedios de predicciones como características de entrada.
%--------------------------------------------------------------------------

close all; clear; clc;

% --- CONFIGURACIÓN ---
NUM_RUNS = 1; % Incrementar para mayor estabilidad estadística (ej. 10 o 20)
fprintf('--- LIME BOMBEO (Q_p): ANÁLISIS CON FEATURE ENGINEERING (MEAN) ---\n');

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
Q_dem_real = profiles.Q_dem_sim; 
P_gen_real = profiles.P_gen_sim; 

% Índices válidos (Pasos de decisión del MPC)
Ts_ratio = mg(1).Ts_mpc / mg(1).Ts_sim;
indices_mpc = 1:Ts_ratio:size(Q_p, 1);
if indices_mpc(end) > size(Q_p, 1), indices_mpc(end) = []; end

%% 2. DETECCIÓN DE ESCENARIOS (BÚSQUEDA MEDIANTE VALORES INSTANTÁNEOS)
% ESCENARIO D: ENERGY EFFICIENCY (Individual)
k_solar_vec = zeros(1, 3);
fprintf('\n>>> DETECTANDO EVENTOS DE EFICIENCIA ENERGÉTICA (BOMBEO SOLAR) <<<\n');
for i=1:3
    q_p_sub = Q_p(indices_mpc, i);
    p_gen_sub = P_gen_real(indices_mpc, i);
    % Score: Coincidencia de bombeo alto con generación solar alta
    scores = q_p_sub .* p_gen_sub;
    [~, idx_max] = max(scores);
    k_solar_vec(i) = indices_mpc(idx_max);
    fprintf('  MG%d: Evento detectado en K=%d (Día %.2f)\n', ...
        i, k_solar_vec(i), k_solar_vec(i)*mg(1).Ts_sim/86400);
end

scenarios(1).name = 'EnergyEfficiency'; scenarios(1).k_list = k_solar_vec;

%% 3. PRE-COMPILACIÓN DEL CONTROLADOR
fprintf('\nCompilando Controlador MPC para auditoría LIME...\n');
[~, params_init] = reconstruct_state_matlab_3mg(k_solar_vec(1));
controller_obj = get_compiled_mpc_controller_3mg(params_init.mg);

%% 4. BUCLE MAESTRO DE ANÁLISIS LIME
for s_idx = 1:length(scenarios)
    scn = scenarios(s_idx);
    
    fprintf('\n==========================================================\n');
    fprintf(' PROCESANDO ESCENARIO: %s\n', scn.name);
    fprintf('==========================================================\n');
    
    for t_idx = 1:3
        k_target_actual = scn.k_list(t_idx);
        fprintf('  > Analizando MG%d en K=%d... ', t_idx, k_target_actual);
        
        % A. Reconstruir Estado Original
        [estado, params] = reconstruct_state_matlab_3mg(k_target_actual);
        
        % B. ACTUALIZACIÓN A ENFOQUE "MEAN" (Feature Engineering)
        % Extraer matrices completas del horizonte N
        P_dem_pred = estado.constants.p_dem_pred_full; % N x 3
        P_gen_pred = estado.constants.p_gen_pred_full; % N x 3
        Q_dem_pred = estado.constants.q_dem_pred_full; % N x 3
        
        % Calcular promedios sobre el horizonte
        mean_P_dem = mean(P_dem_pred, 1);
        mean_P_gen = mean(P_gen_pred, 1);
        mean_Q_dem = mean(Q_dem_pred, 1);
        
        % Reemplazar en el vector X_original para LIME (Índices 3-MG)
        % MG1
        estado.X_original(3) = mean_P_dem(1);
        estado.X_original(4) = mean_P_gen(1);
        estado.X_original(5) = mean_Q_dem(1);
        % MG2
        estado.X_original(8) = mean_P_dem(2);
        estado.X_original(9) = mean_P_gen(2);
        estado.X_original(10) = mean_Q_dem(2);
        % MG3
        estado.X_original(13) = mean_P_dem(3);
        estado.X_original(14) = mean_P_gen(3);
        estado.X_original(15) = mean_Q_dem(3);
        
        % C. Actualizar Nombres de Variables para Gráficos y Tablas
        names = estado.feature_names;
        idx_replace = [3, 4, 5, 8, 9, 10, 13, 14, 15];
        prefixes    = {'P_dem', 'P_gen', 'Q_dem'};
        count = 1;
        for m = 1:3 
            for p = 1:3 
                old_name = names{idx_replace(count)};
                new_name = strrep(old_name, prefixes{p}, ['Mean_' prefixes{p}]);
                estado.feature_names{idx_replace(count)} = new_name;
                count = count + 1;
            end
        end
        
        % D. Sobrescribir Target Real (Asegurar que sea Q_p)
        estado.Y_target_real_vector = results.Q_p(k_target_actual, :); 
        val_real = estado.Y_target_real_vector(t_idx);
        
        % E. Ejecutar LIME BOMBEO (Usando Wrapper Pumping)
        % Silenciamos output interno de LIME
        [~, all_explanations] = evalc('calculate_lime_pumping_3mg(estado, controller_obj, params, NUM_RUNS, t_idx)');
        
        feature_names = estado.feature_names;
        
        % F. Guardar con sufijo _MEAN para diferenciarlos de la versión instantánea
        filename = sprintf('lime_PUMP_%s_MG%d_MEAN.mat', scn.name, t_idx);
        K_TARGET = k_target_actual; 
        target_mg_idx = t_idx;
        
        save(filename, 'all_explanations', 'feature_names', 'estado', ...
             'K_TARGET', 'target_mg_idx', 'scn');
         
        fprintf('[OK] (Q_p Real = %.3f L/s)\n', val_real);
    end
end

fprintf('\n=== ANÁLISIS DE BOMBEO (MEAN) COMPLETADO ===\n');

%% --- Helper: Compilador MPC Supervisor ---
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