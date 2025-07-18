function [SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, h_p, Q_t] = sim_microrred_mpc(mg, P_dem, P_gen, Q_dem)
    % Simulador principal para MÚLTIPLES micro-redes con controlador MPC y predicciones imperfectas.
    
    %% --- 1. Extracción de Parámetros y Preparación ---
    Ts_mpc = mg(1).Ts_mpc;
    Ts_sim = mg(1).Ts_sim;
    num_mg = length(mg);
    Nt_sim = size(P_dem, 1);
    
    % Período de arranque en pasos de simulación (1 min)
    T_arranque_sim = mg(1).max_lags_mpc * (Ts_mpc / Ts_sim);
    
    % Inicialización de matrices de estado
    SoC = zeros(Nt_sim, num_mg); V_tank = zeros(Nt_sim, num_mg);
    P_grid = zeros(Nt_sim, num_mg); Q_p = zeros(Nt_sim, num_mg);
    Q_DNO = zeros(Nt_sim, num_mg); P_pump = zeros(Nt_sim, num_mg);
    Q_t = zeros(Nt_sim, num_mg); V_aq = zeros(Nt_sim, 1);
    h_p = zeros(Nt_sim, num_mg);

    % Condiciones iniciales
    SoC(1, :) = 0.5; V_tank(1, :) = 20000;
    V_aq(1) = mg(1).V_aq_0; h_p(1, :) = mg(1).h_p0;
    
    % Variables para mantener constantes las acciones del MPC
    P_mgref_k = zeros(1, num_mg); Q_p_k = zeros(1, num_mg);
    Q_buy_k = zeros(1, num_mg); Q_t_k = zeros(1, num_mg);
    
    %% --- 2. Lazo Principal de Simulación ---
    for k = 1:Nt_sim - 1
        
        % --- Ejecutar MPC solo DESPUÉS del período de arranque ---
        if k > T_arranque_sim && mod(k - T_arranque_sim, Ts_mpc / Ts_sim) == 0
            fprintf('Ejecutando MPC en t = %.1f horas...\n', (k-1)*Ts_sim/3600);
            
            % Preparar datos históricos (submuestreados para el predictor)
            paso_mpc = Ts_mpc / Ts_sim;
            hist_data.P_dem = P_dem(1:paso_mpc:k, :);
            hist_data.P_gen = P_gen(1:paso_mpc:k, :);
            hist_data.Q_dem = Q_dem(1:paso_mpc:k, :);
            
            [P_dem_pred, P_gen_pred, Q_dem_pred] = ...
                generar_predicciones_AR(mg(1).modelos, hist_data, mg(1).N);
            
            [u_mpc] = controlador_mpc(mg, SoC(k,:), V_tank(k,:), V_aq(k), P_dem_pred, P_gen_pred, Q_dem_pred);
            
            if ~isempty(u_mpc)
                P_mgref_k = u_mpc.P_mgref; Q_p_k = u_mpc.Q_p;
                Q_buy_k = u_mpc.Q_buy; Q_t_k = u_mpc.Q_t;
            else
                fprintf('MPC falló. Manteniendo la última acción de control.\n');
            end
        end
        
        % --- Aplicar acciones y simular dinámica (en cada paso de 1 min) ---
        for i = 1:num_mg
            Q_tank_net = Q_p_k(i) + Q_buy_k(i) - Q_t_k(i) - Q_dem(k, i);
            V_tank(k+1, i) = V_tank(k, i) + Q_tank_net * Ts_sim;
            V_tank(k+1, i) = min(max(V_tank(k+1, i), 0), mg(i).V_max);
            
            Q_p(k, i) = Q_p_k(i); Q_DNO(k, i) = Q_buy_k(i); Q_t(k, i) = Q_t_k(i);
            
            P_pump_i = (mg(i).Mp * 9800 * Q_p_k(i) * (mg(i).h_ptub + mg(i).h_Tank_max)) / 1000;
            P_pump(k, i) = P_pump_i;

            P_batt_target = P_dem(k, i) - P_gen(k, i) + P_pump_i - P_mgref_k(i);
            
            P_chg_max = mg(i).alpha_C * mg(i).P_batt_max * (1 - SoC(k,i));
            P_dis_max = mg(i).alpha_D * mg(i).P_batt_max * SoC(k,i);
            
            P_B_actual = max(-P_chg_max, min(P_batt_target, P_dis_max));

            SoC(k+1, i) = SoC(k, i) - (P_B_actual * Ts_sim / 3600) / mg(i).E_batt_max;
            SoC(k+1, i) = min(max(SoC(k+1, i), mg(i).SoC_min), mg(i).SoC_max);
            
            P_grid(k, i) = P_mgref_k(i) + (P_batt_target - P_B_actual);
        end
        
        % --- Actualización de Recursos Compartidos ---
        recarga_total = (mg(1).Rp * (Ts_sim / 60));
        bombeo_total = sum(Q_p_k) * Ts_sim;
        V_aq(k+1) = max(V_aq(k) + recarga_total - bombeo_total, 0);
        
        for i = 1:num_mg
            s_k = (Q_p_k(i) / (4 * pi * mg(i).T_aq)) * 1; 
            h_p(k+1, i) = mg(i).h_p0 + s_k;
        end
    end
end