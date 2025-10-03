function [SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, h_p, Q_t] = sim_microrred_mpc(mg, P_dem, P_gen, Q_dem, hist_arranque)
    
    %% --- 1. Extracción de Parámetros y Preparación ---
    Ts_mpc = mg(1).Ts_mpc; Ts_sim = mg(1).Ts_sim;
    num_mg = length(mg); Nt_sim = size(P_dem, 1);
    paso_mpc_en_sim = Ts_mpc / Ts_sim;
    
    % Inicialización de matrices de registro
    SoC = zeros(Nt_sim, num_mg); V_tank = zeros(Nt_sim, num_mg);
    P_grid = zeros(Nt_sim, num_mg); Q_p = zeros(Nt_sim, num_mg);
    Q_DNO = zeros(Nt_sim, num_mg); P_pump = zeros(Nt_sim, num_mg);
    Q_t = zeros(Nt_sim, num_mg); V_aq = zeros(Nt_sim, 1); h_p = zeros(Nt_sim, num_mg);
    
    % Condiciones iniciales
    SoC(1, :) = 0.5; V_tank(1, :) = 20000;
    V_aq(1) = mg(1).V_aq_0; h_p(1, :) = mg(1).h_p0;
    
    % Variables para Zero-Order Hold de acciones del MPC
    P_mgref_k = zeros(1, num_mg); Q_p_k = zeros(1, num_mg);
    Q_buy_k = zeros(1, num_mg); Q_t_k = zeros(1, num_mg);
    s_pozo_k = zeros(1, num_mg); 
    
    % Historial de bombeo para el modelo de descenso ---
    Q_p_hist_mpc = zeros(ceil(Nt_sim / paso_mpc_en_sim), num_mg);
    
    %% --- 2. Lazo Principal de Simulación ---
    for k = 1:Nt_sim - 1
        
        if mod(k - 1, paso_mpc_en_sim) == 0
            k_mpc = (k - 1) / paso_mpc_en_sim + 1;
            fprintf('Ejecutando MPC Supervisor en t = %.1f horas... (k_mpc = %d)\n', (k-1)*Ts_sim/3600, k_mpc);
            
            % --- Ventana Deslizante para el historial ---
            datos_sim_sub.P_dem = submuestreo_max(P_dem(1:k, :), paso_mpc_en_sim);
            datos_sim_sub.P_gen = submuestreo_max(P_gen(1:k, :), paso_mpc_en_sim);
            datos_sim_sub.Q_dem = submuestreo_max(Q_dem(1:k, :), paso_mpc_en_sim);
            hist_completo.P_dem = [hist_arranque.P_dem; datos_sim_sub.P_dem];
            hist_completo.P_gen = [hist_arranque.P_gen; datos_sim_sub.P_gen];
            hist_completo.Q_dem = [hist_arranque.Q_dem; datos_sim_sub.Q_dem];
            
            hist_data.P_dem = hist_completo.P_dem(end - mg(1).max_lags_mpc + 1:end, :);
            hist_data.P_gen = hist_completo.P_gen(end - mg(1).max_lags_mpc + 1:end, :);
            hist_data.Q_dem = hist_completo.Q_dem(end - mg(1).max_lags_mpc + 1:end, :);
            
            [P_dem_pred, P_gen_pred, Q_dem_pred] = generar_predicciones_AR(mg(1).modelos, hist_data, mg(1).N);
            
            if k_mpc == 1
                q_p_hist_0 = zeros(1, num_mg);
            else
                q_p_hist_0 = Q_p_hist_mpc(k_mpc - 1, :);
            end
            
            [u_mpc] = controlador_mpc(mg, SoC(k,:), V_tank(k,:), V_aq(k), ...
                P_dem_pred, P_gen_pred, Q_dem_pred, q_p_hist_0, P_mgref_k, k_mpc, Q_p_hist_mpc);
            
            if ~isempty(u_mpc)
                P_mgref_k = u_mpc.P_mgref; 
                Q_p_k = u_mpc.Q_p;
                Q_buy_k = u_mpc.Q_buy; 
                Q_t_k = u_mpc.Q_t;
                % --- CORRECCIÓN --- Se captura el valor de s_pozo calculado por el MPC.
                s_pozo_k = u_mpc.s_pozo;
                Q_p_hist_mpc(k_mpc, :) = Q_p_k;
            else
                fprintf('MPC falló. Manteniendo la última acción.\n');
                if k_mpc > 1
                    Q_p_hist_mpc(k_mpc, :) = Q_p_hist_mpc(k_mpc-1, :);
                end
            end
        end
        
        % --- Simular dinámica con controladores locales ---
        for i = 1:num_mg
            % Capa Eléctrica
            P_pump_i = (mg(i).Mp * 9800 * Q_p_k(i) * (mg(i).h_ptub + mg(i).h_Tank_max)) / 1e6;
            P_pump(k, i) = P_pump_i;
            P_net_i = P_dem(k, i) + P_pump_i - P_gen(k, i);
            e_mg_i = P_mgref_k(i) - P_net_i;
            P_B_actual = sim_energia_local_tesis(mg(i), e_mg_i, SoC(k,i), Ts_sim);
            SoC(k+1, i) = SoC(k, i) - (P_B_actual * Ts_sim / 3600) / mg(i).E_batt_max;
            SoC(k+1, i) = min(max(SoC(k+1, i), mg(i).SoC_min + 1e-5), mg(i).SoC_max - 1e-5);
            P_grid(k, i) = P_net_i - P_B_actual;
            
            % Capa Hídrica
            Q_Tank_final = sim_agua_local_tesis(mg(i), V_tank(k,i), Q_p_k(i), Q_buy_k(i), Q_t_k(i), Q_dem(k, i));
            V_tank(k+1, i) = V_tank(k, i) + Q_Tank_final * Ts_sim;
            V_tank(k+1, i) = min(max(V_tank(k+1, i), 0), mg(i).V_max);
            Q_p(k, i) = Q_p_k(i); 
            Q_DNO(k, i) = Q_buy_k(i); 
            Q_t(k, i) = Q_t_k(i);
        end
        
        % --- Actualización de Recursos Compartidos ---
        recarga_total = (mg(1).Rp * (Ts_sim / 60));
        bombeo_total = sum(Q_p_k) * Ts_sim;
        V_aq(k+1) = max(V_aq(k) + recarga_total - bombeo_total, 0);
        
        % --- Se actualiza el estado h_p en cada paso de la simulación.
        % Se usa la Ecuación 4.15 de la tesis: h_p(t) = h_p(0) + s(t)
        h_p(k+1, :) = mg(1).h_p0 + s_pozo_k;
    end
end