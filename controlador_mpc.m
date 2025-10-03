function [u_opt] = controlador_mpc(mg, soC_0, v_tank_0, v_aq_0, p_dem_pred, p_gen_pred, q_dem_pred, q_p_hist_0, p_mgref_hist_0, k_mpc_actual, Q_p_hist_mpc)
    
    %% --- 1. Parámetros ---
    N = mg(1).N; Ts = mg(1).Ts_mpc; num_mg = length(mg);
    C_p = 110; C_q = 644; lambda_P = 1e-1; lambda_Q = 1e-1;
    costo_shed_P = 1e9; costo_shed_Q = 1e9; costo_slack_EAW = 1e9;
    costo_slack_s_pozo = 1e9; 
    
    % Parámetros del acuífero para el descenso del pozo ---
    S = mg(1).S_aq;      
    T = mg(1).T_aq;      
    r_p = mg(1).r_p;     
    s_max = mg(1).s_max;
    
    % Horizonte histórico para el cálculo del descenso 
    H_hist = 48; % Ventana de 24 horas (48 pasos de 30 min)
    
    %% --- 2. Variables de Optimización ---
    P_mgref = sdpvar(N, num_mg, 'full'); Q_p = sdpvar(N, num_mg, 'full');
    Q_buy   = sdpvar(N, num_mg, 'full'); Q_t = sdpvar(N, num_mg, 'full');
    P_B     = sdpvar(N, num_mg, 'full'); P_p = sdpvar(N, num_mg, 'full');
    SoC     = sdpvar(N+1, num_mg, 'full'); V_tank = sdpvar(N+1, num_mg, 'full');
    EAW     = sdpvar(N+1, 1, 'full');
    s_pozo  = sdpvar(N, num_mg, 'full'); 
    
    % Variables de Holgura (Slack)
    slack_EAW = sdpvar(1, 1, 'full'); P_shed = sdpvar(N, num_mg, 'full'); Q_shed = sdpvar(N, num_mg, 'full');
    slack_s_pozo = sdpvar(N, num_mg, 'full');
    
    %% --- 3. Restricciones y Objetivo ---
    constraints = []; objective = 0;
    
    constraints = [constraints, SoC(1,:) == soC_0, V_tank(1,:) == v_tank_0, EAW(1) == v_aq_0];
    constraints = [constraints, slack_EAW >= 0, P_shed >= 0, Q_shed >= 0, slack_s_pozo >= 0];
    
    delta_Q_p_futuro = sdpvar(N, num_mg, 'full');
    constraints = [constraints, delta_Q_p_futuro(1,:) == Q_p(1,:) - q_p_hist_0];
    for k = 2:N
        constraints = [constraints, delta_Q_p_futuro(k,:) == Q_p(k,:) - Q_p(k-1,:)];
    end
    
    for k = 1:N 
        
        for i = 1:num_mg
            descenso_sum = 0; 
            if k_mpc_actual > 1
                l_start = max(1, k_mpc_actual - H_hist); 
                for l = l_start:(k_mpc_actual - 1)
                    if l == 1
                        delta_Q_l = Q_p_hist_mpc(l, i);
                    else
                        delta_Q_l = Q_p_hist_mpc(l, i) - Q_p_hist_mpc(l-1, i);
                    end
                    tiempo_pasos = (k_mpc_actual - l) + k;
                    u_hist = (S * r_p^2) / (4 * T * tiempo_pasos * Ts);
                    descenso_sum = descenso_sum + (delta_Q_l / 1000) * well_function(u_hist);
                end
            end
            for l = 1:k
                 u_futuro = (S * r_p^2) / (4 * T * (k - l + 1) * Ts);
                 descenso_sum = descenso_sum + (delta_Q_p_futuro(l,i)/1000) * well_function(u_futuro);
            end
            
            constraints = [constraints, s_pozo(k,i) == (1 / (4*pi*T)) * descenso_sum];
            constraints = [constraints, 0 <= s_pozo(k,i) <= s_max + slack_s_pozo(k,i)]; 
        end
        
        for i = 1:num_mg
            constraints = [constraints, P_mgref(k,i) + P_shed(k,i) == p_dem_pred(k,i) - p_gen_pred(k,i) + P_p(k,i) - P_B(k,i)];
            
            demanda_real_agua = q_dem_pred(k,i) - Q_shed(k,i);
            Q_tank_net = Q_p(k,i) + Q_buy(k,i) - Q_t(k,i) - demanda_real_agua;
            constraints = [constraints, V_tank(k+1,i) == V_tank(k,i) + Q_tank_net * Ts];
            
            constraints = [constraints, P_p(k,i) == mg(i).Mp * 9800 * Q_p(k,i) * (mg(i).h_ptub + mg(i).h_Tank_max) / 1e6];
            constraints = [constraints, mg(i).E_batt_max * SoC(k+1,i) == mg(i).E_batt_max * SoC(k,i) - P_B(k,i) * (Ts/3600)];
            constraints = [constraints, mg(i).SoC_min <= SoC(k+1,i) <= mg(i).SoC_max];
            P_chg_max = mg(i).alpha_C * mg(i).P_batt_max * (1 - SoC(k,i));
            P_dis_max = mg(i).alpha_D * mg(i).P_batt_max * SoC(k,i);
            constraints = [constraints, -P_chg_max <= P_B(k,i) <= P_dis_max];
            constraints = [constraints, 0 <= V_tank(k+1,i) <= mg(i).V_max];
            constraints = [constraints, 0 <= Q_p(k,i) <= mg(i).Mp * mg(i).Q_pump_max_unit];
            constraints = [constraints, 0 <= Q_buy(k,i)];
            constraints = [constraints, mg(i).Q_t_min <= Q_t(k,i) <= mg(i).Q_t_max];
        end
        
        constraints = [constraints, sum(Q_t(k,:)) == 0];
        constraints = [constraints, 0 <= sum(P_mgref(k,:)) <= mg(1).P_grid_max];
        constraints = [constraints, 0 <= sum(Q_buy(k,:)) <= mg(1).Q_DNO_max];
        
        P_DNO_k = sum(P_mgref(k,:)); Q_DNO_k = sum(Q_buy(k,:));
        costo_energia = C_p * P_DNO_k * (Ts/3600); costo_agua = C_q * (Q_DNO_k * Ts / 1000);
        
        if k == 1
            delta_P = P_mgref(k,:) - p_mgref_hist_0;
            delta_Q = Q_p(k,:) - q_p_hist_0; 
        else
            delta_P = P_mgref(k,:) - P_mgref(k-1,:); 
            delta_Q = Q_p(k,:) - Q_p(k-1,:); 
        end
        costo_suavizado = lambda_P * sum(delta_P.^2) + lambda_Q * sum(delta_Q.^2);
        costo_penalizacion_P = costo_shed_P * sum(P_shed(k,:)) * (Ts/3600);
        costo_penalizacion_Q = costo_shed_Q * sum(Q_shed(k,:)) * Ts / 1000;
        
        objective = objective + costo_energia + costo_agua + costo_suavizado + costo_penalizacion_P + costo_penalizacion_Q;
    end
    
    recarga_por_paso = (mg(1).Rp * (Ts / 60)); recarga_vector = recarga_por_paso * ones(N, 1);
    bombeo_vector = sum(Q_p, 2) * Ts;
    constraints = [constraints, EAW(2:N+1) == EAW(1:N) + recarga_vector - bombeo_vector];
    constraints = [constraints, EAW(N+1) >= mg(1).V_aq_0 - slack_EAW];
    
    objective = objective + costo_slack_EAW * slack_EAW + costo_slack_s_pozo * sum(sum(slack_s_pozo));
    
    %% --- 4. Resolución ---
    options = sdpsettings('verbose', 2, 'debug', 1, 'solver', 'gurobi');
    sol = optimize(constraints, objective, options);
    
    if sol.problem == 0
        u_opt.P_mgref = value(P_mgref(1,:)); u_opt.Q_p = value(Q_p(1,:));
        u_opt.Q_buy = value(Q_buy(1,:)); u_opt.Q_t = value(Q_t(1,:));
        u_opt.s_pozo = value(s_pozo(1,:));
        if value(slack_EAW) > 1e-3, fprintf('ALERTA: Sostenibilidad del acuífero violada en %.2f L.\n', value(slack_EAW)); end
    else
        yalmip_error_text = yalmiperror(sol.problem);
        fprintf('Error YALMIP: %s\n', yalmip_error_text);
        u_opt = [];
    end
end