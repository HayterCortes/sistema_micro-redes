function [SoC_next, P_grid, P_pump] = sim_energia(mg_params, P_d, P_g, SoC, Ts, Q_p)
    % Lógica de control reactiva para el subsistema de energía de UNA micro-red.
    
    % Extraer parámetros de la struct
    E_batt_max = mg_params.E_batt_max;
    P_batt_max = mg_params.P_batt_max;
    SoC_min = mg_params.SoC_min;
    SoC_max = mg_params.SoC_max;
    P_grid_max = mg_params.P_grid_max;
    alpha_C = mg_params.alpha_C;
    alpha_D = mg_params.alpha_D;
    Mp = mg_params.Mp;

    % Calcular potencia de la bomba (Nexo Agua-Energía)
    h_equiv = 13.6508; % Altura [m] (pozo + estanque)
    B = 9800;          % Constante rho*g [N/m³]
    P_pump = (Mp * B * Q_p * h_equiv) / 1000;  % [kW]

    % Balance de potencia
    P_total = P_d + P_pump;
    P_net = P_g - P_total;
    
    P_grid = 0;

    % Límites dinámicos de la batería según tesis (4.6 y 4.7)
    P_chg_max = alpha_C * P_batt_max * (1 - SoC);
    P_dis_max = alpha_D * P_batt_max * SoC;
    
    if P_net >= 0
        % EXCESO de generación -> Cargar batería
        P_charge = min(P_net, P_chg_max);
        SoC_next = SoC + (P_charge * Ts / 3600) / E_batt_max;
    else
        % DÉFICIT de generación -> Descargar batería
        P_deficit = abs(P_net);
        P_discharge = min(P_deficit, P_dis_max);
        SoC_temp = SoC - (P_discharge * Ts / 3600) / E_batt_max;
        
        if SoC_temp < SoC_min
            % Batería no puede cubrir todo el déficit -> usar lo que queda y comprar de la red
            E_disponible = (SoC - SoC_min) * E_batt_max; % kWh
            P_batt_eff = E_disponible * 3600 / Ts; % kW
            P_grid = P_deficit - P_batt_eff;
            P_grid = min(max(P_grid, 0), P_grid_max); % Limitar compra de la red
            SoC_next = SoC_min;
        else
            % La batería cubre el déficit (parcial o totalmente) sin problema
            SoC_next = SoC_temp;
        end
    end

    % Saturar SoC para que se mantenga en sus límites
    SoC_next = min(max(SoC_next, SoC_min), SoC_max);
end