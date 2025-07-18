function [V_next, Q_p, Q_DNO] = sim_agua(mg_params, Q_d, V_current, V_aquifero_k, Ts) 
    % Lógica de control reactiva para el subsistema de agua de UNA micro-red.
    
    % Extraer parámetros de la struct
    V_max = mg_params.V_max;
    Q_pump_max = mg_params.Mp * mg_params.Q_pump_max_unit;
    Q_DNO_max = mg_params.Q_DNO_max;

    % Demanda en litros para el paso de tiempo
    Q_req = Q_d * Ts; % [L]
    
    % Inicializar caudales
    Q_p   = 0; % caudal desde pozo [L/s]
    Q_DNO = 0; % caudal desde red [L/s]

    % === 1) Usar el estanque primero ===
    if V_current >= Q_req
        % El estanque tiene suficiente agua para cubrir la demanda
        V_next = V_current - Q_req;
    else
        % No alcanza -> comprar el déficit al DNO
        Q_deficit = (Q_req - V_current) / Ts; % [L/s]
        Q_DNO     = min(Q_deficit, Q_DNO_max);
        
        % Actualizar volumen: se vacía y se rellena con agua del DNO
        V_next = V_current - Q_req + Q_DNO * Ts;
    end
    
    % === 2) Bombear si el estanque queda bajo un umbral y hay agua en el acuífero ===
    if V_next < 0.1 * V_max && V_aquifero_k > 0
        % El agua máxima que puede extraer la bomba está limitada por el acuífero
        available_rate = V_aquifero_k / Ts; % [L/s] disponible
        Q_p = min(Q_pump_max, available_rate);
        
        % Llenar el estanque con el agua bombeada
        V_next = V_next + Q_p * Ts;
    end

    % === 3) Saturar el estanque para que se mantenga en sus límites físicos ===
    V_next = min(max(V_next, 0), V_max);
end