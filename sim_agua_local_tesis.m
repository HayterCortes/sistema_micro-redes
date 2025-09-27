% sim_agua_local_tesis.m
function [Q_Tank_final] = sim_agua_local_tesis(mg_params, V_Tank_actual, Q_p_mpc, Q_buy_mpc, Q_t_mpc, Q_L_medido)
    % Controlador local de caudal basado en las reglas de la Sección 4.4
    % de la tesis de Jiménez, L. (2024). 
    % Su función es validar las acciones del MPC contra los límites físicos
    % del estanque.

    % Extraer parámetros
    V_Tank_max = mg_params.V_max;
    
    % Calcular el caudal neto PREVIO según las órdenes del MPC (Ecuación 4.72) 
    Q_Tank_previo = Q_p_mpc + Q_buy_mpc - Q_t_mpc - Q_L_medido;

    % Aplicar el set de reglas de seguridad del estanque (Ecuación 4.73) 
    if Q_Tank_previo >= 0 && V_Tank_actual >= V_Tank_max
        % R1: Si se intenta llenar un estanque que ya está lleno, se anula el flujo.
        Q_Tank_final = 0;
    elseif Q_Tank_previo < 0 && V_Tank_actual <= 0 % Usamos V_tank <= 0 por robustez
        % R4: Si se intenta vaciar un estanque que ya está vacío, se anula el flujo.
        Q_Tank_final = 0;
    else
        % R2 y R3: Si no se violan los límites, la acción del MPC es válida.
        Q_Tank_final = Q_Tank_previo;
    end
end