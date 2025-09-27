% sim_energia_local_tesis.m
function [P_B_actual] = sim_energia_local_tesis(mg_params, e_mg, SoC, Ts)
    % Controlador local de potencia basado en reglas, como se describe en la
    % Sección 4.4 de la tesis de Jiménez, L. (2024). 
    % Su objetivo es calcular la potencia de la batería (P_B_actual) para
    % minimizar el error de seguimiento de potencia e_mg.
    
    % Extraer parámetros de la struct
    P_batt_max = mg_params.P_batt_max;
    SoC_min = mg_params.SoC_min;
    SoC_max = mg_params.SoC_max;
    alpha_C = mg_params.alpha_C;
    alpha_D = mg_params.alpha_D;

    % Calcular límites dinámicos REALES de la batería (Ecuaciones 4.6 y 4.7) 
    P_chg_max = alpha_C * P_batt_max * (1 - SoC);
    P_dis_max = alpha_D * P_batt_max * SoC;

    % Lógica de control basada en reglas 
    % El objetivo es que la batería compense el error: P_B_target = -e_mg
    P_B_target = -e_mg; 

    % Reglas basadas en el estado de carga (SoC)
    if P_B_target < 0 % Se necesita CARGAR la batería (absorber potencia)
        if SoC >= SoC_max
            P_B_target = 0; % No se puede cargar más
        end
    elseif P_B_target > 0 % Se necesita DESCARGAR la batería (inyectar potencia)
        if SoC <= SoC_min
            P_B_target = 0; % No se puede descargar más
        end
    end
    
    % La acción de control final es el objetivo, pero siempre saturado por
    % los límites físicos instantáneos de la batería.
    P_B_actual = max(-P_chg_max, min(P_B_target, P_dis_max));
end