function [SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, h_p] = sim_microrred(mg, P_dem, P_gen, Q_dem, Ts, Tsim)
    % Simulador principal para un sistema de MÚLTIPLES micro-redes
    
    % Extraer parámetros comunes del primer struct
    V_aq_0 = mg(1).V_aq_0;
    h_p0 = mg(1).h_p0;
    T_aq = mg(1).T_aq;
    Rp = mg(1).Rp;
    W_u = 1; % Simplificación de la función de pozo
    
    num_mg = length(mg);
    Nt = Tsim / Ts;

    % Inicializar matrices de estado (cada columna es una micro-red)
    SoC = zeros(Nt, num_mg);
    V_tank = zeros(Nt, num_mg);
    P_grid = zeros(Nt, num_mg);
    Q_p = zeros(Nt, num_mg);
    Q_DNO = zeros(Nt, num_mg);
    P_pump = zeros(Nt, num_mg);
    h_p = zeros(Nt, num_mg);
    
    % El acuífero es único y compartido
    V_aq = zeros(Nt, 1);
    
    % Condiciones iniciales (Tesis Tabla 5.5)
    SoC(1, :)      = 0.5;         % 50% SoC para todas
    V_tank(1, :)   = 20000;       % 20,000 L para todas
    V_aq(1)        = V_aq_0;      % Volumen inicial del acuífero único
    h_p(1, :)      = h_p0;        % Profundidad inicial de cada pozo

    for k = 1:Nt-1
        % Vectores para almacenar resultados del paso k
        Q_p_k = zeros(1, num_mg);
        Q_DNO_k = zeros(1, num_mg);
        P_pump_k = zeros(1, num_mg);
        
        % === PASO 1: Simulación individual de cada micro-red ===
        for i = 1:num_mg
            [V_tank_temp, Q_p_k(i), Q_DNO_k(i)] = sim_agua(mg(i), Q_dem(k, i), V_tank(k, i), V_aq(k), Ts);
            V_tank(k+1, i) = V_tank_temp;
            
            [SoC_temp, P_grid(k, i), P_pump_k(i)] = sim_energia(mg(i), P_dem(k, i), P_gen(k, i), SoC(k, i), Ts, Q_p_k(i));
            SoC(k+1, i) = SoC_temp;
        end
        
        Q_p(k, :) = Q_p_k;
        Q_DNO(k, :) = Q_DNO_k;
        P_pump(k, :) = P_pump_k;

        % === PASO 2: Dinámica del acuífero (único y compartido) ===
        recarga_total = Rp * (Ts / 60); % Recarga en [L] para el paso Ts
        bombeo_total = sum(Q_p_k) * Ts; % Bombeo total de todas las MG
        
        V_aq(k+1) = max(V_aq(k) + recarga_total - bombeo_total, 0);
        
        % === PASO 3: Nivel en cada pozo (dinámica simplificada) ===
        for i = 1:num_mg
            s_k = (Q_p_k(i) / (4 * pi * T_aq)) * W_u;
            h_p(k+1, i) = h_p0 + s_k;
        end
    end
end