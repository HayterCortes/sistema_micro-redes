function mg = configuracion_sistema()
    % Esta función configura los parámetros para las 3 micro-redes (mg)
    % y los devuelve en un arreglo de structs.

    %% ==================== PARÁMETROS COMUNES ====================
    % Parámetros del acuífero (desde la tesis, Tabla 5.6)
    V_aq_0 = 1e6;             % Volumen inicial total del acuífero [L]
    S_aq   = 0.1906;          % Coef. de almacenamiento
    T_aq   = 35.1062 / 86400; % Transmisividad [m²/s]
    rp     = 0.2;             % Radio del pozo [m]
    h_p0   = 2;               % Profundidad inicial del pozo [m]
    
    % Recarga natural del acuífero
    % Tesis: 10,000 L diarios para todo el sistema
    Rp = 10000 / (24 * 60);   % Tasa de recarga total [L/min]

    %% ==================== MICRO-RED 1: 30 VIVIENDAS ====================
    mg(1).nombre = '30 Viviendas';
    % --- Parámetros de Energía (Tabla 5.3) ---
    mg(1).E_batt_max = 136;
    mg(1).P_batt_max = 70.9565;
    mg(1).alpha_C    = 1;
    mg(1).alpha_D    = 1;
    mg(1).SoC_min    = 0.2;
    mg(1).SoC_max    = 0.8;
    % --- Parámetros de Agua (Tabla 5.4) ---
    mg(1).V_max      = 40000;
    mg(1).Mp         = 30; % Multiplicador de bombas
    mg(1).Q_pump_max_unit = 30; % L/s por unidad de bomba
    
    %% ==================== MICRO-RED 2: 60 VIVIENDAS ====================
    mg(2).nombre = '60 Viviendas';
    % --- Parámetros de Energía (Tabla 5.3) ---
    mg(2).E_batt_max = 180;
    mg(2).P_batt_max = 93.9130;
    mg(2).alpha_C    = 1;
    mg(2).alpha_D    = 1;
    mg(2).SoC_min    = 0.2;
    mg(2).SoC_max    = 0.8;
    % --- Parámetros de Agua (Tabla 5.4) ---
    mg(2).V_max      = 40000;
    mg(2).Mp         = 60;
    mg(2).Q_pump_max_unit = 30;

    %% ==================== MICRO-RED 3: ESCUELA ====================
    mg(3).nombre = 'Escuela';
    % --- Parámetros de Energía (Tabla 5.3) ---
    mg(3).E_batt_max = 248;
    mg(3).P_batt_max = 129.3913;
    mg(3).alpha_C    = 1;
    mg(3).alpha_D    = 1;
    mg(3).SoC_min    = 0.2;
    mg(3).SoC_max    = 0.8;
    % --- Parámetros de Agua (Tabla 5.4) ---
    mg(3).V_max      = 40000;
    mg(3).Mp         = 45;
    mg(3).Q_pump_max_unit = 30;
    
    %% =========== ASIGNAR PARÁMETROS GLOBALES DEL SISTEMA A CADA MG ===========
    for i = 1:length(mg)
        mg(i).V_aq_0 = V_aq_0;
        mg(i).S_aq = S_aq;
        mg(i).T_aq = T_aq;
        mg(i).rp = rp;
        mg(i).h_p0 = h_p0;
        mg(i).Rp = Rp;
        mg(i).P_grid_max = 500;
        mg(i).Q_DNO_max  = 1e6;
        mg(i).h_ptub = 10;           % Profundidad tubería [m] (Tabla 5.4)
        mg(i).h_Tank_max = 3.6508;   % Altura máx. estanque [m] (Tabla 5.4)
        mg(i).Q_t_min = -100;        % Caudal mín. de intercambio [L/s] (Tabla 5.4)
        mg(i).Q_t_max = 100;         % Caudal máx. de intercambio [L/s] (Tabla 5.4)
    end
end