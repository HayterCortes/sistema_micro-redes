% --- main_mpc_AR_TS.m ---
clear; clc; close all;
% Establecer ruta de trabajo al directorio del script actual
cd(fileparts(mfilename('fullpath')));

%% ==================== CONFIGURACIÓN DE USUARIO ====================
% Seleccione el tipo de modelo predictivo a utilizar:
% 'AR' -> Modelos Auto-Regresivos Lineales
% 'TS' -> Modelos Difusos Takagi-Sugeno (No lineales)
TIPO_MODELO = 'TS'; 

fprintf('=== INICIANDO SIMULACIÓN EWMS (Modelo: %s) ===\n', TIPO_MODELO);

%% ==================== FASE 1: INICIALIZACIÓN ====================
addpath('models', 'utils');

% 1. Cargar configuración del sistema (parámetros físicos)
mg = configuracion_sistema(); 
fprintf('Parámetros del sistema cargados.\n');

% 2. Carga y preprocesamiento de datos (Entrenamiento/Validación/Simulación)
TRAIN_RATIO = 0.6; % 60% para entrenamiento
VAL_RATIO   = 0.2; % 20% para validación
% Se cargan los datos. P_dem_sim, etc., corresponden al set de PRUEBA (Test)
[P_dem_sim, P_gen_sim, Q_dem_sim, ~, ~, ~, P_dem_val, P_gen_val, Q_dem_val, ~, ~, ~] = cargar_y_preparar_datos(7, 30, TRAIN_RATIO, VAL_RATIO);

% 3. Verificación de existencia de modelos
if strcmp(TIPO_MODELO, 'AR')
    if ~isfile('models/modelos_prediccion_AR.mat')
        error('No se encontró el archivo models/modelos_prediccion_AR.mat. Ejecute entrenar_modelos_AR.m primero.');
    end
    fprintf('Modelos AR detectados correctamente.\n');
elseif strcmp(TIPO_MODELO, 'TS')
    if ~isfile('models/modelos_prediccion_TS.mat')
        error('No se encontró el archivo models/modelos_prediccion_TS.mat. Ejecute entrenar_modelos_TS.m primero.');
    end
    fprintf('Modelos TS detectados correctamente.\n');
else
    error('TIPO_MODELO desconocido. Use "AR" o "TS".');
end

% 4. Preparación del historial de arranque (Cold Start)
% Se necesitan datos previos para llenar el vector de regresores en t=1
fprintf('Preparando historial de arranque con %d pasos del set de validación.\n', mg(1).max_lags_mpc);
hist_arranque.P_dem = P_dem_val(end - mg(1).max_lags_mpc + 1:end, :);
hist_arranque.P_gen = P_gen_val(end - mg(1).max_lags_mpc + 1:end, :);
hist_arranque.Q_dem = Q_dem_val(end - mg(1).max_lags_mpc + 1:end, :);

% 5. Exportación auxiliar (opcional, para análisis externo)
if ~exist('utils', 'dir'), mkdir('utils'); end
save('utils/full_profiles_for_sim.mat', 'P_dem_sim', 'P_gen_sim', 'Q_dem_sim', 'hist_arranque');

%% ==================== FASE 2: SIMULACIÓN CON MPC ====================
fprintf('Iniciando bucle de simulación...\n');

% --- Inicialización de variables de simulación ---
Ts_mpc = mg(1).Ts_mpc; 
Ts_sim = mg(1).Ts_sim;
num_mg = length(mg); 
Nt_sim = size(P_dem_sim, 1);
paso_mpc_en_sim = Ts_mpc / Ts_sim; % Ratio de muestreo (30 min / 1 min = 30)

% Matrices de registro de resultados
SoC = zeros(Nt_sim, num_mg);      % Estado de carga
V_tank = zeros(Nt_sim, num_mg);   % Volumen estanques
P_grid = zeros(Nt_sim, num_mg);   % Potencia importada
Q_p = zeros(Nt_sim, num_mg);      % Caudal extracción
Q_DNO = zeros(Nt_sim, num_mg);    % Caudal comprado
P_pump = zeros(Nt_sim, num_mg);   % Potencia bombas
Q_t = zeros(Nt_sim, num_mg);      % Intercambio hídrico
V_aq = zeros(Nt_sim, 1);          % Volumen acuífero
h_p = zeros(Nt_sim, num_mg);      % Nivel pozo

% Condiciones iniciales físicas
SoC(1, :) = 0.5;          % Baterías al 50%
V_tank(1, :) = 20000;     % Estanques a media capacidad
V_aq(1) = mg(1).V_aq_0;   % Acuífero lleno inicialmente
h_p(1, :) = mg(1).h_p0;   % Nivel estático del pozo

% Variables de retención (Hold) para el MPC (Zero-Order Hold)
P_mgref_k = zeros(1, num_mg); 
Q_p_k = zeros(1, num_mg);
Q_buy_k = zeros(1, num_mg); 
Q_t_k = zeros(1, num_mg);
s_pozo_k = zeros(1, num_mg); 

% Historial de acciones pasadas para el cálculo del descenso del pozo (Theis)
% Se estima un tamaño máximo para pre-asignar memoria
Q_p_hist_mpc = zeros(ceil(Nt_sim / paso_mpc_en_sim), num_mg);

% --- BUCLE TEMPORAL (Minuto a Minuto) ---
for k = 1:Nt_sim - 1
    
    % --- A. EJECUCIÓN DEL MPC (Cada 30 minutos) ---
    if mod(k - 1, paso_mpc_en_sim) == 0
        k_mpc = (k - 1) / paso_mpc_en_sim + 1;
        
        % Imprimir siempre
        fprintf('Simulando... Tiempo: %.1f horas (k_mpc = %d)\n', (k-1)*Ts_sim/3600, k_mpc);
        
        % A.1 Actualización de datos históricos para predicción
        % Tomamos los datos ocurridos hasta 'k' y los submuestreamos
        datos_sim_sub.P_dem = submuestreo_max(P_dem_sim(1:k, :), paso_mpc_en_sim);
        datos_sim_sub.P_gen = submuestreo_max(P_gen_sim(1:k, :), paso_mpc_en_sim);
        datos_sim_sub.Q_dem = submuestreo_max(Q_dem_sim(1:k, :), paso_mpc_en_sim);
        
        % Concatenar con el arranque para tener ventana completa
        hist_completo.P_dem = [hist_arranque.P_dem; datos_sim_sub.P_dem];
        hist_completo.P_gen = [hist_arranque.P_gen; datos_sim_sub.P_gen];
        hist_completo.Q_dem = [hist_arranque.Q_dem; datos_sim_sub.Q_dem];
        
        % Extraer la ventana exacta necesaria para los regresores
        hist_data.P_dem = hist_completo.P_dem(end - mg(1).max_lags_mpc + 1:end, :);
        hist_data.P_gen = hist_completo.P_gen(end - mg(1).max_lags_mpc + 1:end, :);
        hist_data.Q_dem = hist_completo.Q_dem(end - mg(1).max_lags_mpc + 1:end, :);
        
        % A.2 Generación de Predicciones (SELECTOR DE MODELO)
        if strcmp(TIPO_MODELO, 'TS')
            [P_dem_pred, P_gen_pred, Q_dem_pred] = generar_predicciones_TS(hist_data, mg(1).N);
        else
            [P_dem_pred, P_gen_pred, Q_dem_pred] = generar_predicciones_AR(hist_data, mg(1).N);
        end
        
        % A.3 Preparación de estados previos para función de costo
        if k_mpc == 1
            q_p_hist_0 = zeros(1, num_mg);
            p_mgref_hist_0 = zeros(1, num_mg);
        else
            q_p_hist_0 = Q_p_hist_mpc(k_mpc - 1, :);
            p_mgref_hist_0 = P_mgref_k; 
        end
        
        % A.4 Llamada al Optimizador
        [u_mpc] = controlador_mpc(SoC(k,:), V_tank(k,:), V_aq(k), ...
            P_dem_pred, P_gen_pred, Q_dem_pred, q_p_hist_0, p_mgref_hist_0, k_mpc, Q_p_hist_mpc);
        
        % A.5 Procesamiento de la solución
        if ~isempty(u_mpc)
            % Actualizar referencias para el control local
            P_mgref_k = u_mpc.P_mgref; 
            Q_p_k     = u_mpc.Q_p;
            Q_buy_k   = u_mpc.Q_buy; 
            Q_t_k     = u_mpc.Q_t;
            s_pozo_k  = u_mpc.s_pozo;
            
            % Guardar historial para restricción de descenso de pozo futura
            Q_p_hist_mpc(k_mpc, :) = Q_p_k;
        else
            fprintf('ADVERTENCIA: MPC infactible en k_mpc=%d. Manteniendo acción anterior.\n', k_mpc);
            if k_mpc > 1
                Q_p_hist_mpc(k_mpc, :) = Q_p_hist_mpc(k_mpc-1, :);
            end
            % Nota: Si es el primer paso y falla, se mantienen los ceros iniciales.
        end
    end
    
    % --- B. CONTROL LOCAL Y DINÁMICA FÍSICA (Cada 1 minuto) ---
    for i = 1:num_mg
        % B.1 Capa Eléctrica (Control de Batería)
        % Calcular consumo de bomba real (física)
        % Potencia = cte * Caudal * Altura Dinámica
        % Altura = Profundidad Pozo + Altura Estanque
        % Nota: Q_p_k viene del MPC (L/s), se convierte a potencia
        P_pump_i = (mg(i).Mp * 9800 * Q_p_k(i) * (mg(i).h_ptub + mg(i).h_Tank_max)) / 1e6; % [kW]
        P_pump(k, i) = P_pump_i;
        
        % Balance de Potencia Neta Local
        P_net_i = P_dem_sim(k, i) + P_pump_i - P_gen_sim(k, i);
        
        % Error de seguimiento respecto a la referencia del MPC
        e_mg_i = P_mgref_k(i) - P_net_i;
        
        % Controlador Local de Batería
        P_B_actual = sim_energia_local_tesis(mg(i), e_mg_i, SoC(k,i), Ts_sim);
        
        % Dinámica de Batería (Integración)
        SoC(k+1, i) = SoC(k, i) - (P_B_actual * Ts_sim / 3600) / mg(i).E_batt_max;
        % Saturación física dura
        SoC(k+1, i) = min(max(SoC(k+1, i), mg(i).SoC_min + 1e-5), mg(i).SoC_max - 1e-5);
        
        % Potencia final intercambiada con la red
        P_grid(k, i) = P_net_i - P_B_actual;
        
        % B.2 Capa Hídrica (Control de Estanque)
        % Controlador Local de Tanque (Reglas de seguridad)
        Q_Tank_final = sim_agua_local_tesis(mg(i), V_tank(k,i), Q_p_k(i), Q_buy_k(i), Q_t_k(i), Q_dem_sim(k, i));
        
        % Dinámica de Estanque (Integración)
        V_tank(k+1, i) = V_tank(k, i) + Q_Tank_final * Ts_sim;
        V_tank(k+1, i) = min(max(V_tank(k+1, i), 0), mg(i).V_max);
        
        % Guardar flujos aplicados
        Q_p(k, i) = Q_p_k(i); 
        Q_DNO(k, i) = Q_buy_k(i); 
        Q_t(k, i) = Q_t_k(i);
    end
    
    % --- C. RECURSOS COMPARTIDOS (Acuífero) ---
    % Dinámica simplificada de volumen disponible (Balance de masas)
    recarga_total = (mg(1).Rp * (Ts_sim / 60)); % L/min * min = Litros
    bombeo_total = sum(Q_p_k) * Ts_sim;         % L/s * s = Litros
    
    V_aq(k+1) = max(V_aq(k) + recarga_total - bombeo_total, 0);
    
    % Actualización de la profundidad del pozo basada en el cálculo del MPC (Theis)
    % Nota: En una implementación real, esto se mediría con sensor. Aquí usamos
    % la estimación del MPC como "proxy" de la física compleja del suelo.
    h_p(k+1, :) = mg(1).h_p0 + s_pozo_k;
end

fprintf('Simulación completada exitosamente.\n');

%% ================= FASE 3: GUARDADO Y VISUALIZACIÓN =================
if ~exist('results_mpc', 'dir'), mkdir('results_mpc'); end

% Nombre dinámico del archivo según el modelo usado
nombre_archivo = sprintf('results_mpc/resultados_mpc_%s_3mg_7dias.mat', TIPO_MODELO);

save(nombre_archivo, ...
     'SoC', 'V_tank', 'P_grid', 'Q_p', 'Q_DNO', 'P_pump', 'V_aq', 'h_p', 'Q_t', 'mg', 'TIPO_MODELO');

fprintf('Resultados guardados en: %s\n', nombre_archivo);

% Generar gráficos comparativos
fprintf('Generando gráficos...\n');
plot_resultados_mpc(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Q_t, h_p);
fprintf('¡Proceso finalizado!\n');