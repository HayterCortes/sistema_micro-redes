% main_mpc.m
clear; clc; close all;
% Establecer ruta de trabajo
cd(fileparts(mfilename('fullpath')));

%% ==================== FASE 1: CONFIGURACIÓN ====================
addpath('data', 'models', 'utils');
mg = configuracion_sistema();
fprintf('Parámetros del sistema cargados.\n');

% --- Carga y preprocesamiento de datos ---
fprintf('Cargando y pre-procesando perfiles de entrada...\n');

% Carga de datos crudos
d_mr1_raw = load('data/winter_30D.mat').winter_30D;
g_mr1_raw = load('data/pv_wint.mat').pv_wint * 22;
d_mr2_raw = load('data/winter_60D.mat').winter_60D;
g_mr2_raw = load('data/wind_inv.mat').wind_inv * 8.49;
d_mr3_raw = load('data/School_inv.mat').School_inv * 0.45;
h_mr1_raw = load('data/Dwellings30Water.mat').Dwellings30Water;
h_mr2_raw = load('data/Dwellings60Water.mat').Dwellings60Water;
h_mr3_raw = load('data/SchoolWater.mat').SchoolWater;

% --- Lógica de estandarización robusta ---
% 1. Encontrar la longitud del perfil más corto de todos los archivos
min_len = min([...
    length(d_mr1_raw), length(g_mr1_raw), length(d_mr2_raw), length(g_mr2_raw), ...
    length(d_mr3_raw), length(h_mr1_raw), length(h_mr2_raw), length(h_mr3_raw)
]);

% 2. Truncar todos los perfiles a esa longitud mínima para consistencia
d_mr1 = d_mr1_raw(1:min_len);
g_mr1 = g_mr1_raw(1:min_len);
d_mr2 = d_mr2_raw(1:min_len);
g_mr2 = g_mr2_raw(1:min_len);
d_mr3 = d_mr3_raw(1:min_len);
h_mr1 = h_mr1_raw(1:min_len);
h_mr2 = h_mr2_raw(1:min_len);
h_mr3 = h_mr3_raw(1:min_len);

% 3. AHORA, crear el perfil híbrido usando los vectores ya consistentes
g_mr3 = g_mr1 + g_mr2;

% 4. Finalmente, tomar las muestras para los 7 días de simulación
muestras_7_dias = 7 * 24 * 60;
P_dem = [d_mr1(1:muestras_7_dias), d_mr2(1:muestras_7_dias), d_mr3(1:muestras_7_dias)];
P_gen = [g_mr1(1:muestras_7_dias), g_mr2(1:muestras_7_dias), g_mr3(1:muestras_7_dias)];
Q_dem = [h_mr1(1:muestras_7_dias), h_mr2(1:muestras_7_dias), h_mr3(1:muestras_7_dias)];
fprintf('Perfiles de entrada listos para 7 días.\n');

% --- Definir parámetros del MPC ---
mg(1).Ts_mpc = 30 * 60;
mg(1).N      = 48;
mg(1).Ts_sim = 60;

% --- Cargar modelos predictivos ---
load('models/modelos_prediccion_AR.mat');
fprintf('Modelos predictivos AR cargados.\n');
mg(1).modelos = modelos;

% --- Calcular período de arranque para el MPC ---
num_reg_Pdem = [48, 48, 46];
num_reg_Pgen = [48, 45, 46];
num_reg_Qdem = [48, 48, 48];
max_lags = max([num_reg_Pdem, num_reg_Pgen, num_reg_Qdem]);
mg(1).max_lags_mpc = max_lags;
fprintf('Período de arranque requerido: %d pasos de 30 min.\n', max_lags);

%% ==================== FASE 2: SIMULACIÓN CON MPC ====================
fprintf('Iniciando simulación con MPC para 3 micro-redes...\n');

[SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, h_p, Q_t] = ...
    sim_microrred_mpc(mg, P_dem, P_gen, Q_dem);

fprintf('Simulación con MPC completada.\n');

%% ================= FASE 3: GUARDADO Y VISUALIZACIÓN =================
if ~exist('results_mpc', 'dir')
   mkdir('results_mpc');
end
save('results_mpc/resultados_mpc_3mg_7dias.mat', ...
     'SoC', 'V_tank', 'P_grid', 'Q_p', 'Q_DNO', 'P_pump' ,'V_aq', 'h_p', 'Q_t', 'mg');
fprintf('Resultados guardados en results_mpc/resultados_mpc_3mg_7dias.mat\n');

plot_resultados_mpc(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Q_t);
fprintf('Gráficos generados y guardados en la carpeta results_mpc.\n');