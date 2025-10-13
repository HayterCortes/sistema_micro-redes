% --- main_mpc.m ---
clear; clc; close all;
% Establecer ruta de trabajo
cd(fileparts(mfilename('fullpath')));

%% ==================== FASE 1: CONFIGURACIÓN ====================
addpath('models', 'utils');

% Carga la configuración completa, incluyendo ahora los parámetros del MPC.
mg = configuracion_sistema(); 
fprintf('Parámetros del sistema cargados.\n');

% --- Carga y preprocesamiento de datos ---
TRAIN_RATIO = 0.6; % 60% para entrenamiento
VAL_RATIO   = 0.2; % 20% para validación
[P_dem_sim, P_gen_sim, Q_dem_sim, ~, ~, ~, P_dem_val, P_gen_val, Q_dem_val, ~, ~, ~] = cargar_y_preparar_datos(7, 30, TRAIN_RATIO, VAL_RATIO);

% --- Cargar modelos predictivos ---
load('models/modelos_prediccion_AR.mat');
fprintf('Modelos predictivos AR cargados.\n');
mg(1).modelos = modelos;

fprintf('Preparando historial de arranque con %d pasos del set de validación.\n', mg(1).max_lags_mpc);
hist_arranque.P_dem = P_dem_val(end - mg(1).max_lags_mpc + 1:end, :);
hist_arranque.P_gen = P_gen_val(end - mg(1).max_lags_mpc + 1:end, :);
hist_arranque.Q_dem = Q_dem_val(end - mg(1).max_lags_mpc + 1:end, :);

% --- Guardar datos para el explicador de Python ---
fprintf('--- Exportando perfiles de simulación para Python LIME ---\n');
if ~exist('utils', 'dir'), mkdir('utils'); end
save('utils/full_profiles_for_sim.mat', 'P_dem_sim', 'P_gen_sim', 'Q_dem_sim', 'hist_arranque');
fprintf('Datos para el explicador de Python guardados en "utils/full_profiles_for_sim.mat".\n\n');

%% ==================== FASE 2: SIMULACIÓN CON MPC ====================
fprintf('Iniciando simulación con MPC para 3 micro-redes...\n');
[SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, h_p, Q_t] = ...
    sim_microrred_mpc(mg, P_dem_sim, P_gen_sim, Q_dem_sim, hist_arranque);
fprintf('Simulación con MPC completada.\n');

%% ================= FASE 3: GUARDADO Y VISUALIZACIÓN =================
if ~exist('results_mpc', 'dir'), mkdir('results_mpc'); end
save('results_mpc/resultados_mpc_3mg_7dias.mat', ...
     'SoC', 'V_tank', 'P_grid', 'Q_p', 'Q_DNO', 'P_pump' ,'V_aq', 'h_p', 'Q_t', 'mg');
fprintf('Resultados guardados en results_mpc/resultados_mpc_3mg_7dias.mat\n');
plot_resultados_mpc(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Q_t, h_p);
fprintf('Gráficos generados y guardados en la carpeta results_mpc.\n');