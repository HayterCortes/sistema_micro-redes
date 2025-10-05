% --- main_mpc.m (Versión Final Corregida) ---
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

% --- Parámetros del MPC ---
% --- ELIMINADO ---
% Las siguientes líneas se han movido a 'configuracion_sistema.m' para centralizar
% todos los parámetros y evitar errores.
% mg(1).Ts_mpc = 30 * 60;
% mg(1).N      = 48;
% mg(1).Ts_sim = 60;

% --- Cargar modelos predictivos ---
load('models/modelos_prediccion_AR.mat');
fprintf('Modelos predictivos AR cargados.\n');
mg(1).modelos = modelos;

% --- Parámetros de pozo y acuífero ---
% --- ELIMINADO --- 
% Estos parámetros ahora también se cargan desde 'configuracion_sistema.m'
% mg(1).S_aq = 0.1906;        
% mg(1).T_aq = 35.1062 / (24*60*60); 
% mg(1).r_p = 0.2;            
% mg(1).s_max = 8;            
% fprintf('Parámetros de acuífero y pozo cargados para el modelo de descenso.\n');

% --- Preparar el historial de arranque desde el set de validación ---
% --- ELIMINADO ---
% El parámetro 'max_lags' ahora está en la configuración.
% max_lags = 48; 
% mg(1).max_lags_mpc = max_lags;
fprintf('Preparando historial de arranque con %d pasos del set de validación.\n', mg(1).max_lags_mpc);
hist_arranque.P_dem = P_dem_val(end - mg(1).max_lags_mpc + 1:end, :);
hist_arranque.P_gen = P_gen_val(end - mg(1).max_lags_mpc + 1:end, :);
hist_arranque.Q_dem = Q_dem_val(end - mg(1).max_lags_mpc + 1:end, :);

% --- MANTENIDO: Guardar datos para el explicador de Python ---
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