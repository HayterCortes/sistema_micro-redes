clear; clc; close all;
% Establecer ruta de trabajo a la carpeta del script
cd(fileparts(mfilename('fullpath')));

%% ==================== FASE 1: CONFIGURACIÓN ====================
% Cargar parámetros para las 3 micro-redes en la struct 'mg'
addpath('data', 'models', 'utils');
mg = configuracion_sistema();
fprintf('Parámetros del sistema cargados.\n');

% --- Carga y preprocesamiento de datos según Readme ---
fprintf('Cargando y pre-procesando perfiles de entrada...\n');

% --- Carga de datos crudos ---
d_mr1 = load('data/winter_30D.mat');
g_mr1 = load('data/pv_wint.mat');
h_mr1 = load('data/Dwellings30Water.mat');

d_mr2 = load('data/winter_60D.mat');
g_mr2 = load('data/wind_inv.mat');
h_mr2 = load('data/Dwellings60Water.mat');

d_mr3 = load('data/School_inv.mat');
h_mr3 = load('data/SchoolWater.mat');

% --- MODIFICACIÓN: Definir duración fija de 7 días ---
muestras_7_dias = 7 * 24 * 60; % 7 días * 24 horas/día * 60 minutos/hora

% Verificación de que los datos son suficientemente largos
Nt_min = min([...
    length(d_mr1.winter_30D), length(g_mr1.pv_wint), length(h_mr1.Dwellings30Water), ...
    length(d_mr2.winter_60D), length(g_mr2.wind_inv), length(h_mr2.Dwellings60Water), ...
    length(d_mr3.School_inv), length(h_mr3.SchoolWater)]);

if Nt_min < muestras_7_dias
    error('Los archivos de datos no contienen suficientes muestras para una simulación de 7 días.');
end
fprintf('Ajustando todos los perfiles a una duración de 7 días (%d muestras).\n', muestras_7_dias);

% Pre-alocar matrices con el tamaño correcto
P_dem = zeros(muestras_7_dias, 3);
P_gen = zeros(muestras_7_dias, 3);
Q_dem = zeros(muestras_7_dias, 3);

% --- Asignación y escalado de datos (ahora para 7 días) ---
% Micro-red 1
P_dem(:,1) = d_mr1.winter_30D(1:muestras_7_dias) * 1;
P_gen(:,1) = g_mr1.pv_wint(1:muestras_7_dias) * 22;
Q_dem(:,1) = h_mr1.Dwellings30Water(1:muestras_7_dias) * 1;

% Micro-red 2
P_dem(:,2) = d_mr2.winter_60D(1:muestras_7_dias) * 1;
P_gen(:,2) = g_mr2.wind_inv(1:muestras_7_dias) * 8.49;
Q_dem(:,2) = h_mr2.Dwellings60Water(1:muestras_7_dias) * 1;

% Micro-red 3
P_dem(:,3) = d_mr3.School_inv(1:muestras_7_dias) * 0.45;
P_gen(:,3) = P_gen(1:muestras_7_dias, 1) + P_gen(1:muestras_7_dias, 2); % Perfil híbrido
Q_dem(:,3) = h_mr3.SchoolWater(1:muestras_7_dias) * 1;

fprintf('Perfiles de entrada listos.\n');

% Configuración de tiempo
Ts = 60;                % Paso de simulación [s]
Tsim = (muestras_7_dias - 1) * Ts; % Duración basada en las muestras de 7 días
fprintf('Duración de la simulación: %.1f horas.\n', Tsim / 3600);

%% ==================== FASE 2: SIMULACIÓN ====================
fprintf('Iniciando simulación para 3 micro-redes...\n');
[SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, h_p] = ...
    sim_microrred(mg, P_dem, P_gen, Q_dem, Ts, Tsim);
fprintf('Simulación completada.\n');

%% ================= FASE 3: GUARDADO Y VISUALIZACIÓN =================
if ~exist('results', 'dir')
   mkdir('results');
end

% Guardar resultados en un único archivo
save('results/resultados_3mg_7dias.mat', ...
     'SoC', 'V_tank', 'P_grid', 'Q_p', 'Q_DNO', 'P_pump' ,'V_aq', 'h_p', 'Ts', 'mg');
fprintf('Resultados guardados en results/resultados_3mg_7dias.mat\n');
 
% Graficar resultados
plot_resultados(mg, SoC, V_tank, P_grid, Q_p, Q_DNO, P_pump, V_aq, Ts);
fprintf('Gráficos generados y guardados en la carpeta results.\n');