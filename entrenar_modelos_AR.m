% entrenar_modelos_AR.m
clear; clc; close all;
cd(fileparts(mfilename('fullpath')));
addpath('data');

fprintf('Iniciando el entrenamiento de modelos predictivos AR...\n\n');

%% --- 1. Carga y Preparación de Datos ---
% (Esta sección es idéntica a la versión anterior y ya funciona)
fprintf('Cargando y pre-procesando perfiles de entrada...\n');
d_mr1_raw = load('data/winter_30D.mat').winter_30D;
g_mr1_raw = load('data/pv_wint.mat').pv_wint * 22;
d_mr2_raw = load('data/winter_60D.mat').winter_60D;
g_mr2_raw = load('data/wind_inv.mat').wind_inv * 8.49;
d_mr3_raw = load('data/School_inv.mat').School_inv * 0.45;
h_mr1_raw = load('data/Dwellings30Water.mat').Dwellings30Water;
h_mr2_raw = load('data/Dwellings60Water.mat').Dwellings60Water;
h_mr3_raw = load('data/SchoolWater.mat').SchoolWater;

min_len = min([...
    length(d_mr1_raw), length(g_mr1_raw), length(d_mr2_raw), length(g_mr2_raw), ...
    length(d_mr3_raw), length(h_mr1_raw), length(h_mr2_raw), length(h_mr3_raw)
]);
fprintf('Estandarizando todos los perfiles a una longitud de %d muestras.\n', min_len);
d_mr1 = d_mr1_raw(1:min_len); g_mr1 = g_mr1_raw(1:min_len);
d_mr2 = d_mr2_raw(1:min_len); g_mr2 = g_mr2_raw(1:min_len);
d_mr3 = d_mr3_raw(1:min_len); h_mr1 = h_mr1_raw(1:min_len);
h_mr2 = h_mr2_raw(1:min_len); h_mr3 = h_mr3_raw(1:min_len);
g_mr3 = g_mr1 + g_mr2;

paso_mpc = 30;
P_dem_ts = [d_mr1(1:paso_mpc:end), d_mr2(1:paso_mpc:end), d_mr3(1:paso_mpc:end)];
P_gen_ts = [g_mr1(1:paso_mpc:end), g_mr2(1:paso_mpc:end), g_mr3(1:paso_mpc:end)];
Q_dem_ts = [h_mr1(1:paso_mpc:end), h_mr2(1:paso_mpc:end), h_mr3(1:paso_mpc:end)];

%% --- 2. Definición de Parámetros de Entrenamiento para Modelos AR ---
% Se usa el número de regresores definido en la tesis para los modelos AR.
params.P_dem.regresores = [48, 48, 46]; % Tabla 5.8
params.P_gen.regresores = [48, 45, 46]; % Tabla 5.11
params.Q_dem.regresores = [48, 48, 48]; % Tabla 5.16

%% --- 3. Entrenamiento Automatizado de los 9 Modelos AR ---
modelos = struct();
tipos_de_senal = {'P_dem', 'P_gen', 'Q_dem'};
datos_completos = {P_dem_ts, P_gen_ts, Q_dem_ts};

for i = 1:3 % Bucle para cada micro-red
    for j = 1:length(tipos_de_senal)
        tipo_actual = tipos_de_senal{j};
        datos_actuales = datos_completos{j};
        num_regresores = params.(tipo_actual).regresores(i);
        
        fprintf('Entrenando modelo AR para %s de Micro-red %d... (Regresores: %d)\n', ...
            tipo_actual, i, num_regresores);
            
        [train_inputs, train_outputs] = prepare_time_series_data(datos_actuales(:, i), num_regresores);
        
        % --- Lógica de entrenamiento AR usando Mínimos Cuadrados ---
        X = [ones(size(train_inputs, 1), 1), train_inputs]; % Añadir columna de unos para el término de offset
        % Resolver y = X*theta usando el operador de MATLAB (eficiente y robusto)
        theta = X \ train_outputs; 
        
        % Guardar los parámetros del modelo (vector theta)
        nombre_modelo = sprintf('mg%d_%s_ar', i, tipo_actual);
        modelos.(nombre_modelo).theta = theta;
        modelos.(nombre_modelo).num_regresores = num_regresores;
    end
    fprintf('\n');
end

%% --- 4. Guardar Todos los Modelos ---
if ~exist('models', 'dir'), mkdir('models'); end
save('models/modelos_prediccion_AR.mat', 'modelos');
fprintf('¡Entrenamiento completado! Modelos guardados en "models/modelos_prediccion_AR.mat"\n');

%% --- Función Auxiliar para Preparar Datos ---
function [inputs, outputs] = prepare_time_series_data(data, num_lags)
    inputs = zeros(length(data) - num_lags, num_lags);
    outputs = zeros(length(data) - num_lags, 1);
    for i = 1:length(data) - num_lags
        inputs(i, :) = data(i:i+num_lags-1)';
        outputs(i) = data(i+num_lags);
    end
end