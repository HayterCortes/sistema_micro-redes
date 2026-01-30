% --- entrenar_modelos_TS.m ---
% Versión: Búsqueda Exhaustiva (Alta Fidelidad a la Tesis)
clear; clc; close all;
cd(fileparts(mfilename('fullpath')));
addpath('models', 'utils');

% Fijar semilla para reproducibilidad del Clustering Difuso (FCM)
rng(42); 

% Cargar funciones del núcleo TS
ts_funcs = ts_core_functions();

fprintf('Iniciando entrenamiento de modelos Difusos Takagi-Sugeno (TS)...\n');
fprintf('MODO: Búsqueda Exhaustiva (1 a 48 regresores). Esto puede tomar unos minutos.\n\n');

%% --- 1. Preparación de Datos ---
TRAIN_RATIO = 0.6; VAL_RATIO = 0.2;
% Carga de datos
[~,~,~, P_dem_train, P_gen_train, Q_dem_train, P_dem_val, P_gen_val, Q_dem_val, P_dem_test, P_gen_test, Q_dem_test] = cargar_y_preparar_datos(7, 30, TRAIN_RATIO, VAL_RATIO);

%% --- 2. BÚSQUEDA DE HIPERPARÁMETROS (Regresores y Reglas) ---
% Configuración basada estrictamente en la Tesis (Sección 5.2.2)
MAX_REGRESORES = 48; % Horizonte de 24 horas (pasos de 30 min)
LAGS_A_PROBAR = 1:MAX_REGRESORES; % Búsqueda paso a paso para encontrar óptimos locales (ej. 11, 43)
RANGO_CLUSTERS = 2:5; % Número de reglas difusas a probar

params_optimos = struct();
tipos_de_senal = {'P_dem', 'P_gen', 'Q_dem'};
datos_entrenamiento = {P_dem_train, P_gen_train, Q_dem_train};
datos_validacion = {P_dem_val, P_gen_val, Q_dem_val};

for i = 1:3 % Iterar sobre cada Micro-red
    for j = 1:length(tipos_de_senal)
        tipo = tipos_de_senal{j};
        d_train = datos_entrenamiento{j}(:, i);
        d_val = datos_validacion{j}(:, i);
        
        fprintf('Optimizando estructura para MG %d - %s... ', i, tipo);
        
        best_rmse = inf;
        best_struct = struct('lags', 1, 'rules', 2);
        
        % Bucle exhaustivo sobre número de regresores (p)
        for p = LAGS_A_PROBAR 
            % Preparar matriz de regresores para entrenamiento
            [X_t, y_t] = prepare_time_series_data(d_train, p);
            
            % Preparar matriz de regresores para validación
            % Se concatena el final del train para tener historia suficiente
            hist_val = [d_train(end-p+1:end); d_val];
            [X_v, y_v] = prepare_time_series_data(hist_val, p);
            
            % Bucle sobre número de reglas/clusters (c)
            for c = RANGO_CLUSTERS
                try
                    % 1. Entrenar candidato (FCM + WLS)
                    model_cand = ts_funcs.train(X_t, y_t, c);
                    
                    % 2. Evaluar en conjunto de validación
                    y_pred_v = ts_funcs.eval(model_cand, X_v);
                    
                    % Calcular error (RMSE)
                    rmse_val = sqrt(mean((y_v - y_pred_v).^2));
                    
                    % 3. Guardar si es el mejor hasta ahora
                    if rmse_val < best_rmse
                        best_rmse = rmse_val;
                        best_struct.lags = p;
                        best_struct.rules = c;
                    end
                catch
                    % En caso de error numérico en FCM (raro), continuar
                    continue;
                end
            end
        end
        
        % Guardar la mejor estructura encontrada para esta señal/MG
        params_optimos.(tipo).struct(i) = best_struct;
        
        % Reportar resultado
        fprintf('-> Óptimo encontrado: %d Regresores, %d Reglas (RMSE Val: %.4f)\n', ...
            best_struct.lags, best_struct.rules, best_rmse);
    end
    fprintf('\n'); % Separador visual entre micro-redes
end

%% --- 3. ENTRENAMIENTO FINAL Y GUARDADO ---
fprintf('--- Entrenando modelos finales TS con las estructuras óptimas ---\n');
modelos_ts = struct();

% Para el modelo final que se usará en el MPC, es práctica común re-entrenar
% usando los datos de Entrenamiento + Validación combinados para maximizar
% la información capturada.
datos_full = { [P_dem_train; P_dem_val], [P_gen_train; P_gen_val], [Q_dem_train; Q_dem_val] };

for i = 1:3
    for j = 1:length(tipos_de_senal)
        tipo = tipos_de_senal{j};
        d_full = datos_full{j}(:, i);
        
        % Recuperar estructura óptima
        s = params_optimos.(tipo).struct(i);
        
        % Preparar datos completos
        [X, y] = prepare_time_series_data(d_full, s.lags);
        
        % Entrenar modelo definitivo
        final_model = ts_funcs.train(X, y, s.rules);
        
        % Guardar metadatos críticos para la predicción
        final_model.num_regresores = s.lags; 
        
        % Almacenar en estructura global
        nombre = sprintf('mg%d_%s_ts', i, tipo);
        modelos_ts.(nombre) = final_model;
    end
end

% Guardar archivo .mat
if ~exist('models', 'dir'), mkdir('models'); end
save('models/modelos_prediccion_TS.mat', 'modelos_ts');
fprintf('Modelos TS guardados exitosamente en "models/modelos_prediccion_TS.mat".\n');
fprintf('Listo para ejecutar main_mpc.m\n');

%% --- Función auxiliar local ---
function [inputs, outputs] = prepare_time_series_data(data, num_lags)
    % Transforma una serie de tiempo en matriz de regresores X y objetivo Y
    if length(data) <= num_lags
        inputs = []; outputs = []; return;
    end
    n_samples = length(data) - num_lags;
    inputs = zeros(n_samples, num_lags);
    
    % Construcción vectorizada de la matriz de Hankel/Regresores
    for k = 1:num_lags
        inputs(:, k) = data(k : k + n_samples - 1);
    end
    outputs = data(num_lags + 1 : end);
end