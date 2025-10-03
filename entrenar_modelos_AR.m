% entrenar_modelos_AR.m
clear; clc; close all;
cd(fileparts(mfilename('fullpath')));

fprintf('Iniciando el pipeline de entrenamiento y validación de modelos AR...\n\n');

%% --- 1. Carga, Preparación y División de Datos ---
TRAIN_RATIO = 0.6; % 60% para entrenamiento
VAL_RATIO   = 0.2; % 20% para validación
% El 20% restante es para prueba
[~,~,~, P_dem_train, P_gen_train, Q_dem_train, P_dem_val, P_gen_val, Q_dem_val, P_dem_test, P_gen_test, Q_dem_test] = cargar_y_preparar_datos(7, 30, TRAIN_RATIO, VAL_RATIO);
paso_mpc = 30;

%% --- 2. BÚSQUEDA DE HIPERPARÁMETROS (N° ÓPTIMO DE REGRESORES) ---
fprintf('--- Iniciando búsqueda de regresores óptimos ---\n');
MAX_REGRESORES = 48; % Equivalente a 24 horas
params_optimos = struct();

tipos_de_senal = {'P_dem', 'P_gen', 'Q_dem'};
datos_entrenamiento = {P_dem_train, P_gen_train, Q_dem_train};
datos_validacion = {P_dem_val, P_gen_val, Q_dem_val};

for i = 1:3 % Para cada micro-red
    for j = 1:length(tipos_de_senal) % Para cada tipo de señal
        tipo_actual = tipos_de_senal{j};
        datos_train = datos_entrenamiento{j}(:, i);
        datos_val = datos_validacion{j}(:, i);
        
        fprintf('Validando para %s de Micro-red %d...\n', tipo_actual, i);
        
        errores_val = [];
        for p = 1:MAX_REGRESORES % Probar cada número de regresores
            % Entrenar modelo temporal con 'p' regresores en datos de entrenamiento
            [train_inputs, train_outputs] = prepare_time_series_data(datos_train, p);
            X_train = [ones(size(train_inputs, 1), 1), train_inputs];
            theta_temp = X_train \ train_outputs;
            
            % Evaluar en el conjunto de validación
            % Se necesita historial del final del set de entrenamiento para predecir el inicio del de validación
            datos_hist_val = [datos_train(end-p+1:end); datos_val];
            [val_inputs, val_outputs] = prepare_time_series_data(datos_hist_val, p);
            X_val = [ones(size(val_inputs, 1), 1), val_inputs];
            predicciones_val = X_val * theta_temp;
            
            rmse_val = sqrt(mean((predicciones_val - val_outputs).^2));
            errores_val(p) = rmse_val;
        end
        
        % Encontrar el número de regresores que minimizó el error de validación
        [~, p_optimo] = min(errores_val);
        params_optimos.(tipo_actual).regresores(i) = p_optimo;
        fprintf('Número óptimo de regresores para %s de MG %d: %d\n', tipo_actual, i, p_optimo);
    end
end
fprintf('--- Búsqueda de hiperparámetros completada ---\n\n');

%% --- 3. ENTRENAMIENTO DEL MODELO FINAL ---
fprintf('--- Entrenando modelos finales con regresores óptimos ---\n');
modelos = struct();
% Para el modelo final, se usa el conjunto de entrenamiento y validación combinados
datos_completos_train_val = {
    [P_dem_train; P_dem_val], ...
    [P_gen_train; P_gen_val], ...
    [Q_dem_train; Q_dem_val]
};

for i = 1:3 
    for j = 1:length(tipos_de_senal)
        tipo_actual = tipos_de_senal{j};
        datos_actuales = datos_completos_train_val{j};
        num_regresores = params_optimos.(tipo_actual).regresores(i);
        
        fprintf('Entrenando modelo final para %s de Micro-red %d... (Regresores: %d)\n', ...
            tipo_actual, i, num_regresores);
            
        [train_inputs, train_outputs] = prepare_time_series_data(datos_actuales(:, i), num_regresores);
        X = [ones(size(train_inputs, 1), 1), train_inputs];
        theta = X \ train_outputs; 
        
        nombre_modelo = sprintf('mg%d_%s_ar', i, tipo_actual);
        modelos.(nombre_modelo).theta = theta;
        modelos.(nombre_modelo).num_regresores = num_regresores;
    end
end

%% --- 4. Guardar Todos los Modelos ---
if ~exist('models', 'dir'), mkdir('models'); end
save('models/modelos_prediccion_AR.mat', 'modelos');
fprintf('Entrenamiento completado. Modelos guardados en "models/modelos_prediccion_AR.mat"\n\n');

%% --- FASE 5: VISUALIZACIÓN DE PREDICCIONES (SOBRE CONJUNTO DE PRUEBA) ---
fprintf('Generando gráficos de evaluación sobre el conjunto de prueba no visto...\n');
full_ts_data = {[P_dem_train; P_dem_val; P_dem_test], [P_gen_train; P_gen_val; P_gen_test], [Q_dem_train; Q_dem_val; Q_dem_test]};
test_sets = {P_dem_test, P_gen_test, Q_dem_test};
start_idx = size([P_dem_train; P_dem_val], 1) + 1;
end_idx = start_idx + size(P_dem_test, 1) - 1;
ventana_plot = start_idx:end_idx;
tiempo_plot = (0:size(P_dem_test, 1)-1) * (paso_mpc / 60);

plot_prediction_figure(tiempo_plot, P_dem_test, generar_predicciones_ventana(modelos, full_ts_data{1}, 'P_dem', ventana_plot), 'Demanda Eléctrica (Prueba)', 'Potencia [kW]', 'AR');
plot_prediction_figure(tiempo_plot, P_gen_test, generar_predicciones_ventana(modelos, full_ts_data{2}, 'P_gen', ventana_plot), 'Generación Eléctrica (Prueba)', 'Potencia [kW]', 'AR');
plot_prediction_figure(tiempo_plot, Q_dem_test, generar_predicciones_ventana(modelos, full_ts_data{3}, 'Q_dem', ventana_plot), 'Demanda Hídrica (Prueba)', 'Caudal [lt/s]', 'AR');
fprintf('Gráficos de predicción generados y guardados en la carpeta "models".\n');

%% --- Funciones Auxiliares ---
function [inputs, outputs] = prepare_time_series_data(data, num_lags)
    if length(data) <= num_lags
        inputs = []; outputs = [];
        return;
    end
    inputs = zeros(length(data) - num_lags, num_lags);
    outputs = zeros(length(data) - num_lags, 1);
    for i = 1:length(data) - num_lags
        inputs(i, :) = data(i:i+num_lags-1)';
        outputs(i) = data(i+num_lags);
    end
end
function predicciones = generar_predicciones_ventana(modelos, datos_reales, tipo_senal, ventana)
    num_mg = size(datos_reales, 2);
    predicciones = NaN(length(ventana), num_mg);
    for i = 1:num_mg
        nombre_modelo = sprintf('mg%d_%s_ar', i, tipo_senal);
        modelo_actual = modelos.(nombre_modelo);
        theta = modelo_actual.theta;
        num_lags = modelo_actual.num_regresores;
        for k_idx = 1:length(ventana)
            k = ventana(k_idx);
            historia = datos_reales(k-num_lags : k-1, i)';
            entrada = [1, historia];
            predicciones(k_idx, i) = entrada * theta;
        end
    end
end
function plot_prediction_figure(t, datos_reales, datos_predichos, titulo_base, ylabel_text, tipo_modelo)
    fig = figure('Name', [titulo_base ' (' tipo_modelo ')'], 'Position', [100, 100, 900, 700]);
    leyendas = {'Datos reales', 'Predicción'};
    for i = 1:3
        subplot(3, 1, i);
        hold on;
        if strcmp(ylabel_text, 'Potencia [kW]')
            plot(t, datos_reales(:, i) , 'b.', 'DisplayName', leyendas{1});
            plot(t, datos_predichos(:, i) , 'r-', 'LineWidth', 1.5, 'DisplayName', leyendas{2});
        else
            plot(t, datos_reales(:, i), 'b.', 'DisplayName', leyendas{1});
            plot(t, datos_predichos(:, i), 'r-', 'LineWidth', 1.5, 'DisplayName', leyendas{2});
        end
        hold off;
        title(sprintf('%s Micro-red %d', strrep(titulo_base, 'Eléctrica', 'Eléctrica'), i));
        ylabel(ylabel_text);
        grid on; box on;
        legend('Location', 'best');
        if ~isempty(t), xlim([t(1), t(end)]); end
    end
    xlabel('Tiempo [hrs]');
    filename_base = sprintf('models/Prediccion_%s_%s', strrep(titulo_base, ' ', '_'), tipo_modelo);
    saveas(fig, [filename_base '.png']);
    savefig(fig, [filename_base '.fig']);
end