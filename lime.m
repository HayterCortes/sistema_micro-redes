%% SCRIPT DE ANÁLISIS LIME PARA DECISIONES DEL MPC 
% Objetivo: Explicar por qué la microrred "Escuela" decide bombear agua en un
%           instante específico, utilizando LIME de forma robusta.
%
clear; clc; close all;
% --- CONFIGURACIÓN ---
addpath('data', 'results_mpc', 'models', 'utils'); 
num_samples = 10; % Número de muestras para la aproximación LIME
perturbation_strength = 0.1; % Perturbación del 50% sobre las características
fprintf('Iniciando análisis LIME para el sistema de gestión de microrredes...\n\n');

%% PASO 0: CARGAR RESULTADOS E IDENTIFICAR LA INSTANCIA DE INTERÉS
fprintf('--- PASO 0: Identificando la instancia a explicar ---\n');
load('resultados_mpc_3mg_7dias.mat');
% Bloque de código para generar las series de tiempo (idéntico al original)
fprintf('Cargando y pre-procesando perfiles de entrada históricos...\n');
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
d_mr1 = d_mr1_raw(1:min_len); g_mr1 = g_mr1_raw(1:min_len);
d_mr2 = d_mr2_raw(1:min_len); g_mr2 = g_mr2_raw(1:min_len);
d_mr3 = d_mr3_raw(1:min_len); h_mr1 = h_mr1_raw(1:min_len);
h_mr2 = h_mr2_raw(1:min_len); h_mr3 = h_mr3_raw(1:min_len);
g_mr3 = g_mr1 + g_mr2;
paso_mpc = 30;
P_dem_ts = [d_mr1(1:paso_mpc:end), d_mr2(1:paso_mpc:end), d_mr3(1:paso_mpc:end)];
P_gen_ts = [g_mr1(1:paso_mpc:end), g_mr2(1:paso_mpc:end), g_mr3(1:paso_mpc:end)];
Q_dem_ts = [h_mr1(1:paso_mpc:end), h_mr2(1:paso_mpc:end), h_mr3(1:paso_mpc:end)];
fprintf('Series temporales históricas generadas correctamente.\n');
% Lógica de búsqueda 
Ts_mpc = mg(1).Ts_mpc; Ts_sim = mg(1).Ts_sim;
paso_mpc_en_sim = Ts_mpc / Ts_sim;
k_instance_sim = find(Q_p(:, 3) > 0.01, 1, 'first');
if isempty(k_instance_sim), error('No se encontró ningún evento de bombeo significativo.'); end
k_instance_mpc = floor((k_instance_sim - 1) / paso_mpc_en_sim) * paso_mpc_en_sim + 1;
Qp_real = Q_p(k_instance_sim, 3);
if Qp_real < 0.01, error('La lógica de búsqueda falló.'); end
hora_evento = (k_instance_mpc - 1) * Ts_sim / 3600;
fprintf('Instancia encontrada en la hora %.2f (paso de simulación k=%d).\n', hora_evento, k_instance_mpc);
fprintf('Decisión a explicar: Bombeo de la Escuela Q_p = %.4f [L/s]\n\n', Qp_real);

%% PASO 1: AISLAR LA INSTANCIA Y EXTRAER CARACTERÍSTICAS
fprintf('--- PASO 1: Aislando datos y extrayendo características ---\n');
SoC_real = SoC(k_instance_mpc, :); V_tank_real = V_tank(k_instance_mpc, :); V_aq_real = V_aq(k_instance_mpc);
max_lags = mg(1).max_lags_mpc; k_hist_end = round(hora_evento * 60 / paso_mpc);
hist_data.P_dem = P_dem_ts(k_hist_end - max_lags + 1 : k_hist_end, :);
hist_data.P_gen = P_gen_ts(k_hist_end - max_lags + 1 : k_hist_end, :);
hist_data.Q_dem = Q_dem_ts(k_hist_end - max_lags + 1 : k_hist_end, :);
[P_dem_pred, P_gen_pred, Q_dem_pred] = generar_predicciones_AR(mg(1).modelos, hist_data, mg(1).N);
feature_names = {
    'SoC MG1 (%)', 'SoC MG2 (%)', 'SoC MG3 (%)', ...
    'V_tank MG1 (m3)', 'V_tank MG2 (m3)', 'V_tank MG3 (m3)', ...
    'V_aq (m3)', ...
    'P_dem_avg_24h (kW)', 'P_gen_avg_24h (kW)', 'Q_dem_sum_24h (m3)'
    };
X_original = [
    SoC_real * 100, V_tank_real, V_aq_real, ...
    mean(sum(P_dem_pred, 2)), mean(sum(P_gen_pred, 2)), sum(sum(Q_dem_pred, 2)) * Ts_mpc
    ];
fprintf('Vector de características original extraído.\n\n');

%% PASOS 2 y 3: GENERAR MUESTRAS PERTURBADAS Y ETIQUETARLAS
fprintf('--- PASOS 2 y 3: Generando y etiquetando %d muestras ---\n', num_samples);
X_samples = zeros(num_samples, length(X_original));
y_samples = zeros(num_samples, 1);
tic;
for i = 1:num_samples
    perturbation = (1 + perturbation_strength * (2*rand(1, length(X_original)) - 1)); % Perturbación uniforme
    X_perturbed = X_original .* perturbation;
    X_samples(i, :) = X_perturbed;
    
    SoC_p = X_perturbed(1:3) / 100; V_tank_p = X_perturbed(4:6); V_aq_p = X_perturbed(7);
    P_dem_pred_p = P_dem_pred * (X_perturbed(8) / X_original(8));
    P_gen_pred_p = P_gen_pred * (X_perturbed(9) / X_original(9));
    Q_dem_pred_p = Q_dem_pred * (X_perturbed(10) / X_original(10));
    
    u_opt = controlador_mpc(mg, SoC_p, V_tank_p, V_aq_p, P_dem_pred_p, P_gen_pred_p, Q_dem_pred_p);
    
    y_samples(i) = iff(~isempty(u_opt), u_opt.Q_p(3), NaN);
    if mod(i, round(num_samples/10)) == 0, fprintf('...muestra %d/%d etiquetada.\n', i, num_samples); end
end
toc;
fprintf('Etiquetado completado.\n\n');

%% PASO 4: ENTRENAR EL MODELO LINEAL SUSTITUTO
fprintf('--- PASO 4: Entrenando el modelo lineal explicativo ---\n');

% --- INICIO: NUEVO BLOQUE DE ESTANDARIZACIÓN (Z-SCORE) ---
% 1. Calcular media y desviación estándar del vecindario de muestras
mean_samples = mean(X_samples, 1);
std_samples = std(X_samples, 1) + 1e-8; % Se añade epsilon para evitar división por cero

% 2. Estandarizar tanto las muestras como la instancia original
X_samples_std = (X_samples - mean_samples) ./ std_samples;
X_original_std = (X_original - mean_samples) ./ std_samples;
fprintf('Características estandarizadas para un análisis imparcial.\n');
% --- FIN: NUEVO BLOQUE DE ESTANDARIZACIÓN ---

% Calcular la distancia euclidiana USANDO DATOS ESTANDARIZADOS
diff_sq = (X_samples_std - X_original_std).^2; 
distances = sqrt(sum(diff_sq, 2));

% Calcular pesos
kernel_width = 0.75 * std(distances);
weights = exp(-distances.^2 / kernel_width^2);

% Filtrar muestras inválidas (donde el MPC pudo haber fallado)
valid_samples = ~isnan(y_samples);
X_samples_std_valid = X_samples_std(valid_samples, :); % --- MODIFICADO: Usar datos estandarizados
y_samples_valid = y_samples(valid_samples);
weights_valid = weights(valid_samples);

% --- Bloque de Regresión Lineal Ponderada ---
X_design = [ones(size(X_samples_std_valid, 1), 1), X_samples_std_valid]; % --- MODIFICADO
W_sqrt = diag(sqrt(weights_valid));
X_weighted = W_sqrt * X_design;
y_weighted = W_sqrt * y_samples_valid;
coeffs_vector = X_weighted \ y_weighted;
fprintf('Modelo sustituto entrenado con datos estandarizados.\n\n');

%% PASO 5: INTERPRETAR LOS PESOS DEL MODELO
fprintf('--- PASO 5: EXPLICACIÓN DE LA DECISIÓN DE BOMBEO ---\n\n');
[~, sorted_idx] = sort(abs(coeffs_vector(2:end)), 'descend');
fprintf('La decisión de bombear %.4f [L/s] se debió a la siguiente combinación de factores:\n', Qp_real);
fprintf('----------------------------------------------------------------------------------\n');
fprintf('%-25s | %-20s | %-20s \n', 'Característica', 'Coeficiente (w)', 'Influencia');
fprintf('----------------------------------------------------------------------------------\n');
for i = 1:length(sorted_idx)
    idx = sorted_idx(i);
    feature_name = feature_names{idx};
    weight_val = coeffs_vector(idx + 1);
    
    if weight_val > 1e-3, influencia = 'AUMENTA el bombeo';
    elseif weight_val < -1e-3, influencia = 'REDUCE el bombeo';
    else, influencia = 'Neutra'; end
    
    fprintf('%-25s | %-20.4f | %-20s \n', feature_name, weight_val, influencia);
end
fprintf('----------------------------------------------------------------------------------\n');